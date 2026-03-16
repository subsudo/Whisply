from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
import logging
import math
from pathlib import Path
import threading
import time
from typing import cast

import numpy as np

from backends import create_backend
from model_store import get_model_status, mark_installed

log = logging.getLogger(__name__)


class Transcriber:
    def __init__(
        self,
        backend_name: str = "auto",
        model_size: str = "medium",
        language: str = "auto",
        compute_type: str = "auto",
        beam_size: int = 5,
        download_root: str | Path | None = None,
        on_cuda_fallback: Callable[[str], None] | None = None,
        on_model_load_start: Callable[[str, str], None] | None = None,
        on_model_load_progress: Callable[[int], None] | None = None,
        on_model_load_done: Callable[[], None] | None = None,
    ) -> None:
        self.download_root = str(download_root) if download_root else None
        self.compute_type = compute_type
        self.backend = create_backend(backend_name, compute_type, self.download_root)
        self.backend_name = backend_name
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._model_loaded = False
        self._last_used_monotonic = time.monotonic()
        self._on_cuda_fallback = on_cuda_fallback
        self._on_model_load_start = on_model_load_start
        self._on_model_load_progress = on_model_load_progress
        self._on_model_load_done = on_model_load_done
        self._warmup_duration_estimates: dict[tuple[str, str], float] = {}

    def transcribe_async(self, audio: np.ndarray) -> Future[str]:
        return self._executor.submit(self._transcribe_with_lazy_load, audio)

    def _is_cuda_backend(self) -> bool:
        return self.backend.get_device_info().lower().startswith("cuda")

    def _fallback_from_cuda_to_cpu_locked(self, reason: Exception) -> None:
        fallback_model = "medium" if self.model_size in {"large-v3", "large-v3-turbo"} else self.model_size
        reason_text = str(reason)
        log.warning(
            "CUDA unavailable during transcription (%s). Falling back to CPU with model='%s'.",
            reason_text,
            fallback_model,
        )
        self.backend = create_backend("cpu", "auto", self.download_root)
        self.backend_name = "cpu"
        self.model_size = fallback_model
        self._model_loaded = False
        self.backend.load_model(self.model_size)
        self._model_loaded = True
        if self.download_root:
            try:
                mark_installed(self.model_size, self.download_root)
            except Exception:
                log.exception("Failed to update installed model store after CUDA fallback load.")
        if self._on_cuda_fallback:
            try:
                self._on_cuda_fallback(reason_text)
            except Exception:
                log.exception("on_cuda_fallback callback failed")

    def _transcribe_with_lazy_load(self, audio: np.ndarray) -> str:
        with self._lock:
            if not self._model_loaded:
                log.info("Loading model '%s' ...", self.model_size)
                load_mode = self._infer_model_load_mode(self.model_size)
                load_started_at = time.monotonic()
                if self._on_model_load_start:
                    try:
                        self._on_model_load_start(self.model_size, load_mode)
                    except Exception:
                        log.exception("on_model_load_start callback failed")
                progress_stop = threading.Event()
                load_complete = threading.Event()
                progress_thread: threading.Thread | None = None
                try:
                    if self._on_model_load_progress:
                        try:
                            self._on_model_load_progress(0)
                        except Exception:
                            log.exception("on_model_load_progress callback failed")
                        if load_mode == "download":
                            progress_thread = self._start_model_load_progress_monitor(
                                self.model_size,
                                progress_stop,
                            )
                        else:
                            progress_thread = self._start_model_warmup_progress_monitor(
                                progress_stop,
                                load_complete,
                                self._estimated_warmup_duration_sec(self.model_size),
                            )
                    self.backend.load_model(self.model_size)
                except Exception as exc:
                    if self._is_cuda_backend():
                        self._fallback_from_cuda_to_cpu_locked(exc)
                    else:
                        raise
                finally:
                    observed_load_sec = max(0.05, time.monotonic() - load_started_at)
                    if load_mode == "warmup":
                        load_complete.set()
                    else:
                        progress_stop.set()
                    if progress_thread is not None:
                        progress_thread.join(timeout=1.2 if load_mode == "warmup" else 0.6)
                        if progress_thread.is_alive():
                            progress_stop.set()
                            progress_thread.join(timeout=0.3)
                    if load_mode == "warmup":
                        self._record_warmup_duration(self.model_size, observed_load_sec)
                    if self._on_model_load_progress:
                        try:
                            self._on_model_load_progress(100)
                        except Exception:
                            log.exception("on_model_load_progress callback failed")
                    if self._on_model_load_done:
                        try:
                            self._on_model_load_done()
                        except Exception:
                            log.exception("on_model_load_done callback failed")
                self._model_loaded = True
                if self.download_root:
                    try:
                        mark_installed(self.model_size, self.download_root)
                    except Exception:
                        log.exception("Failed to update installed model store after model load.")
                log.info("Model loaded.")
            log.info(
                "Transcribing with model=%s language=%s beam_size=%s backend=%s",
                self.model_size,
                self.language,
                self.beam_size,
                self.backend.get_device_info(),
            )
            try:
                text = self.backend.transcribe(audio, self.language, self.beam_size)
            except Exception as exc:
                if self._is_cuda_backend():
                    self._fallback_from_cuda_to_cpu_locked(exc)
                    log.info(
                        "Retrying transcription after CUDA fallback with backend=%s",
                        self.backend.get_device_info(),
                    )
                    text = self.backend.transcribe(audio, self.language, self.beam_size)
                else:
                    raise
            self._last_used_monotonic = time.monotonic()
            return text

    def _start_model_load_progress_monitor(
        self,
        model_size: str,
        stop_event: threading.Event,
    ) -> threading.Thread | None:
        if not self.download_root:
            return None

        download_root = Path(self.download_root)
        if not download_root.exists():
            return None

        estimated_total = self._estimate_model_download_bytes(model_size)
        if estimated_total <= 0:
            return None

        baseline_size = self._safe_dir_size(download_root)

        def _worker() -> None:
            last_progress = -1
            while not stop_event.wait(0.25):
                current_size = self._safe_dir_size(download_root)
                downloaded = max(0, current_size - baseline_size)
                progress = int(min(95, (downloaded * 100) // estimated_total))
                if progress <= last_progress:
                    continue
                last_progress = progress
                try:
                    cast(Callable[[int], None], self._on_model_load_progress)(progress)
                except Exception:
                    log.exception("on_model_load_progress callback failed")
                    return

        thread = threading.Thread(
            target=_worker,
            name="model-load-progress",
            daemon=True,
        )
        thread.start()
        return thread

    def _start_model_warmup_progress_monitor(
        self,
        stop_event: threading.Event,
        load_complete_event: threading.Event,
        estimated_duration_sec: float,
    ) -> threading.Thread:
        def _worker() -> None:
            start = time.monotonic()
            duration = max(0.35, float(estimated_duration_sec))
            last_progress = -1
            tick_sec = 0.04
            linear_cap = 88.0
            overrun_cap = 97.0

            while not stop_event.wait(tick_sec):
                elapsed = max(0.0, time.monotonic() - start)
                if elapsed <= duration:
                    progress = int((elapsed / duration) * linear_cap)
                else:
                    overrun_elapsed = elapsed - duration
                    tail_ratio = 1.0 - math.exp(-overrun_elapsed / max(0.28, duration * 0.35))
                    progress = int(linear_cap + ((overrun_cap - linear_cap) * tail_ratio))
                progress = min(int(overrun_cap), max(0, progress))
                if progress == last_progress:
                    if load_complete_event.is_set():
                        break
                    continue
                try:
                    cast(Callable[[int], None], self._on_model_load_progress)(progress)
                except Exception:
                    log.exception("on_model_load_progress callback failed")
                    return
                last_progress = progress
                if load_complete_event.is_set():
                    break

            if stop_event.is_set():
                return

            start_progress = max(0, min(last_progress, 99))
            remaining = max(0, 100 - start_progress)
            if remaining <= 0:
                return

            finish_sec = min(0.28, max(0.14, 0.10 + (remaining / 100.0) * 0.18))
            finish_started_at = time.monotonic()

            while not stop_event.wait(0.03):
                finish_elapsed = max(0.0, time.monotonic() - finish_started_at)
                ratio = min(1.0, finish_elapsed / finish_sec)
                progress = min(100, int(round(start_progress + (remaining * ratio))))
                if progress == last_progress:
                    if ratio >= 1.0:
                        return
                    continue
                try:
                    cast(Callable[[int], None], self._on_model_load_progress)(progress)
                except Exception:
                    log.exception("on_model_load_progress callback failed")
                    return
                last_progress = progress
                if ratio >= 1.0:
                    return

        thread = threading.Thread(
            target=_worker,
            name="model-warmup-progress",
            daemon=True,
        )
        thread.start()
        return thread

    def _warmup_profile_key(self, model_size: str) -> tuple[str, str]:
        try:
            device_info = self.backend.get_device_info().lower()
        except Exception:
            device_info = str(self.backend_name).lower()
        profile = "cuda" if device_info.startswith("cuda") else "cpu"
        return profile, str(model_size).strip().lower()

    def _estimated_warmup_duration_sec(self, model_size: str) -> float:
        key = self._warmup_profile_key(model_size)
        cached = self._warmup_duration_estimates.get(key)
        if cached is not None and cached > 0:
            return float(cached)

        defaults = {
            ("cpu", "small"): 0.9,
            ("cpu", "medium"): 1.4,
            ("cpu", "large-v3"): 2.2,
            ("cpu", "large-v3-turbo"): 1.8,
            ("cuda", "small"): 0.55,
            ("cuda", "medium"): 0.85,
            ("cuda", "large-v3"): 1.2,
            ("cuda", "large-v3-turbo"): 1.0,
        }
        return float(defaults.get(key, 1.8))

    def _record_warmup_duration(self, model_size: str, observed_sec: float) -> None:
        key = self._warmup_profile_key(model_size)
        clamped = min(8.0, max(0.35, float(observed_sec)))
        previous = self._warmup_duration_estimates.get(key)
        if previous is None:
            updated = clamped
        else:
            updated = (previous * 0.4) + (clamped * 0.6)
        self._warmup_duration_estimates[key] = updated
        log.debug(
            "Warmup duration estimate updated for %s/%s: observed=%.3fs estimate=%.3fs",
            key[0],
            key[1],
            clamped,
            updated,
        )

    def _infer_model_load_mode(self, model_size: str) -> str:
        if not self.download_root:
            return "warmup"
        try:
            status = get_model_status(self.download_root)
            if bool(status.get(model_size, False)):
                return "warmup"
        except Exception:
            log.exception("Failed to infer model load mode; falling back to download mode.")
        return "download"

    def _safe_dir_size(self, root: Path) -> int:
        total = 0
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                try:
                    total += path.stat().st_size
                except OSError:
                    continue
        except OSError:
            return total
        return total

    def _estimate_model_download_bytes(self, model_size: str) -> int:
        estimates = {
            "small": 550 * 1024 * 1024,
            "medium": 1600 * 1024 * 1024,
            "large-v3": 3200 * 1024 * 1024,
            "large-v3-turbo": 1700 * 1024 * 1024,
        }
        return int(estimates.get(model_size, 0))

    def set_model(self, model_size: str) -> None:
        if model_size == self.model_size:
            return
        with self._lock:
            previous_model = self.model_size
            self.model_size = model_size
            if self._model_loaded:
                self.backend.unload_model()
                self._model_loaded = False
                log.info(
                    "Model switched from %s to %s. Current model unloaded; next transcription will lazy-load the new model.",
                    previous_model,
                    model_size,
                )
            self._last_used_monotonic = time.monotonic()

    def set_language(self, language: str) -> None:
        self.language = language

    def set_backend(
        self,
        backend_name: str,
        compute_type: str = "auto",
        download_root: str | Path | None = None,
    ) -> None:
        with self._lock:
            if self._model_loaded:
                self.backend.unload_model()
            if download_root is not None:
                self.download_root = str(download_root)
            self.backend = create_backend(backend_name, compute_type, self.download_root)
            self.backend_name = backend_name
            self.compute_type = compute_type
            self._model_loaded = False
            self._last_used_monotonic = time.monotonic()

    def unload_if_idle(self, idle_sec: int) -> bool:
        if idle_sec <= 0:
            return False
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if not self._model_loaded:
                return False
            idle = time.monotonic() - self._last_used_monotonic
            if idle < float(idle_sec):
                return False
            self.backend.unload_model()
            self._model_loaded = False
            log.info("Model unloaded after %.1fs idle (threshold=%ss).", idle, idle_sec)
            return True
        finally:
            self._lock.release()

    def unload_now(self) -> bool:
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if not self._model_loaded:
                return False
            self.backend.unload_model()
            self._model_loaded = False
            log.info("Model unloaded manually.")
            return True
        finally:
            self._lock.release()

    def is_model_loaded(self) -> bool:
        with self._lock:
            return self._model_loaded

    def device_info(self) -> str:
        return self.backend.get_device_info()

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        with self._lock:
            if self._model_loaded:
                self.backend.unload_model()
                self._model_loaded = False
