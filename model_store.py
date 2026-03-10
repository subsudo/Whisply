from __future__ import annotations

from collections.abc import Callable
import io
import json
import logging
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time

from backends import create_backend


log = logging.getLogger(__name__)


class _NullWriter(io.TextIOBase):
    """Minimal file-like sink for PyInstaller --windowed mode (sys.stdout/stderr are None)."""

    encoding = "utf-8"

    def write(self, s: str) -> int:  # noqa: ANN401
        return len(s) if isinstance(s, str) else 0

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _ensure_safe_stdio() -> None:
    """Patch sys.stdout/stderr if they are None (PyInstaller --windowed / --noconsole)."""
    if sys.stdout is None:
        sys.stdout = _NullWriter()  # type: ignore[assignment]
    if sys.stderr is None:
        sys.stderr = _NullWriter()  # type: ignore[assignment]

SUPPORTED_MODELS = ("small", "medium", "large-v3", "large-v3-turbo")

_MODEL_HINTS = {
    "small": ("faster-whisper-small",),
    "medium": ("faster-whisper-medium",),
    "large-v3": ("faster-whisper-large-v3",),
    "large-v3-turbo": ("faster-whisper-large-v3-turbo",),
}

_MODEL_REPOS = {
    "small": ("Systran/faster-whisper-small",),
    "medium": ("Systran/faster-whisper-medium",),
    "large-v3": ("Systran/faster-whisper-large-v3",),
    "large-v3-turbo": (
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "dropbox-dash/faster-whisper-large-v3-turbo",
    ),
}

_MODEL_REQUIRED_FILES = ("model.bin", "config.json", "tokenizer.json")
_MODEL_OPTIONAL_FILES = ("preprocessor_config.json",)
_VOCABULARY_CANDIDATES = ("vocabulary.txt", "vocabulary.json", "vocabulary.bin")


def _normalize_model(model: str) -> str:
    value = str(model).strip().lower()
    if value not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model}'")
    return value


def _model_store_path(download_root: str | Path) -> Path:
    return Path(download_root) / "installed_models.json"


def _flat_models_root(download_root: str | Path) -> Path:
    return Path(download_root) / "whisply-flat"


def _repo_local_name(repo_id: str) -> str:
    return str(repo_id).split("/")[-1].strip().lower()


def _find_existing_flat_dir(model: str, download_root: str | Path) -> Path | None:
    root = _flat_models_root(download_root)
    hints = tuple(x.lower() for x in _MODEL_HINTS.get(model, ()))
    if not _safe_is_dir(root):
        return None
    for hint in hints:
        candidate = root / hint
        if not _safe_is_dir(candidate):
            continue
        if _flat_dir_has_required_files(candidate):
            return candidate
    return None


def _flat_dir_has_required_files(model_dir: Path) -> bool:
    if not _safe_is_dir(model_dir):
        return False
    if not all(_safe_is_file(model_dir / name) for name in _MODEL_REQUIRED_FILES):
        return False
    return _find_downloaded_vocabulary_file(model_dir) is not None


def _find_downloaded_vocabulary_file(model_dir: Path) -> Path | None:
    if not _safe_is_dir(model_dir):
        return None
    try:
        for candidate in model_dir.iterdir():
            if not _safe_is_file(candidate):
                continue
            if candidate.name.lower().startswith("vocabulary."):
                return candidate
    except OSError:
        return None
    return None


def _resolve_repo_files(repo_id: str) -> tuple[list[str], str | None]:
    try:
        from huggingface_hub import HfApi  # type: ignore

        files = HfApi().list_repo_files(repo_id)
    except Exception as exc:
        log.debug("Could not list repo files for %s: %s", repo_id, exc)
        files = []

    vocabulary_name: str | None = None
    lower_map = {str(name).lower(): str(name) for name in files}
    for candidate in _VOCABULARY_CANDIDATES:
        resolved = lower_map.get(candidate.lower())
        if resolved:
            vocabulary_name = resolved
            break

    files_to_download = list(_MODEL_REQUIRED_FILES)
    for name in _MODEL_OPTIONAL_FILES:
        resolved = lower_map.get(name.lower())
        if resolved:
            files_to_download.append(resolved)
    if vocabulary_name:
        files_to_download.append(vocabulary_name)
    else:
        files_to_download.append(_VOCABULARY_CANDIDATES[0])

    return files_to_download, vocabulary_name


def get_model_path(model: str, download_root: str | Path) -> str | None:
    normalized = _normalize_model(model)
    existing = _find_existing_flat_dir(normalized, download_root)
    if existing is None:
        return None
    return str(existing)


def _empty_status() -> dict[str, bool]:
    return {name: False for name in SUPPORTED_MODELS}


def _load_store(download_root: str | Path) -> dict[str, bool] | None:
    path = _model_store_path(download_root)
    if not _safe_exists(path):
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        log.warning("Could not parse model status store: %s", path)
        return None
    status = _empty_status()
    if isinstance(payload, dict):
        for name in SUPPORTED_MODELS:
            status[name] = bool(payload.get(name, False))
    return status


def _save_store(download_root: str | Path, status: dict[str, bool]) -> None:
    path = _model_store_path(download_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, indent=2, sort_keys=True), encoding="utf-8")


def _scan_download_root(download_root: str | Path) -> dict[str, bool]:
    root = Path(download_root)
    scanned = _empty_status()
    if not _safe_exists(root):
        return scanned

    for model in SUPPORTED_MODELS:
        scanned[model] = get_model_path(model, root) is not None

    names: list[str] = []
    try:
        for dirpath, _, _ in os.walk(root, followlinks=False):
            try:
                names.append(Path(dirpath).name.lower())
            except Exception:
                continue
    except OSError:
        return scanned
    for model, hints in _MODEL_HINTS.items():
        scanned[model] = bool(scanned.get(model, False)) or any(
            any(folder == hint or folder.startswith(hint + "-") for hint in hints)
            for folder in names
        )
    return scanned


def get_model_status(download_root: str | Path) -> dict[str, bool]:
    loaded = _load_store(download_root)
    if loaded is not None:
        return loaded

    rebuilt = _scan_download_root(download_root)
    _save_store(download_root, rebuilt)
    return rebuilt


def refresh_model_status(download_root: str | Path) -> dict[str, bool]:
    scanned = _scan_download_root(download_root)
    loaded = _load_store(download_root) or _empty_status()
    merged = {name: bool(scanned.get(name, False) or loaded.get(name, False)) for name in SUPPORTED_MODELS}
    if merged != loaded:
        _save_store(download_root, merged)
    return merged


def mark_installed(model: str, download_root: str | Path) -> None:
    normalized = _normalize_model(model)
    status = get_model_status(download_root)
    if status.get(normalized):
        return
    status[normalized] = True
    _save_store(download_root, status)


def _safe_dir_size(root: Path) -> int:
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


def _safe_exists(path: Path) -> bool:
    try:
        return path.exists()
    except OSError:
        return False


def _safe_is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def _safe_is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False


def _safe_stat_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return -1


def _estimate_model_download_bytes(model_size: str) -> int:
    estimates = {
        "small": 550 * 1024 * 1024,
        "medium": 1600 * 1024 * 1024,
        "large-v3": 3200 * 1024 * 1024,
        "large-v3-turbo": 1700 * 1024 * 1024,
    }
    return int(estimates.get(model_size, 0))


def _materialise_real_file(file_path: Path) -> bool:
    """If *file_path* is a symlink or junction, replace it with a copy of the target.

    Returns True if the file is usable afterwards, False otherwise.
    """
    try:
        # Quick check: can we open the file for reading?
        with open(file_path, "rb") as fh:
            fh.read(4)
        return True
    except OSError:
        pass

    # The file exists as a symlink/junction but can't be read (untrusted reparse point).
    # Try to resolve the link target and copy the real blob file.
    real_target: Path | None = None
    try:
        raw_target = os.readlink(file_path)
        candidate = Path(raw_target)
        if not candidate.is_absolute():
            candidate = file_path.parent / candidate
        candidate = candidate.resolve(strict=False)
        if candidate.is_file():
            real_target = candidate
    except (OSError, ValueError):
        pass

    if real_target is None:
        # Try to find the blob via the HF cache directory structure.
        # Typical layout: models--<org>--<name>/blobs/<hash>
        #                 models--<org>--<name>/snapshots/<rev>/<filename> -> ../../blobs/<hash>
        try:
            snapshot_dir = file_path.parent
            repo_cache = snapshot_dir.parent.parent  # up from snapshots/<rev>/
            blobs_dir = repo_cache / "blobs"
            if blobs_dir.is_dir():
                for blob in blobs_dir.iterdir():
                    try:
                        if blob.is_file() and blob.stat().st_size > 0:
                            # For model.bin, pick the largest blob; for small files,
                            # match by name is unreliable so only large blobs qualify.
                            if file_path.name == "model.bin":
                                if blob.stat().st_size > 10 * 1024 * 1024:
                                    real_target = blob
                                    break
                            else:
                                # Small config files: try to read the first blob
                                # that has sensible size (<10 MB).
                                if blob.stat().st_size < 10 * 1024 * 1024:
                                    real_target = blob
                                    break
                    except OSError:
                        continue
        except OSError:
            pass

    if real_target is None:
        return False

    try:
        tmp = file_path.with_suffix(".tmp_copy")
        shutil.copy2(str(real_target), str(tmp))
        try:
            file_path.unlink()
        except OSError:
            pass
        tmp.rename(file_path)
        log.info("Materialised real file from blob: %s (source=%s)", file_path, real_target)
        return True
    except OSError as exc:
        log.warning("Failed to materialise %s from blob: %s", file_path, exc)
        return False


def _download_model_flat(model: str, download_root: str | Path) -> Path:
    """Download model files into a flat directory without symlinks.

    Uses per-file ``hf_hub_download`` instead of ``snapshot_download`` to
    avoid tqdm crashes in PyInstaller --windowed mode and to work around
    the symlink/reparse-point issue on locked-down Windows machines.
    """
    normalized = _normalize_model(model)
    existing = _find_existing_flat_dir(normalized, download_root)
    if existing is not None:
        return existing

    _ensure_safe_stdio()

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub unavailable: {exc}") from exc

    repos = _MODEL_REPOS.get(normalized) or ()
    if not repos:
        raise RuntimeError(f"No Hugging Face repo mapping for model '{normalized}'")

    flat_root = _flat_models_root(download_root)
    flat_root.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None

    for repo_id in repos:
        local_dir = flat_root / _repo_local_name(repo_id)
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            files_to_download, vocabulary_name = _resolve_repo_files(repo_id)
            log.info(
                "Resolved model file set: model=%s repo=%s vocabulary=%s files=%s",
                normalized,
                repo_id,
                vocabulary_name or "fallback",
                files_to_download,
            )
            for filename in files_to_download:
                try:
                    hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        local_dir=str(local_dir),
                    )
                    # If hf_hub_download created a symlink instead of a copy,
                    # materialise it into a real file now.
                    dest = local_dir / filename
                    if _safe_exists(dest) and not _materialise_real_file(dest):
                        log.warning("Downloaded %s but file is not readable.", dest)
                        if filename in _MODEL_REQUIRED_FILES or filename == vocabulary_name:
                            raise RuntimeError(
                                f"Downloaded '{filename}' is not readable (reparse point?): {dest}"
                            )
                except Exception as file_exc:
                    if filename in _MODEL_REQUIRED_FILES or filename == vocabulary_name:
                        raise
                    log.debug(
                        "Optional model file '%s' not available from %s: %s",
                        filename,
                        repo_id,
                        file_exc,
                    )
            if _flat_dir_has_required_files(local_dir):
                log.info(
                    "Flat model download ready: model=%s repo=%s dir=%s",
                    normalized,
                    repo_id,
                    local_dir,
                )
                return local_dir
            last_exc = RuntimeError(f"Flat model folder missing required files: {local_dir}")
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Flat model download failed: model=%s repo=%s error=%s",
                normalized,
                repo_id,
                exc,
            )
            continue

    if last_exc is None:
        raise RuntimeError(f"Flat model download failed for '{normalized}'")
    raise RuntimeError(f"Flat model download failed for '{normalized}': {last_exc}") from last_exc


def _start_prefetch_progress_monitor(
    model_size: str,
    download_root: str | Path,
    stop_event: threading.Event,
    progress_cb: Callable[[int], None],
) -> threading.Thread:
    root = Path(download_root)
    estimated_total = _estimate_model_download_bytes(model_size)
    baseline_size = _safe_dir_size(root) if root.exists() else 0

    def _worker() -> None:
        progress = 0
        last_size_progress = -1
        while not stop_event.wait(0.18):
            if estimated_total > 0 and root.exists():
                current_size = _safe_dir_size(root)
                downloaded = max(0, current_size - baseline_size)
                size_progress = int(min(94, (downloaded * 100) // estimated_total))
                if size_progress > last_size_progress:
                    last_size_progress = size_progress
                progress = max(progress, max(2, size_progress))
            else:
                progress = min(90, progress + 1)

            try:
                progress_cb(progress)
            except Exception:
                log.exception("Model prefetch progress callback failed.")
                return

    thread = threading.Thread(
        target=_worker,
        name="model-prefetch-progress",
        daemon=True,
    )
    thread.start()
    return thread


def _is_missing_model_bin_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "unable to open file 'model.bin'" in text


def _is_untrusted_reparse_error(exc: Exception) -> bool:
    text = str(exc).lower()
    if (
        "winerror 448" in text
        or "untrusted reparse point" in text
        or "nicht vertrauenswürdigen bereitstellungspunkt" in text
    ):
        return True
    # Also check chained exceptions (__context__ / __cause__)
    chained = getattr(exc, "__context__", None) or getattr(exc, "__cause__", None)
    if chained is not None and chained is not exc:
        return _is_untrusted_reparse_error(chained)
    return False


def _probe_reparse_on_model_bin(model_dir: Path | None) -> bool:
    """Try to stat model.bin directly; return True if WinError 448 is the cause."""
    if model_dir is None:
        return False
    model_bin = model_dir / "model.bin"
    try:
        model_bin.stat()
        return False
    except OSError as e:
        return _is_untrusted_reparse_error(e)
    except Exception:
        return False


def _extract_model_path_from_error(exc: Exception) -> Path | None:
    text = str(exc)
    match = re.search(r"in model '([^']+)'", text)
    if not match:
        return None
    try:
        return Path(match.group(1))
    except Exception:
        return None


def _log_snapshot_state(model_dir: Path | None) -> None:
    if model_dir is None:
        log.warning("Missing model.bin error did not include a model directory path.")
        return
    if not _safe_exists(model_dir):
        log.warning("Model snapshot directory does not exist: %s", model_dir)
        return

    model_bin = model_dir / "model.bin"
    model_bin_exists = _safe_exists(model_bin)
    model_bin_size = _safe_stat_size(model_bin) if model_bin_exists else -1
    log.warning(
        "Model snapshot state: dir=%s model.bin_exists=%s model.bin_size=%s",
        model_dir,
        model_bin_exists,
        model_bin_size,
    )
    try:
        entries: list[str] = []
        for item in sorted(model_dir.iterdir(), key=lambda p: p.name.lower())[:20]:
            if _safe_is_file(item):
                entries.append(f"{item.name}:{_safe_stat_size(item)}")
            else:
                entries.append(f"{item.name}/")
        log.warning("Model snapshot entries (first 20): %s", entries)
    except OSError:
        log.warning("Could not inspect snapshot directory entries (OS error): %s", model_dir)
    except Exception:
        log.exception("Could not inspect snapshot directory entries: %s", model_dir)


def _wait_for_model_bin_stable(
    model_dir: Path | None,
    timeout_sec: float = 18.0,
    poll_sec: float = 0.5,
) -> bool:
    if model_dir is None:
        return False

    model_bin = model_dir / "model.bin"
    deadline = time.monotonic() + max(1.0, timeout_sec)
    last_size = -1
    stable_hits = 0

    while time.monotonic() < deadline:
        if _safe_exists(model_bin):
            try:
                size = int(model_bin.stat().st_size)
            except OSError:
                size = -1
            if size > 0:
                if size == last_size:
                    stable_hits += 1
                else:
                    stable_hits = 0
                last_size = size
                if stable_hits >= 2:
                    log.info(
                        "model.bin became stable after wait: %s (size=%s)",
                        model_bin,
                        size,
                    )
                    return True
        time.sleep(max(0.1, poll_sec))

    log.warning("model.bin did not become stable within %.1fs: %s", timeout_sec, model_bin)
    return False


def _cleanup_broken_model_cache(model: str, download_root: str | Path, exc: Exception) -> None:
    root = Path(download_root)
    if not _safe_exists(root):
        return

    targets: list[Path] = []
    parsed_path = _extract_model_path_from_error(exc)
    if parsed_path is not None:
        # snapshot dir -> remove corresponding repo cache root
        repo_root = parsed_path
        for _ in range(2):
            if repo_root.name.lower() == "snapshots":
                repo_root = repo_root.parent
                break
            repo_root = repo_root.parent
        if _safe_exists(repo_root) and root in repo_root.parents:
            targets.append(repo_root)

    hints = _MODEL_HINTS.get(model, ())
    try:
        for candidate in root.glob("models--*"):
            lower_name = candidate.name.lower()
            if any(hint in lower_name for hint in hints):
                targets.append(candidate)
    except OSError:
        pass

    unique_targets: list[Path] = []
    seen: set[str] = set()
    for target in targets:
        key = str(target).lower()
        if key in seen:
            continue
        seen.add(key)
        unique_targets.append(target)

    for target in unique_targets:
        try:
            shutil.rmtree(target, ignore_errors=True)
            log.warning("Removed broken model cache: %s", target)
        except Exception:
            log.exception("Failed to remove broken model cache: %s", target)


def ensure_model_installed(
    model: str,
    backend_hint: str,
    download_root: str | Path,
    progress_cb: Callable[[int], None] | None = None,
) -> tuple[bool, str]:
    _ensure_safe_stdio()
    normalized = _normalize_model(model)
    status = get_model_status(download_root)
    if status.get(normalized):
        if progress_cb:
            progress_cb(100)
        return True, "already_installed"

    if progress_cb:
        progress_cb(0)

    last_exc: Exception | None = None
    for attempt in (1, 2, 3):
        progress_stop = threading.Event()
        progress_thread: threading.Thread | None = None
        try:
            backend = create_backend(backend_hint, "auto", download_root)
            if progress_cb:
                progress_thread = _start_prefetch_progress_monitor(
                    model_size=normalized,
                    download_root=download_root,
                    stop_event=progress_stop,
                    progress_cb=progress_cb,
                )
            try:
                _download_model_flat(normalized, download_root)
            except Exception as flat_exc:
                # Keep compatibility: backend can still perform classic model load/download.
                log.warning(
                    "Flat model download unavailable for '%s', using backend fallback: %s",
                    normalized,
                    flat_exc,
                )
            backend.load_model(normalized)
            progress_stop.set()
            if progress_thread is not None:
                progress_thread.join(timeout=0.6)
            if progress_cb:
                progress_cb(96)
            backend.unload_model()
            mark_installed(normalized, download_root)
            if progress_cb:
                progress_cb(100)
            return True, "ok"
        except Exception as exc:
            last_exc = exc
            log.warning(
                "Model prefetch attempt %s for '%s' failed: %s",
                attempt,
                normalized,
                exc,
            )
            progress_stop.set()
            if progress_thread is not None:
                try:
                    progress_thread.join(timeout=0.4)
                except Exception:
                    pass

            missing_model_bin = _is_missing_model_bin_error(exc)
            untrusted_reparse = _is_untrusted_reparse_error(exc)

            if missing_model_bin or untrusted_reparse:
                snapshot_dir = _extract_model_path_from_error(exc)
                # After safe-wrapper fix, the RuntimeError from faster_whisper
                # no longer chains with OSError.  Probe the file directly to
                # detect the reparse-point issue that Windows policies cause.
                if missing_model_bin and not untrusted_reparse:
                    untrusted_reparse = _probe_reparse_on_model_bin(snapshot_dir)
                _log_snapshot_state(snapshot_dir)
                wait_timeout = 40.0 if untrusted_reparse else 18.0
                if _wait_for_model_bin_stable(snapshot_dir, timeout_sec=wait_timeout, poll_sec=0.5):
                    log.warning(
                        "Model prefetch attempt %s for %s: retrying after model.bin wait (no cleanup).",
                        attempt,
                        normalized,
                    )
                    if progress_cb:
                        progress_cb(0)
                    continue

                if untrusted_reparse and attempt < 3:
                    log.warning(
                        "Model prefetch attempt %s for %s failed due untrusted reparse point; retrying without cleanup.",
                        attempt,
                        normalized,
                    )
                    time.sleep(1.5 * attempt)
                    if progress_cb:
                        progress_cb(0)
                    continue

                if not missing_model_bin or attempt != 1:
                    log.exception("Model prefetch failed for %s", normalized)
                    return False, str(exc)

                log.warning(
                    "Model prefetch attempt %s for %s failed with missing model.bin. Cleaning cache and retrying.",
                    attempt,
                    normalized,
                )
                _cleanup_broken_model_cache(normalized, download_root, exc)
                if progress_cb:
                    progress_cb(0)
                continue

            log.exception("Model prefetch failed for %s", normalized)
            return False, str(exc)

    if last_exc is None:
        return False, "unknown prefetch error"
    return False, str(last_exc)
