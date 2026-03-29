from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np
import sounddevice as sd

log = logging.getLogger(__name__)

# ── FFT band definitions ──────────────────────────────────────────────────────
# 12 log-spaced frequency bands for speech visualisation (≈ 80 Hz – 8 kHz).
# Computed for a 512-point FFT at 16 kHz sample rate:  bin_idx = hz / 31.25
# Edges in Hz:  ~80, 125, 188, 281, 406, 594, 906, 1344, 2000, 3000, 4500, 6750
_FFT_SIZE = 512
_BAND_EDGES_BIN: tuple[int, ...] = (3, 4, 6, 9, 13, 19, 29, 43, 64, 96, 144, 216, 257)


def _fft_band_levels(samples: np.ndarray) -> list[float]:
    """Return 12 per-band FFT magnitude averages (float, raw, unnormalised)."""
    buf = np.zeros(_FFT_SIZE, dtype=np.float32)
    n = min(len(samples), _FFT_SIZE)
    buf[:n] = samples[:n].astype(np.float32)
    mag = np.abs(np.fft.rfft(buf))          # 257 bins (DC … Nyquist)
    result: list[float] = []
    for i in range(len(_BAND_EDGES_BIN) - 1):
        lo = _BAND_EDGES_BIN[i]
        hi = min(_BAND_EDGES_BIN[i + 1], len(mag))
        if lo >= len(mag):
            result.append(0.0)
        elif lo >= hi:
            result.append(float(mag[lo]))
        else:
            result.append(float(np.mean(mag[lo:hi])))
    return result


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        device: int | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self._stream: sd.InputStream | None = None
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._running = False
        # Callback now receives list[float] (12 FFT band levels) instead of float.
        self._level_callback: Callable[[list[float]], None] | None = None

    def set_level_callback(self, callback: Callable[[list[float]], None]) -> None:
        self._level_callback = callback

    @staticmethod
    def _is_invalid_device_error(exc: Exception) -> bool:
        message = str(exc).lower()
        return (
            "invalid device" in message
            or "error querying device -1" in message
            or "paerrorcode -9996" in message
        )

    @staticmethod
    def _enumerate_input_devices() -> tuple[list, list, int | None]:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        default_idx: int | None = None
        try:
            default_input = sd.query_devices(kind="input")
            default_idx = int(default_input["index"])
        except Exception as exc:
            log.warning("Default input device unavailable, falling back to first input: %s", exc)

        return list(devices), list(hostapis), default_idx

    @classmethod
    def _stream_device_candidates(cls, configured_device: int | None) -> list[int | None]:
        if configured_device is not None:
            return [configured_device]

        try:
            devices, _, default_idx = cls._enumerate_input_devices()
        except Exception as exc:
            log.warning("Failed to enumerate audio devices: %s", exc)
            return [None]

        input_indices = [
            idx
            for idx, device in enumerate(devices)
            if int(device.get("max_input_channels", 0)) > 0
        ]

        if default_idx is not None:
            return [None, *[idx for idx in input_indices if idx != default_idx]]

        for idx in input_indices:
            log.info("Using input device %s because default input is unavailable.", idx)
        return input_indices or [None]

    @staticmethod
    def _reinitialize_portaudio() -> None:
        try:
            sd._terminate()
            sd._initialize()
            log.info("PortAudio re-initialised after invalid audio device error.")
        except Exception as exc:
            log.warning("PortAudio reinit failed: %s", exc)

    def start(self) -> None:
        if self._running:
            return
        self._chunks.clear()

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if status:
                return
            chunk = indata[:, 0]
            with self._lock:
                self._chunks.append(np.copy(chunk))
            if self._level_callback:
                self._level_callback(_fft_band_levels(chunk))

        last_error: Exception | None = None
        reinit_attempted = False

        for _pass in range(2):
            candidates = self._stream_device_candidates(self.device)
            tried_candidates: set[int | None] = set()

            for candidate in candidates:
                if candidate in tried_candidates:
                    continue
                tried_candidates.add(candidate)
                try:
                    self._stream = sd.InputStream(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype="int16",
                        device=candidate,
                        callback=callback,
                    )
                    self._stream.start()
                    self._running = True
                    return
                except Exception as exc:
                    last_error = exc
                    self._stream = None
                    self._running = False
                    if self.device is None and candidate is not None:
                        log.warning("Input device %s rejected by PortAudio: %s", candidate, exc)

            if last_error is not None and not reinit_attempted and self._is_invalid_device_error(last_error):
                reinit_attempted = True
                self._reinitialize_portaudio()
                continue
            break

        if last_error is not None:
            log.error("Failed to open audio device: %s", last_error)
        self._stream = None
        self._running = False

    def stop(self) -> np.ndarray:
        if not self._running:
            return np.array([], dtype=np.int16)

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._running = False
        with self._lock:
            if not self._chunks:
                return np.array([], dtype=np.int16)
            return np.concatenate(self._chunks).astype(np.int16)

    def get_current_device_token(self) -> str:
        if self.device is None:
            return "default"
        return str(self.device)

    def set_device(self, token: str) -> bool:
        if self._running:
            log.warning("Audio device change rejected while recording is active.")
            return False

        if token == "default":
            old = self.get_current_device_token()
            self.device = None
            log.info("Audio device switched: %s -> default", old)
            return True

        try:
            device_id = int(token)
        except ValueError:
            log.warning("Invalid audio device token: %s", token)
            return False

        try:
            info = sd.query_devices(device_id)
        except Exception as exc:
            log.warning("Failed to query audio device %s: %s", device_id, exc)
            return False

        if int(info.get("max_input_channels", 0)) <= 0:
            log.warning("Audio device %s has no input channels.", device_id)
            return False

        old = self.get_current_device_token()
        self.device = device_id
        log.info("Audio device switched: %s -> %s", old, token)
        return True

    @staticmethod
    def list_input_devices() -> list[dict[str, str | bool]]:
        try:
            devices, hostapis, default_idx = AudioRecorder._enumerate_input_devices()
        except Exception as exc:
            log.warning("Failed to enumerate audio devices: %s", exc)
            return []

        result: list[dict[str, str | bool]] = []
        for idx, device in enumerate(devices):
            if int(device.get("max_input_channels", 0)) <= 0:
                continue
            host_name = "unknown"
            try:
                host_name = str(hostapis[int(device["hostapi"])]["name"])
            except Exception:
                host_name = "unknown"
            label = f"{device['name']} ({host_name})"
            result.append(
                {
                    "token": str(idx),
                    "label": label,
                    "is_default": idx == default_idx,
                }
            )
        return result

    @property
    def is_running(self) -> bool:
        return self._running
