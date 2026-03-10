from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VadResult:
    speech: bool
    effective_threshold: float
    active_ms: int
    duration_ms: int


def has_speech(
    audio: np.ndarray,
    sample_rate: int = 16000,
    raw_threshold: float = 0.5,
    min_speech_ms: int = 120,
) -> VadResult:
    if audio.size == 0 or sample_rate <= 0:
        return VadResult(False, 0.0, 0, 0)

    signal = audio.astype(np.float32)
    if np.issubdtype(audio.dtype, np.integer):
        max_val = float(np.iinfo(audio.dtype).max)
        signal = signal / max_val
    if signal.ndim > 1:
        signal = signal.reshape(-1)

    frame_ms = 20
    frame_size = max(1, int(sample_rate * (frame_ms / 1000.0)))

    frames: list[np.ndarray] = []
    for start in range(0, signal.size, frame_size):
        frame = signal[start : start + frame_size]
        if frame.size < frame_size:
            frame = np.pad(frame, (0, frame_size - frame.size))
        frames.append(frame)

    if not frames:
        return VadResult(False, 0.0, 0, int(signal.size / sample_rate * 1000))

    frame_rms = np.array([float(np.sqrt(np.mean(np.square(frame)))) for frame in frames], dtype=np.float32)

    raw = max(float(raw_threshold), 0.0)
    base = raw if raw <= 0.1 else raw / 100.0

    sorted_rms = np.sort(frame_rms)
    low_count = max(1, int(sorted_rms.size * 0.3))
    noise_floor = float(np.median(sorted_rms[:low_count]))
    effective = max(base, noise_floor * 2.2)

    active_frames = int(np.sum(frame_rms >= effective))
    active_ms = active_frames * frame_ms
    duration_ms = int((signal.size / sample_rate) * 1000)
    speech = active_ms >= max(int(min_speech_ms), 1)

    return VadResult(
        speech=speech,
        effective_threshold=effective,
        active_ms=active_ms,
        duration_ms=duration_ms,
    )

