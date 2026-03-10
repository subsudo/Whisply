from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class CPUBackend:
    def __init__(self, compute_type: str = "int8", download_root: str | Path | None = None) -> None:
        self.compute_type = "int8" if compute_type == "auto" else compute_type
        self.download_root = str(download_root) if download_root else None
        self._model = None
        self._model_size = "medium"

    def load_model(self, model_size: str) -> None:
        from faster_whisper import WhisperModel  # type: ignore
        from model_store import get_model_path

        self._model_size = model_size
        local_model_path = get_model_path(model_size, self.download_root) if self.download_root else None
        if local_model_path:
            self._model = WhisperModel(
                local_model_path,
                device="cpu",
                compute_type=self.compute_type,
            )
            return

        self._model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=self.compute_type,
            download_root=self.download_root,
        )

    def transcribe(self, audio: np.ndarray, language: str = "auto", beam_size: int = 5) -> str:
        if self._model is None:
            self.load_model(self._model_size)

        audio = audio.astype(np.float32) / 32768.0
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            if 0.0 < peak < 0.08:
                gain = min(0.9 / peak, 12.0)
                audio = np.clip(audio * gain, -1.0, 1.0)

        kwargs: dict[str, Any] = {
            "beam_size": beam_size,
            "task": "transcribe",
            "condition_on_previous_text": False,
        }
        if language != "auto":
            kwargs["language"] = language
            if language == "de":
                kwargs["initial_prompt"] = "Dies ist ein deutsches Diktat."

        segments, _ = self._model.transcribe(audio, **kwargs)
        return " ".join(segment.text.strip() for segment in segments).strip()

    def unload_model(self) -> None:
        self._model = None

    def get_available_models(self) -> list[str]:
        return ["small", "medium", "large-v3", "large-v3-turbo"]

    def get_device_info(self) -> str:
        return f"CPU ({self.compute_type})"
