from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class OpenVINOBackend:
    def __init__(self, compute_type: str = "auto", download_root: str | Path | None = None) -> None:
        self.compute_type = compute_type
        self.download_root = str(download_root) if download_root else None
        self._pipeline = None
        self._model_size = "medium"

    def load_model(self, model_size: str) -> None:
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline  # type: ignore
        from model_store import get_model_path

        self._model_size = model_size
        model_id = f"openai/whisper-{model_size}"
        model_ref: str = model_id
        local_model_path = get_model_path(model_size, self.download_root) if self.download_root else None
        if local_model_path:
            # Optional best-effort local path usage. Falls back to HF model id if incompatible.
            config_file = Path(local_model_path) / "config.json"
            if config_file.exists():
                model_ref = local_model_path

        try:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_ref)
            processor = AutoProcessor.from_pretrained(model_ref)
        except Exception:
            model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
            processor = AutoProcessor.from_pretrained(model_id)
        self._pipeline = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=20,
            device="cpu",
        )

    def transcribe(self, audio: np.ndarray, language: str = "auto", beam_size: int = 5) -> str:
        if self._pipeline is None:
            self.load_model(self._model_size)

        audio = audio.astype(np.float32) / 32768.0
        if audio.size:
            peak = float(np.max(np.abs(audio)))
            if 0.0 < peak < 0.08:
                gain = min(0.9 / peak, 12.0)
                audio = np.clip(audio * gain, -1.0, 1.0)

        generate_kwargs: dict[str, Any] = {"num_beams": beam_size, "task": "transcribe"}
        if language != "auto":
            generate_kwargs["language"] = language

        result = self._pipeline(
            {"sampling_rate": 16000, "raw": audio},
            generate_kwargs=generate_kwargs,
        )
        return result["text"].strip()

    def unload_model(self) -> None:
        self._pipeline = None

    def get_available_models(self) -> list[str]:
        return ["small", "medium", "large-v3", "large-v3-turbo"]

    def get_device_info(self) -> str:
        return "OpenVINO"
