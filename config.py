from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from paths import get_config_path, get_log_dir, get_model_dir


DEFAULT_CONFIG: dict[str, Any] = {
    "hotkey": {
        "combination": "win+ctrl",
        "mode": "hold",
        "debounce_ms": 200,
        "debug_trace": False,
        "debug_global": False,
    },
    "audio": {
        "device": None,
        "sample_rate": 16000,
        "channels": 1,
        "vad_enabled": True,
        "vad_threshold": 0.5,
        "vad_min_speech_ms": 120,
    },
    "whisper": {
        "model": "medium",
        "language": "auto",
        "backend": "auto",
        "compute_type": "auto",
        "beam_size": 5,
        "unload_after_idle_sec": 300,
        "download_root": str(get_model_dir()),
    },
    "overlay": {
        "width": 160,
        "height": 76,
        "monitor_index": -1,
        "opacity": 1.0,
        "bottom_offset": 100,
        "waveform_style": "gradient",
        "waveform_color": "#56F64E",
        "waveform_gradient_start": "#56F64E",
        "waveform_gradient_end": "#0096FF",
        "color_transcribing": "#FFC107",
        "transcribing_delay_ms": 250,
        "transcription_timeout_ms": 45000,
        "color_done": "#2196F3",
        "color_error": "#F44336",
        "display_duration_ms": 300,
    },
    "insertion": {
        "restore_clipboard": True,
        "paste_delay_ms": 50,
        "append_trailing_space": True,
        "rescue_enabled": True,
        "rescue_timeout_sec": 120,
        "rescue_never_expire": False,
    },
    "general": {
        "autostart": True,
        "log_level": "INFO",
        "debug_logging": False,
        "language_ui": "de",
        "log_to_file": True,
        "log_dir": str(get_log_dir()),
        "log_retention_days": 14,
    },
}


class ConfigManager:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else get_config_path()
        self.data: dict[str, Any] = deepcopy(DEFAULT_CONFIG)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            self.save(DEFAULT_CONFIG)
            self.data = deepcopy(DEFAULT_CONFIG)
            return self.data

        with self.path.open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        self.data = self._merge_dicts(deepcopy(DEFAULT_CONFIG), loaded)
        return self.data

    def save(self, config: dict[str, Any] | None = None) -> None:
        payload = config if config is not None else self.data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)

    def update(self, updates: dict[str, Any]) -> dict[str, Any]:
        self.data = self._merge_dicts(self.data, updates)
        self.save(self.data)
        return self.data

    def _merge_dicts(self, base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._merge_dicts(base[key], value)
            else:
                base[key] = value
        return base
