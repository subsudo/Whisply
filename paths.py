from __future__ import annotations

import os
from pathlib import Path


APP_NAME = "Whisply"


def get_appdata_dir() -> Path:
    base = os.environ.get("APPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Roaming")
    return Path(base) / APP_NAME


def get_localdata_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = str(Path.home() / "AppData" / "Local")
    return Path(base) / APP_NAME


def get_config_path() -> Path:
    return get_appdata_dir() / "config.yaml"


def get_marker_path() -> Path:
    return get_appdata_dir() / ".first_run_complete"


def get_model_dir() -> Path:
    return get_localdata_dir() / "models"


def get_log_dir() -> Path:
    return get_localdata_dir() / "logs"


def get_cuda_runtime_dir() -> Path:
    return get_localdata_dir() / "cuda_runtime"


def resolve_user_path(path_value: str | Path) -> Path:
    expanded = os.path.expandvars(str(path_value))
    return Path(expanded).expanduser()
