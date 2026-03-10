from __future__ import annotations

import logging
from pathlib import Path

from hardware import detect_hardware

log = logging.getLogger(__name__)


def create_backend(preferred: str, compute_type: str, download_root: str | Path | None = None):
    if preferred and preferred != "auto":
        return _build(preferred, compute_type, download_root)

    detected = detect_hardware().backend
    return _build(detected, compute_type, download_root)


def _build(name: str, compute_type: str, download_root: str | Path | None):
    try:
        if name == "cuda":
            from .backend_cuda import CUDABackend
            return CUDABackend(compute_type, download_root=download_root)
        if name == "openvino":
            from .backend_openvino import OpenVINOBackend
            return OpenVINOBackend(compute_type, download_root=download_root)
        from .backend_cpu import CPUBackend
        return CPUBackend(compute_type, download_root=download_root)
    except (ImportError, OSError, RuntimeError) as e:
        log.warning("Backend '%s' unavailable (%s), falling back to CPU", name, e)
        from .backend_cpu import CPUBackend
        return CPUBackend(compute_type, download_root=download_root)
