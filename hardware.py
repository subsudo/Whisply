from __future__ import annotations

import ctypes
import importlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import dataclass

from paths import get_cuda_runtime_dir


logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    backend: str
    device_name: str
    details: str


def _module_exists(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_hidden(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    kwargs: dict[str, object] = {
        "capture_output": True,
        "text": True,
        "timeout": timeout,
        "check": False,
    }
    if sys.platform.startswith("win"):
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        kwargs["startupinfo"] = startupinfo
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return subprocess.run(args, **kwargs)  # type: ignore[arg-type]


def _list_video_controller_names() -> list[str]:
    if not sys.platform.startswith("win"):
        return []
    try:
        proc = _run_hidden(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
            ],
            timeout=3.0,
        )
        if proc.returncode != 0:
            return []
        names = [line.strip() for line in (proc.stdout or "").splitlines() if line.strip()]
        return names
    except Exception:
        return []


def _detect_gpu_vendor() -> str:
    names = [name.lower() for name in _list_video_controller_names()]
    if any("nvidia" in name for name in names):
        return "nvidia"
    if any(("amd" in name) or ("radeon" in name) for name in names):
        return "amd"
    if any("intel" in name for name in names):
        return "intel"
    return "unknown"


def _detect_cuda() -> HardwareInfo | None:
    if not _module_exists("ctranslate2"):
        return None

    try:
        import ctranslate2  # type: ignore

        devices = ctranslate2.get_supported_compute_types("cuda")
        if devices:
            return HardwareInfo(
                backend="cuda",
                device_name="NVIDIA GPU",
                details=f"CUDA compute types: {', '.join(devices)}",
            )
    except Exception as exc:
        logger.debug("CUDA detection failed: %s", exc)
    return None


def _find_nvidia_smi_executable() -> str | None:
    resolved = shutil.which("nvidia-smi")
    if resolved:
        return resolved

    candidates: list[Path] = []
    for env_name in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
        base = os.environ.get(env_name)
        if not base:
            continue
        candidates.append(Path(base) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe")

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _detect_openvino() -> HardwareInfo | None:
    if not _module_exists("openvino"):
        return None

    try:
        from openvino.runtime import Core  # type: ignore

        core = Core()
        devices = core.available_devices
        preferred = [d for d in devices if d in {"NPU", "GPU"}] or devices
        if preferred:
            name = preferred[0]
            return HardwareInfo(
                backend="openvino",
                device_name=f"Intel {name}",
                details=f"OpenVINO devices: {', '.join(devices)}",
            )
    except Exception as exc:
        logger.debug("OpenVINO detection failed: %s", exc)
    return None


def detect_hardware() -> HardwareInfo:
    vendor = _detect_gpu_vendor()
    if vendor == "unknown" and has_nvidia_gpu():
        vendor = "nvidia"
    if vendor != "nvidia":
        if vendor == "amd":
            details = "CPU fallback (AMD GPU detected; CUDA unsupported)"
            device_name = "AMD GPU"
        elif vendor == "intel":
            details = "CPU fallback (Intel GPU detected; CUDA unsupported)"
            device_name = "Intel GPU"
        else:
            details = "CPU fallback (no NVIDIA GPU)"
            device_name = "CPU"
        cpu = HardwareInfo(backend="cpu", device_name=device_name, details=details)
        logger.info("Detected hardware: %s", cpu.details)
        return cpu

    cuda_ok, reason = check_cuda_runtime_available()
    if cuda_ok:
        cuda = HardwareInfo(backend="cuda", device_name="NVIDIA GPU", details="CUDA runtime available")
        logger.info("Detected hardware: %s", cuda.details)
        return cuda

    cpu = HardwareInfo(
        backend="cpu",
        device_name="CPU",
        details=f"CPU fallback (NVIDIA present, CUDA unavailable: {reason})",
    )
    logger.info("Detected hardware: %s", cpu.details)
    return cpu


def has_nvidia_gpu() -> bool:
    """Returns True if an NVIDIA CUDA-capable GPU appears to be available."""
    nvidia_smi = _find_nvidia_smi_executable()
    if nvidia_smi:
        try:
            proc = _run_hidden([nvidia_smi, "-L"], timeout=2.0)
            output = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
            if (
                proc.returncode == 0
                and "gpu " in output
                and "no devices were found" not in output
                and "couldn't communicate" not in output
                and "failed" not in output
            ):
                return True
        except Exception:
            pass

    return _detect_gpu_vendor() == "nvidia"


def detect_nvidia_vram_gb() -> float | None:
    nvidia_smi = _find_nvidia_smi_executable()
    if nvidia_smi:
        try:
            proc = _run_hidden(
                [
                    nvidia_smi,
                    "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits",
                ],
                timeout=2.0,
            )
            if proc.returncode == 0:
                values_mb: list[float] = []
                for line in proc.stdout.splitlines():
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        values_mb.append(float(raw))
                    except ValueError:
                        logger.debug("Could not parse VRAM value from '%s'", raw)
                        continue
                if values_mb:
                    return max(values_mb) / 1024.0
            else:
                logger.debug("nvidia-smi exited with code %s", proc.returncode)
        except Exception as exc:
            logger.debug("nvidia-smi VRAM query failed: %s", exc)

    # Fallback for systems where nvidia-smi is unavailable but NVIDIA is present.
    if sys.platform.startswith("win"):
        try:
            proc = _run_hidden(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Get-CimInstance Win32_VideoController "
                        "| Where-Object { $_.Name -like '*NVIDIA*' } "
                        "| Select-Object -ExpandProperty AdapterRAM"
                    ),
                ],
                timeout=3.0,
            )
            if proc.returncode == 0:
                values_bytes: list[float] = []
                for line in (proc.stdout or "").splitlines():
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        values_bytes.append(float(raw))
                    except ValueError:
                        continue
                if values_bytes:
                    return max(values_bytes) / (1024.0 ** 3)
        except Exception as exc:
            logger.debug("CIM VRAM query failed: %s", exc)

    return None


def recommend_model(backend: str, cuda_vram_gb: float | None = None) -> str:
    if backend == "cuda":
        if cuda_vram_gb is None:
            return "medium"
        if cuda_vram_gb >= 10.0:
            return "large-v3-turbo"
        return "medium"
    if backend == "cpu":
        return "medium"
    return "medium"


def check_cuda_runtime_available() -> tuple[bool, str]:
    """Checks if CUDA runtime can actually be used by ctranslate2."""
    try:
        import ctranslate2  # type: ignore

        compute_types = ctranslate2.get_supported_compute_types("cuda")
        if not compute_types:
            return False, "no_compute_types"

        # On Windows we proactively verify key CUDA runtime DLLs to avoid
        # late fallback only during first transcription.
        if sys.platform.startswith("win"):
            runtime_dir = get_cuda_runtime_dir()
            for dll_name in ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn64_9.dll"):
                try:
                    ctypes.WinDLL(dll_name)
                except OSError as exc:
                    dll_path = runtime_dir / dll_name
                    if dll_path.exists():
                        try:
                            ctypes.WinDLL(str(dll_path))
                            continue
                        except OSError as nested_exc:
                            return False, f"dll_error: {dll_name}: {nested_exc}"
                    return False, f"dll_error: {dll_name}: {exc}"

        return True, "ok"
    except ImportError:
        return False, "ctranslate2_missing"
    except OSError as exc:
        return False, f"dll_error: {exc}"
    except RuntimeError as exc:
        return False, f"runtime_error: {exc}"
    except Exception as exc:  # pragma: no cover - defensive fallback
        return False, f"unknown: {exc}"
