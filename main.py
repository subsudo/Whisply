from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import os
import signal
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
try:
    import winreg
except ImportError:  # pragma: no cover - non-Windows fallback
    winreg = None  # type: ignore[assignment]

# Prefer classic HTTP download path on Windows to avoid sporadic XET cache/reparse issues.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from paths import get_cuda_runtime_dir

# Add NVIDIA DLL paths so ctranslate2 can find cublas/cudnn (dev + bundled exe)
def _register_cuda_dll_paths() -> None:
    if not hasattr(os, "add_dll_directory"):
        return

    log = logging.getLogger(__name__)
    candidates: list[Path] = []

    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "nvidia_bins")

    candidates.append(get_cuda_runtime_dir())

    exe_dir = Path(sys.executable).resolve().parent
    candidates.append(exe_dir / "runtime" / "cuda")
    candidates.append(exe_dir / "nvidia_bins")

    venv_site = Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "nvidia"
    if venv_site.is_dir():
        for pkg_name in ("cublas", "cudnn", "cuda_runtime"):
            candidates.append(venv_site / pkg_name / "bin")

    for env_name, env_value in os.environ.items():
        upper = env_name.upper()
        if not upper.startswith("CUDA_PATH"):
            continue
        if not env_value:
            continue
        candidates.append(Path(env_value) / "bin")

    if winreg is not None:
        try:
            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA",
            )
            index = 0
            while True:
                try:
                    subkey_name = winreg.EnumKey(key, index)
                except OSError:
                    break
                try:
                    subkey = winreg.OpenKey(key, subkey_name)
                    install_dir, _ = winreg.QueryValueEx(subkey, "InstallDir")
                    candidates.append(Path(str(install_dir)) / "bin")
                    winreg.CloseKey(subkey)
                except OSError:
                    pass
                index += 1
            winreg.CloseKey(key)
        except OSError:
            pass

    program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
    cuda_root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
    if cuda_root.is_dir():
        for version_dir in sorted(cuda_root.glob("v*")):
            candidates.append(version_dir / "bin")

    cudnn_root = Path(program_files) / "NVIDIA" / "CUDNN"
    if cudnn_root.is_dir():
        for version_dir in sorted(cudnn_root.glob("*")):
            candidates.append(version_dir / "bin")

    seen: set[str] = set()
    for dll_dir in candidates:
        if not dll_dir.is_dir():
            continue
        key = str(dll_dir.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            os.add_dll_directory(str(dll_dir))
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
        except Exception:
            continue

    if seen:
        log.info("CUDA DLL directories registered: %s", sorted(seen))
    else:
        log.info("No CUDA DLL directories found.")


_register_cuda_dll_paths()

from PySide6.QtCore import QObject, QTimer, Signal, qInstallMessageHandler
from PySide6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon

from config import ConfigManager
from cuda_downloader import (
    CudaDownloadDialog,
    install_cuda_runtime_headless,
    is_cuda_runtime_downloaded,
)
from hardware import (
    check_cuda_runtime_available,
    detect_hardware,
    detect_nvidia_vram_gb,
    has_nvidia_gpu,
    recommend_model,
)
from hotkey import HoldHotkey
from i18n import normalize_ui_language, tr
from inserter import TextInserter
from first_run_dialog import FirstRunSetupDialog
from overlay import OverlayWidget
from model_store import SUPPORTED_MODELS, ensure_model_installed, get_model_status, refresh_model_status
from paths import (
    get_config_path,
    get_log_dir,
    get_marker_path,
    get_model_dir,
    resolve_user_path,
)
from recorder import AudioRecorder
from settings_dialog import SettingsDialog
from tray import AppTray
from transcriber import Transcriber
from vad import has_speech


def _is_cuda_fallback_oom(reason: str) -> bool:
    reason_lower = str(reason).lower()
    has_out_of_memory = "out of memory" in reason_lower
    has_cuda_alloc = "alloc" in reason_lower and "cuda" in reason_lower
    return has_out_of_memory or has_cuda_alloc


class EventBridge(QObject):
    start_recording = Signal()
    stop_recording = Signal()
    toggle_recording = Signal()
    level_update = Signal(object)   # carries list[float] – 12 FFT band levels
    model_loading_start = Signal(str, str)
    model_loading_progress = Signal(int)
    model_loading_done = Signal()
    transcription_done = Signal(str)
    transcription_error = Signal(str)
    transcription_settled = Signal()
    cuda_fallback = Signal(str)
    model_install_started = Signal(str)
    model_install_progress = Signal(int)
    model_install_finished = Signal(str, bool, str)


class CrashSafeFileHandler(logging.FileHandler):
    """File handler that flushes and fsyncs each record for crash debugging."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        try:
            self.flush()
            if self.stream and hasattr(self.stream, "fileno"):
                os.fsync(self.stream.fileno())
        except Exception:
            pass


class WhisperTypeApp:
    def __init__(self) -> None:
        self._shutting_down = False
        self._fault_log_handle = None
        self._debug_trace_handle = None
        self._debug_state_path: Path | None = None
        self._qt_message_handler_installed = False
        self._config_path = get_config_path()
        self._migrate_legacy_config_file(self._config_path)
        self.config_manager = ConfigManager(self._config_path)
        self.config = self.config_manager.load()
        self._normalize_storage_paths()
        self._setup_logging()
        self._setup_debug_trace()
        self._install_global_exception_hooks()
        self._install_fault_handler()
        log = logging.getLogger(__name__)
        self._normalize_legacy_insertion_mode()
        self._normalize_overlay_waveform_color()
        self._normalize_ui_language()
        self._normalize_hotkey_mode()
        self._normalize_backend_choice()
        self._sync_autostart_from_registry()
        self._log_runtime_snapshot()

        log.info("Starting %s ...", self._t("app_name"))
        log.info("Creating Qt application")
        self.qt_app = QApplication(sys.argv)
        self._install_qt_message_handler()
        self.qt_app.setQuitOnLastWindowClosed(False)
        self.qt_app.screenAdded.connect(self._on_screen_added)
        self.qt_app.screenRemoved.connect(self._on_screen_removed)
        self.bridge = EventBridge()

        log.info("Creating overlay")
        self.overlay = OverlayWidget(self.config["overlay"])

        log.info("Creating audio recorder")
        configured_device = self._configured_audio_device()
        self.recorder = AudioRecorder(
            sample_rate=self.config["audio"]["sample_rate"],
            channels=self.config["audio"]["channels"],
            device=configured_device,
        )

        log.info("Creating transcriber (backend=%s)", self.config["whisper"]["backend"])
        self.transcriber = Transcriber(
            backend_name=self.config["whisper"]["backend"],
            model_size=self.config["whisper"]["model"],
            language=self.config["whisper"]["language"],
            compute_type=self.config["whisper"]["compute_type"],
            beam_size=self.config["whisper"]["beam_size"],
            download_root=self.config["whisper"]["download_root"],
            on_cuda_fallback=self.bridge.cuda_fallback.emit,
            on_model_load_start=self.bridge.model_loading_start.emit,
            on_model_load_progress=self.bridge.model_loading_progress.emit,
            on_model_load_done=self.bridge.model_loading_done.emit,
        )

        log.info("Creating inserter")
        self.inserter = TextInserter(
            paste_delay_ms=self.config["insertion"]["paste_delay_ms"],
            restore_clipboard=self.config["insertion"]["restore_clipboard"],
            append_trailing_space=self.config["insertion"]["append_trailing_space"],
        )

        self._idle_unload_timer = QTimer()
        self._idle_unload_timer.setInterval(10_000)
        self._idle_unload_timer.timeout.connect(self._on_idle_unload_tick)

        self._transcribing_delay_ms = int(self.config["overlay"].get("transcribing_delay_ms", 250))
        self._transcription_timeout_ms = int(self.config["overlay"].get("transcription_timeout_ms", 45000))
        self._transcription_in_progress = False
        self._model_loading_active = False
        self._model_loading_model: str | None = None
        self._model_loading_mode = "download"
        self._discard_timed_out_result = False
        self._transcribing_delay_timer = QTimer()
        self._transcribing_delay_timer.setSingleShot(True)
        self._transcribing_delay_timer.timeout.connect(self._on_transcribing_delay_elapsed)
        self._transcription_timeout_timer = QTimer()
        self._transcription_timeout_timer.setSingleShot(True)
        self._transcription_timeout_timer.timeout.connect(self._on_transcription_timeout)

        self._recording_started_at = 0.0
        self._transcribe_lock = threading.Lock()
        self._settings_dialog: SettingsDialog | None = None
        self._cuda_prompt_active = False
        self._cuda_prompt_shown = False
        self._pending_cuda_reenable = False
        self._cuda_restart_prompt_pending = False
        self._model_install_in_progress = False
        self._model_install_target: str | None = None
        self._rescue_text = ""
        self._rescue_expires_at = 0.0
        self._rescue_expire_timer = QTimer()
        self._rescue_expire_timer.setSingleShot(True)
        self._rescue_expire_timer.timeout.connect(self._on_rescue_expired)
        self._audio_device_fingerprint: frozenset | None = None
        self._tray_retry_timer = QTimer()
        self._tray_retry_timer.setInterval(1500)
        self._tray_retry_timer.timeout.connect(self._ensure_tray_visible)
        self._tray_retry_attempts = 0
        self._tray_retry_max_attempts = 40
        self._wire_events()

        log.info("Setting up tray icon")
        self._setup_tray()

        self._device_monitor_timer = QTimer()
        self._device_monitor_timer.setInterval(5000)
        self._device_monitor_timer.timeout.connect(self._check_audio_device_changes)
        self._device_monitor_timer.start()

        log.info("Setting up hotkey: %s", self.config["hotkey"]["combination"])
        self._setup_hotkey()

        QTimer.singleShot(300, self._run_first_start_if_needed)
        self._idle_unload_timer.start()
        if self._hotkey_mode() == "press":
            log.info("%s ready! Press '%s' to start/stop recording.", self._t("app_name"), self.config["hotkey"]["combination"])
        else:
            log.info("%s ready! Hold '%s' to record.", self._t("app_name"), self.config["hotkey"]["combination"])

    def _migrate_legacy_config_file(self, target_config_path: Path) -> None:
        if target_config_path.exists():
            return

        candidates: list[Path] = [Path("config.yaml").resolve()]
        appdata = os.environ.get("APPDATA")
        if appdata:
            candidates.append(Path(appdata) / "WhisperType" / "config.yaml")

        for legacy_config in candidates:
            if not legacy_config.exists():
                continue
            target_config_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(legacy_config, target_config_path)
            return

    def _normalize_storage_paths(self) -> None:
        changed = False

        whisper_cfg = self.config.setdefault("whisper", {})
        default_download_root = str(get_model_dir())
        raw_download_root = whisper_cfg.get("download_root") or default_download_root
        download_root = resolve_user_path(raw_download_root)
        if str(download_root).lower().endswith("\\whispertype\\models"):
            whisper_cfg["download_root"] = default_download_root
            changed = True
        elif str(whisper_cfg.get("download_root")) != str(download_root):
            whisper_cfg["download_root"] = str(download_root)
            changed = True

        general_cfg = self.config.setdefault("general", {})
        default_log_dir = str(get_log_dir())
        raw_log_dir = general_cfg.get("log_dir") or default_log_dir
        log_dir = resolve_user_path(raw_log_dir)
        if str(log_dir).lower().endswith("\\whispertype\\logs"):
            general_cfg["log_dir"] = default_log_dir
            changed = True
        elif str(general_cfg.get("log_dir")) != str(log_dir):
            general_cfg["log_dir"] = str(log_dir)
            changed = True
        if "log_to_file" not in general_cfg:
            general_cfg["log_to_file"] = True
            changed = True
        if "log_retention_days" not in general_cfg:
            general_cfg["log_retention_days"] = 14
            changed = True

        if changed:
            self.config_manager.save(self.config)

    def _normalize_legacy_insertion_mode(self) -> None:
        insertion_cfg = self.config.setdefault("insertion", {})
        legacy = str(insertion_cfg.get("method", "clipboard")).lower()
        if legacy != "clipboard":
            logging.info("Legacy insertion.method '%s' detected; forcing clipboard mode.", legacy)
            insertion_cfg["method"] = "clipboard"
            self.config_manager.save(self.config)

    def _normalize_overlay_waveform_color(self) -> None:
        overlay_cfg = self.config.setdefault("overlay", {})
        changed = False
        fallback_single = str(overlay_cfg.get("waveform_color", overlay_cfg.get("color_recording", "#56F64E")))
        if not overlay_cfg.get("waveform_color"):
            overlay_cfg["waveform_color"] = fallback_single
            changed = True

        style = str(overlay_cfg.get("waveform_style", "gradient")).strip().lower()
        if style not in {"gradient", "single"}:
            overlay_cfg["waveform_style"] = "gradient"
            changed = True

        if not overlay_cfg.get("waveform_gradient_start"):
            overlay_cfg["waveform_gradient_start"] = str(overlay_cfg.get("waveform_color", "#56F64E"))
            changed = True
        if not overlay_cfg.get("waveform_gradient_end"):
            overlay_cfg["waveform_gradient_end"] = "#0096FF"
            changed = True

        raw_monitor_index = overlay_cfg.get("monitor_index", -1)
        try:
            monitor_index = int(raw_monitor_index)
        except (TypeError, ValueError):
            monitor_index = -1
        if monitor_index < -1:
            monitor_index = -1
        if overlay_cfg.get("monitor_index") != monitor_index:
            overlay_cfg["monitor_index"] = monitor_index
            changed = True

        if changed:
            self.config_manager.save(self.config)

    def _normalize_ui_language(self) -> None:
        general_cfg = self.config.setdefault("general", {})
        normalized = normalize_ui_language(str(general_cfg.get("language_ui", "de")))
        if general_cfg.get("language_ui") != normalized:
            general_cfg["language_ui"] = normalized
            self.config_manager.save(self.config)

    def _normalize_hotkey_mode(self) -> None:
        hotkey_cfg = self.config.setdefault("hotkey", {})
        mode = str(hotkey_cfg.get("mode", "press")).strip().lower()
        if mode not in {"hold", "press"}:
            hotkey_cfg["mode"] = "press"
            self.config_manager.save(self.config)

    def _hotkey_mode(self) -> str:
        mode = str(self.config.get("hotkey", {}).get("mode", "press")).strip().lower()
        if mode not in {"hold", "press"}:
            return "press"
        return mode

    def _normalize_backend_choice(self) -> None:
        whisper_cfg = self.config.setdefault("whisper", {})
        backend = str(whisper_cfg.get("backend", "auto")).strip().lower()
        if backend == "openvino":
            whisper_cfg["backend"] = "cpu"
            self.config_manager.save(self.config)
            logging.info("Backend normalized from openvino to cpu.")
            return
        if backend not in {"auto", "cuda", "cpu"}:
            whisper_cfg["backend"] = "auto"
            self.config_manager.save(self.config)
            logging.info("Unknown backend '%s' normalized to auto.", backend)

    def _ui_language(self) -> str:
        return normalize_ui_language(str(self.config.get("general", {}).get("language_ui", "de")))

    def _t(self, key: str, **kwargs: object) -> str:
        return tr(self._ui_language(), key, **kwargs)

    def _autostart_command(self) -> str:
        if getattr(sys, "frozen", False):
            return f'"{Path(sys.executable).resolve()}"'
        return f'"{Path(sys.executable).resolve()}" "{Path(__file__).resolve()}"'

    def _set_autostart(self, enabled: bool) -> bool:
        if winreg is None:
            return False
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        value_name = "Whisply"
        try:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                key_path,
                0,
                winreg.KEY_SET_VALUE,
            ) as key:
                if enabled:
                    winreg.SetValueEx(key, value_name, 0, winreg.REG_SZ, self._autostart_command())
                else:
                    try:
                        winreg.DeleteValue(key, value_name)
                    except FileNotFoundError:
                        pass
            return True
        except Exception:
            logging.exception("Failed to update autostart registry value.")
            return False

    def _is_autostart_enabled(self) -> bool:
        if winreg is None:
            return bool(self.config.get("general", {}).get("autostart", True))
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        value_name = "Whisply"
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_QUERY_VALUE) as key:
                value, _ = winreg.QueryValueEx(key, value_name)
                return bool(str(value).strip())
        except FileNotFoundError:
            return False
        except OSError:
            return False

    def _sync_autostart_from_registry(self) -> None:
        actual = self._is_autostart_enabled()
        general_cfg = self.config.setdefault("general", {})
        if bool(general_cfg.get("autostart", True)) != actual:
            general_cfg["autostart"] = actual
            self.config_manager.save(self.config)

    def _configured_audio_device(self) -> int | None:
        raw = self.config["audio"].get("device")
        if raw is None or raw == "default" or raw == "":
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            logging.warning("Invalid audio.device value '%s'. Using default input.", raw)
            self.config["audio"]["device"] = None
            self.config_manager.save(self.config)
            return None

    def _effective_log_level(self) -> int:
        if bool(self.config.get("general", {}).get("debug_logging", False)):
            return logging.DEBUG
        return getattr(logging, self.config["general"]["log_level"].upper(), logging.INFO)

    def _apply_logging_level(self) -> None:
        level = self._effective_log_level()
        root = logging.getLogger()
        root.setLevel(level)
        for handler in root.handlers:
            handler.setLevel(level)

    def _reconfigure_runtime_logging(self) -> None:
        self._setup_logging()
        self._setup_debug_trace()
        self._apply_logging_level()
        logging.info("Runtime logging reconfigured. debug=%s", self._debug_logging_enabled())

    def _setup_logging(self) -> None:
        level = self._effective_log_level()
        handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

        if bool(self.config["general"].get("log_to_file", True)):
            raw_log_dir = self.config["general"].get("log_dir") or str(get_log_dir())
            log_dir = resolve_user_path(raw_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            retention_days = int(self.config["general"].get("log_retention_days", 14))
            self._cleanup_old_logs(log_dir, retention_days)
            log_file = log_dir / f"whisply-{time.strftime('%Y%m%d')}.log"
            if bool(self.config.get("general", {}).get("debug_logging", False)):
                handlers.append(CrashSafeFileHandler(log_file, encoding="utf-8"))
            else:
                handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            handlers=handlers,
            force=True,
        )

    def _debug_logging_enabled(self) -> bool:
        return bool(self.config.get("general", {}).get("debug_logging", False))

    def _setup_debug_trace(self) -> None:
        self._close_debug_trace()
        if not self._debug_logging_enabled():
            return
        if not bool(self.config["general"].get("log_to_file", True)):
            return
        try:
            raw_log_dir = self.config["general"].get("log_dir") or str(get_log_dir())
            log_dir = resolve_user_path(raw_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._debug_state_path = log_dir / "whisply-last-critical.json"
            trace_path = log_dir / "whisply-debug-trace.log"
            self._debug_trace_handle = open(trace_path, "a", encoding="utf-8", buffering=1)
            self._debug_trace("debug_session_started")
            self._recover_debug_state()
        except Exception:
            logging.exception("Failed to initialise debug trace logging.")

    def _close_debug_trace(self) -> None:
        if self._debug_trace_handle is not None:
            try:
                self._debug_trace_handle.flush()
                self._debug_trace_handle.close()
            except Exception:
                pass
            finally:
                self._debug_trace_handle = None
        self._debug_state_path = None

    def _debug_trace(self, event: str, **fields: object) -> None:
        if self._debug_trace_handle is None:
            return
        try:
            payload = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event": str(event),
                "fields": fields,
            }
            self._debug_trace_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self._debug_trace_handle.flush()
            os.fsync(self._debug_trace_handle.fileno())
        except Exception:
            logging.exception("Failed to write debug trace event: %s", event)

    def _debug_set_critical_step(self, step: str, **fields: object) -> None:
        if self._debug_state_path is None:
            return
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "step": str(step),
            "fields": fields,
        }
        try:
            with self._debug_state_path.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False, indent=2))
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            logging.exception("Failed to persist debug critical step: %s", step)
        self._debug_trace("critical_step_set", step=step, **fields)

    def _debug_clear_critical_step(self, note: str | None = None) -> None:
        if self._debug_state_path is not None and self._debug_state_path.exists():
            try:
                self._debug_state_path.unlink(missing_ok=True)
            except Exception:
                logging.exception("Failed to clear debug critical step file.")
        self._debug_trace("critical_step_cleared", note=note or "")

    def _recover_debug_state(self) -> None:
        if self._debug_state_path is None or not self._debug_state_path.exists():
            return
        try:
            payload = json.loads(self._debug_state_path.read_text(encoding="utf-8"))
            step = payload.get("step", "unknown")
            ts = payload.get("ts", "unknown")
            fields = payload.get("fields", {})
            logging.warning(
                "Previous session may have terminated during critical step '%s' at %s: %s",
                step,
                ts,
                fields,
            )
            self._debug_trace("previous_session_terminated_during_critical_step", step=step, ts=ts, fields=fields)
        except Exception:
            logging.exception("Failed to recover previous debug critical step.")

    def _install_qt_message_handler(self) -> None:
        if self._qt_message_handler_installed:
            return

        def _qt_message_handler(mode, context, message) -> None:  # noqa: ANN001
            category = getattr(context, "category", "") if context is not None else ""
            file_name = getattr(context, "file", "") if context is not None else ""
            line_no = getattr(context, "line", 0) if context is not None else 0
            text = f"Qt message [{category}] {message} ({file_name}:{line_no})"
            mode_name = str(mode)
            try:
                if "QtFatalMsg" in mode_name:
                    logging.critical(text)
                    self._debug_trace("qt_fatal", message=message, category=category, file=file_name, line=line_no)
                elif "QtCriticalMsg" in mode_name:
                    logging.error(text)
                elif "QtWarningMsg" in mode_name:
                    logging.warning(text)
                elif self._debug_logging_enabled():
                    logging.debug(text)
            except Exception:
                pass

        qInstallMessageHandler(_qt_message_handler)
        self._qt_message_handler_installed = True
        logging.info("Qt message handler installed.")

    def _install_fault_handler(self) -> None:
        if not bool(self.config["general"].get("log_to_file", True)):
            return
        try:
            raw_log_dir = self.config["general"].get("log_dir") or str(get_log_dir())
            log_dir = resolve_user_path(raw_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            fault_log = log_dir / "whisply-fault.log"
            self._fault_log_handle = open(fault_log, "a", encoding="utf-8", buffering=1)
            self._fault_log_handle.write(
                f"\n=== Fault handler active: {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n"
            )
            self._fault_log_handle.flush()
            os.fsync(self._fault_log_handle.fileno())
            faulthandler.enable(file=self._fault_log_handle, all_threads=True)
            logging.info("Fault handler enabled: %s", fault_log)
        except Exception:
            logging.exception("Failed to enable Python fault handler.")

    def _log_runtime_snapshot(self) -> None:
        try:
            logging.info("Runtime snapshot: config_path=%s", self._config_path)
            logging.info("Runtime snapshot: model_dir=%s", self.config["whisper"]["download_root"])
            logging.info("Runtime snapshot: backend=%s model=%s language=%s",
                         self.config["whisper"]["backend"],
                         self.config["whisper"]["model"],
                         self.config["whisper"]["language"])
            logging.info("Runtime snapshot: log_dir=%s", self.config["general"].get("log_dir"))
            logging.info("Runtime snapshot: ui_language=%s hotkey=%s mode=%s",
                         self.config["general"].get("language_ui"),
                         self.config["hotkey"].get("combination"),
                         self.config["hotkey"].get("mode"))
        except Exception:
            logging.exception("Failed to write runtime snapshot.")

    def _install_global_exception_hooks(self) -> None:
        def _sys_excepthook(exc_type, exc_value, exc_tb) -> None:  # noqa: ANN001
            if issubclass(exc_type, KeyboardInterrupt):
                try:
                    sys.__excepthook__(exc_type, exc_value, exc_tb)
                except Exception:
                    pass
                return
            logging.critical(
                "Uncaught exception (sys): %s",
                "".join(traceback.format_exception(exc_type, exc_value, exc_tb)).strip(),
            )

        def _threading_excepthook(args) -> None:  # noqa: ANN001
            try:
                logging.critical(
                    "Uncaught exception (thread '%s'): %s",
                    getattr(args, "thread", None).name if getattr(args, "thread", None) else "unknown",
                    "".join(
                        traceback.format_exception(
                            getattr(args, "exc_type", Exception),
                            getattr(args, "exc_value", Exception("unknown")),
                            getattr(args, "exc_traceback", None),
                        )
                    ).strip(),
                )
            except Exception:
                logging.exception("Failed to log uncaught thread exception.")

        sys.excepthook = _sys_excepthook
        if hasattr(threading, "excepthook"):
            threading.excepthook = _threading_excepthook

    def _cleanup_old_logs(self, log_dir: Path, retention_days: int) -> None:
        if retention_days <= 0:
            return
        cutoff = time.time() - (retention_days * 24 * 60 * 60)
        for candidate in log_dir.glob("whisply-*.log"):
            try:
                if candidate.stat().st_mtime < cutoff:
                    candidate.unlink(missing_ok=True)
            except Exception:
                logging.debug("Could not remove old log file: %s", candidate)

    def _wire_events(self) -> None:
        self.recorder.set_level_callback(self.bridge.level_update.emit)
        self.bridge.level_update.connect(self.overlay.set_audio_levels)
        self.bridge.start_recording.connect(self._start_recording)
        self.bridge.stop_recording.connect(self._stop_recording)
        self.bridge.toggle_recording.connect(self._toggle_recording)
        self.bridge.model_loading_start.connect(self._on_model_loading_start)
        self.bridge.model_loading_progress.connect(self._on_model_loading_progress)
        self.bridge.model_loading_done.connect(self._on_model_loading_done)
        self.bridge.transcription_done.connect(self._apply_transcription)
        self.bridge.transcription_error.connect(self.overlay.show_error)
        self.bridge.transcription_settled.connect(self._cancel_transcribing_delay)
        self.bridge.transcription_settled.connect(self._cancel_transcription_timeout)
        self.bridge.transcription_settled.connect(self._apply_pending_cuda_reenable)
        self.bridge.transcription_settled.connect(self._maybe_prompt_cuda_restart)
        self.bridge.cuda_fallback.connect(self._on_cuda_fallback)
        self.bridge.model_install_started.connect(self._on_model_install_started)
        self.bridge.model_install_progress.connect(self._on_model_install_progress)
        self.bridge.model_install_finished.connect(self._on_model_install_finished)

    def _setup_hotkey(self) -> None:
        combination = str(self.config["hotkey"]["combination"])
        mode = self._hotkey_mode()
        if mode == "press":
            on_down = self.bridge.toggle_recording.emit
            on_up = lambda: None
        else:
            on_down = self.bridge.start_recording.emit
            on_up = self.bridge.stop_recording.emit
        try:
            self.hotkey = HoldHotkey(
                combination=combination,
                on_down=on_down,
                on_up=on_up,
                debounce_ms=self.config["hotkey"]["debounce_ms"],
                debug_trace=bool(self.config["hotkey"].get("debug_trace", False)),
                debug_global=bool(self.config["hotkey"].get("debug_global", False)),
            )
        except ValueError as exc:
            logging.warning("Invalid hotkey '%s' (%s). Falling back to win+ctrl.", combination, exc)
            combination = "win+ctrl"
            self.hotkey = HoldHotkey(
                combination=combination,
                on_down=on_down,
                on_up=on_up,
                debounce_ms=self.config["hotkey"]["debounce_ms"],
                debug_trace=bool(self.config["hotkey"].get("debug_trace", False)),
                debug_global=bool(self.config["hotkey"].get("debug_global", False)),
            )
            self.config["hotkey"]["combination"] = self.hotkey.combination
            self.config_manager.save(self.config)

        self.hotkey.start()
        self.config["hotkey"]["combination"] = self.hotkey.combination

    def _rebind_hotkey(self, new_combination: str) -> None:
        old_combination = self.config["hotkey"]["combination"]
        mode = self._hotkey_mode()
        if mode == "press":
            on_down = self.bridge.toggle_recording.emit
            on_up = lambda: None
        else:
            on_down = self.bridge.start_recording.emit
            on_up = self.bridge.stop_recording.emit
        self.hotkey.stop()
        try:
            new_hotkey = HoldHotkey(
                combination=new_combination,
                on_down=on_down,
                on_up=on_up,
                debounce_ms=self.config["hotkey"]["debounce_ms"],
                debug_trace=bool(self.config["hotkey"].get("debug_trace", False)),
                debug_global=bool(self.config["hotkey"].get("debug_global", False)),
            )
            new_hotkey.start()
            self.hotkey = new_hotkey
            self.config["hotkey"]["combination"] = new_hotkey.combination
        except Exception:
            rollback_hotkey = HoldHotkey(
                combination=old_combination,
                on_down=on_down,
                on_up=on_up,
                debounce_ms=self.config["hotkey"]["debounce_ms"],
                debug_trace=bool(self.config["hotkey"].get("debug_trace", False)),
                debug_global=bool(self.config["hotkey"].get("debug_global", False)),
            )
            rollback_hotkey.start()
            self.hotkey = rollback_hotkey
            raise

    def _toggle_recording(self) -> None:
        if self.recorder.is_running:
            self._stop_recording()
            return
        self._start_recording()

    def _open_settings(self) -> None:
        try:
            logging.info("Open settings requested.")
            self._debug_set_critical_step("settings_open_requested")
            if self._settings_dialog and self._settings_dialog.isVisible():
                logging.info("Settings dialog already visible; raising existing window.")
                self._debug_trace("settings_dialog_raise_existing")
                self._debug_clear_critical_step("settings_dialog_already_visible")
                self._settings_dialog.raise_()
                self._settings_dialog.activateWindow()
                return

            logging.info("Creating SettingsDialog instance.")
            self._debug_set_critical_step("settings_dialog_creating")
            self._settings_dialog = SettingsDialog(
                config=self.config,
                on_save=self._apply_settings,
                cuda_status_provider=self._cuda_status_payload,
                on_cuda_download=self._download_cuda_runtime_from_settings,
                model_status_provider=self._model_status_for_tray,
                on_open_logs=self._open_logs,
                on_open_config=self._open_config,
            )
            self._settings_dialog.destroyed.connect(lambda *_: logging.info("Settings dialog destroyed."))
            self._settings_dialog.destroyed.connect(
                lambda *_: self._debug_trace("settings_dialog_destroyed")
            )
            logging.info("SettingsDialog instance created.")
            logging.info("Showing SettingsDialog.")
            self._debug_set_critical_step("settings_dialog_show")
            self._settings_dialog.show()
            logging.info("Settings dialog opened.")
            self._debug_trace("settings_dialog_opened")
            QTimer.singleShot(0, lambda: self._debug_clear_critical_step("settings_dialog_opened"))
        except Exception:
            logging.exception("Failed to open settings dialog.")
            self._debug_trace("settings_dialog_open_failed")
            self._show_tray_message(self._t("tray_error_prefix") + ": settings")

    def _show_tray_message(self, body: str) -> None:
        try:
            self.tray.showMessage(self._t("app_name"), body)
        except Exception:
            logging.debug("Tray message failed: %s", body)

    def _unload_available(self) -> bool:
        try:
            return self.transcriber.is_model_loaded()
        except Exception:
            logging.exception("Failed to query model-loaded state.")
            return False

    def _unload_never_enabled(self) -> bool:
        try:
            return int(self.config["whisper"].get("unload_after_idle_sec", 300)) == 0
        except Exception:
            logging.exception("Failed to query unload-never state.")
            return False

    def _toggle_unload_never_from_tray(self) -> None:
        whisper_cfg = self.config["whisper"]
        current = int(whisper_cfg.get("unload_after_idle_sec", 300))
        previous = int(whisper_cfg.get("unload_after_idle_prev_sec", 300) or 300)
        if current == 0:
            restore = previous if previous > 0 else 300
            whisper_cfg["unload_after_idle_sec"] = restore
            logging.info("Tray unload-never disabled; restored timeout=%s sec.", restore)
        else:
            whisper_cfg["unload_after_idle_prev_sec"] = current
            whisper_cfg["unload_after_idle_sec"] = 0
            logging.info("Tray unload-never enabled; previous timeout=%s sec.", current)
        self.config_manager.save(self.config)
        self.tray.refresh_status()

    def _unload_model_from_tray(self) -> None:
        if self._transcription_in_progress or self.recorder.is_running:
            self._show_tray_message(self._t("model_install_busy"))
            return
        try:
            unloaded = self.transcriber.unload_now()
        except Exception:
            logging.exception("Manual model unload failed.")
            self._show_tray_message(self._t("tray_error_prefix"))
            return
        if unloaded:
            self._show_tray_message(self._t("tray_menu_unload_model"))
        self.tray.refresh_status()

    def _recommended_first_run_models(self, backend: str, cuda_vram_gb: float | None) -> list[str]:
        if cuda_vram_gb is not None:
            if cuda_vram_gb >= 10.0:
                return ["medium", "large-v3-turbo"]
            if cuda_vram_gb >= 6.0:
                return ["medium"]
            return ["small", "medium"]
        if backend == "cuda":
            return ["medium"]
        return ["small", "medium"]

    def _has_nvidia_gpu(self) -> bool:
        detected = has_nvidia_gpu()
        logging.info("NVIDIA GPU detected: %s", detected)
        return detected

    def _cuda_status_payload(self) -> dict[str, str | bool]:
        if not self._has_nvidia_gpu():
            return {
                "text": self._t("settings_cuda_state_no_gpu"),
                "downloadable": False,
            }

        ok, reason = check_cuda_runtime_available()
        device_name = "NVIDIA GPU"
        if ok:
            return {
                "text": self._t("settings_cuda_state_active", device=device_name),
                "downloadable": False,
            }

        downloadable = reason != "ctranslate2_missing"
        if reason == "unknown":
            downloadable = False
        return {
            "text": self._t("settings_cuda_state_missing", device=device_name),
            "downloadable": downloadable,
        }

    def _offer_cuda_runtime_download(
        self,
        source: str,
        force: bool = False,
        allow_without_gpu: bool = False,
        schedule_restart_prompt: bool = True,
    ) -> bool:
        if self._cuda_prompt_active:
            return False
        if self._cuda_prompt_shown and not force:
            return False
        if not allow_without_gpu and not self._has_nvidia_gpu():
            self._show_tray_message(self._t("settings_cuda_state_no_gpu"))
            return False

        # If runtime was already downloaded earlier, attempt re-register before prompting.
        if is_cuda_runtime_downloaded():
            _register_cuda_dll_paths()
            ok, reason = check_cuda_runtime_available()
            if ok:
                logging.info("CUDA runtime already present and usable (%s).", source)
                return True
            logging.warning(
                "CUDA runtime exists but is still unavailable (%s): %s",
                source,
                reason,
            )

        self._cuda_prompt_active = True
        if not force:
            self._cuda_prompt_shown = True
        try:
            try:
                dialog = CudaDownloadDialog(language=self._ui_language())
                dialog.exec()
            except Exception as exc:
                logging.exception("Could not open CUDA download dialog (%s).", source)
                self._show_tray_message(self._t("cuda_download_failed", error=str(exc)))
                return False

            if not dialog.was_successful():
                logging.info("CUDA runtime download skipped/failed (%s): %s", source, dialog.last_message())
                return False

            _register_cuda_dll_paths()
            ok, reason = check_cuda_runtime_available()
            if ok:
                logging.info("CUDA runtime ready after download (%s).", source)
                self._show_tray_message(self._t("cuda_download_success"))
                if schedule_restart_prompt:
                    self._schedule_cuda_restart_prompt()
                return True

            logging.warning("CUDA runtime still unavailable after download (%s): %s", source, reason)
            self._show_tray_message(self._t("cuda_runtime_missing"))
            return False
        finally:
            self._cuda_prompt_active = False

    def _resolve_backend_request(self, backend: str, source: str) -> str:
        requested = str(backend).strip().lower()
        if requested == "openvino":
            logging.info("OpenVINO backend request from %s mapped to cpu.", source)
            return "cpu"
        if requested not in {"auto", "cuda", "cpu"}:
            logging.warning("Unknown backend request '%s' from %s; using auto.", requested, source)
            return "auto"
        if requested != "cuda":
            return requested

        ok, reason = check_cuda_runtime_available()
        if ok:
            return "cuda"

        allow_without_gpu = str(reason).startswith(("dll_error:", "runtime_error:"))

        if reason == "ctranslate2_missing":
            logging.warning("CUDA requested but ctranslate2 is not available (%s).", source)
            self._show_tray_message(self._t("cuda_runtime_missing"))
            return "cpu"

        if not self._has_nvidia_gpu() and not allow_without_gpu:
            logging.warning("CUDA requested but no NVIDIA GPU detected (%s).", source)
            self._show_tray_message(self._t("settings_cuda_state_no_gpu"))
            return "cpu"

        force_prompt = source == "settings"
        if self._offer_cuda_runtime_download(
            source=source,
            force=force_prompt,
            allow_without_gpu=allow_without_gpu,
        ):
            ok, _ = check_cuda_runtime_available()
            if ok:
                return "cuda"

        self._show_tray_message(self._t("cuda_runtime_missing"))
        return "cpu"

    def _set_backend_choice(self, backend: str) -> None:
        self.config["whisper"]["backend"] = backend
        self.config_manager.save(self.config)
        self.transcriber.set_backend(
            backend,
            self.config["whisper"]["compute_type"],
            download_root=self.config["whisper"]["download_root"],
        )
        self.tray.set_selected(
            model=self.config["whisper"]["model"],
            transcription_language=self.config["whisper"]["language"],
            backend=self.config["whisper"]["backend"],
            ui_language=self.config["general"]["language_ui"],
        )
        self.tray.refresh_status()

    def _set_backend_config_only(self, backend: str) -> None:
        self.config["whisper"]["backend"] = backend
        self.config_manager.save(self.config)
        self.tray.set_selected(
            model=self.config["whisper"]["model"],
            transcription_language=self.config["whisper"]["language"],
            backend=self.config["whisper"]["backend"],
            ui_language=self.config["general"]["language_ui"],
        )
        self.tray.refresh_status()

    def _download_cuda_runtime_from_settings(self) -> bool:
        success = self._offer_cuda_runtime_download(source="settings_button", force=True)
        if success and self.config["whisper"].get("backend") == "cuda":
            self._set_backend_choice("cuda")
        return success

    def _on_cuda_fallback(self, reason: str) -> None:
        logging.warning("CUDA fallback triggered: %s", reason)
        is_oom = _is_cuda_fallback_oom(reason)
        if is_oom:
            self._show_tray_message(self._t("cuda_fallback_oom_notice"))
            self.overlay.show_warning(self._t("cuda_fallback_oom_notice"), ms=1800)
        else:
            self._show_tray_message(self._t("cuda_fallback_notice"))
        configured_backend = str(self.config["whisper"].get("backend", "auto"))
        if configured_backend not in {"cuda", "auto"}:
            if configured_backend != "cpu":
                self._set_backend_config_only("cpu")
            return

        if is_oom:
            self._pending_cuda_reenable = True
            if not self._transcription_in_progress:
                self._apply_pending_cuda_reenable()
            return

        if self._offer_cuda_runtime_download(
            source="runtime_fallback",
            force=False,
            allow_without_gpu=True,
        ):
            self.config["whisper"]["backend"] = "cuda"
            self.config_manager.save(self.config)
            self.tray.set_selected(
                model=self.config["whisper"]["model"],
                transcription_language=self.config["whisper"]["language"],
                backend=self.config["whisper"]["backend"],
                ui_language=self.config["general"]["language_ui"],
            )
            self.tray.refresh_status()
            self._pending_cuda_reenable = True
            if not self._transcription_in_progress:
                self._apply_pending_cuda_reenable()
            return

        self._set_backend_config_only("cpu")
        self.overlay.show_warning(self._t("cuda_runtime_missing"), ms=1400)

    def _apply_pending_cuda_reenable(self) -> None:
        if not self._pending_cuda_reenable:
            return
        self._pending_cuda_reenable = False
        if str(self.config["whisper"].get("backend")) != "cuda":
            return
        try:
            self._set_backend_choice("cuda")
        except Exception:
            logging.exception("Failed to re-enable CUDA backend after runtime install.")

    def _schedule_cuda_restart_prompt(self) -> None:
        self._cuda_restart_prompt_pending = True
        self._maybe_prompt_cuda_restart()

    def _maybe_prompt_cuda_restart(self) -> None:
        if not self._cuda_restart_prompt_pending:
            return
        if self._shutting_down:
            return
        if self._transcription_in_progress:
            return

        self._cuda_restart_prompt_pending = False
        choice = QMessageBox.question(
            None,
            self._t("cuda_restart_title"),
            self._t("cuda_restart_prompt"),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if choice == QMessageBox.Yes:
            self._show_tray_message(self._t("cuda_restart_manual"))

    def _apply_settings(self, payload: dict[str, object]) -> None:
        model = payload["model"]
        language = payload["language"]
        backend = payload["backend"]
        hotkey = payload["hotkey"]
        hotkey_mode = str(payload.get("hotkey_mode", self._hotkey_mode())).strip().lower()
        waveform_color = payload.get("waveform_color", self.config["overlay"]["waveform_color"])
        waveform_style = str(payload.get("waveform_style", self.config["overlay"].get("waveform_style", "gradient"))).strip().lower()
        waveform_gradient_start = payload.get(
            "waveform_gradient_start",
            self.config["overlay"].get("waveform_gradient_start", self.config["overlay"]["waveform_color"]),
        )
        waveform_gradient_end = payload.get(
            "waveform_gradient_end",
            self.config["overlay"].get("waveform_gradient_end", "#0096FF"),
        )
        overlay_monitor_index = int(
            payload.get("overlay_monitor_index", self.config["overlay"].get("monitor_index", -1))
        )
        language_ui = payload.get("language_ui", self.config["general"]["language_ui"])
        autostart = bool(payload.get("autostart", self.config["general"].get("autostart", True)))
        debug_logging = bool(payload.get("debug_logging", self.config["general"].get("debug_logging", False)))
        unload_after_idle_sec = int(
            payload.get("unload_after_idle_sec", self.config["whisper"].get("unload_after_idle_sec", 300))
        )
        rescue_enabled = bool(payload.get("rescue_enabled", self.config["insertion"].get("rescue_enabled", True)))
        rescue_timeout_sec = int(payload.get("rescue_timeout_sec", self.config["insertion"].get("rescue_timeout_sec", 120)))
        rescue_never_expire = bool(payload.get("rescue_never_expire", self.config["insertion"].get("rescue_never_expire", False)))

        needs_hotkey_rebind = False
        if hotkey != self.config["hotkey"]["combination"]:
            needs_hotkey_rebind = True
        if hotkey_mode not in {"hold", "press"}:
            hotkey_mode = "hold"
        if hotkey_mode != self._hotkey_mode():
            self.config["hotkey"]["mode"] = hotkey_mode
            needs_hotkey_rebind = True

        if needs_hotkey_rebind:
            self._rebind_hotkey(str(hotkey))

        selected_model = str(model).strip().lower()
        if selected_model != self.config["whisper"]["model"]:
            model_status = self._model_status_for_tray()
            if model_status.get(selected_model, False):
                self.config["whisper"]["model"] = selected_model
                self.transcriber.set_model(selected_model)
            else:
                logging.info(
                    "Selected model '%s' is not installed yet; starting install job.",
                    selected_model,
                )
                self._on_model_install_request(selected_model)

        if language != self.config["whisper"]["language"]:
            self.config["whisper"]["language"] = str(language)
            self.transcriber.set_language(str(language))

        resolved_backend = self._resolve_backend_request(str(backend), source="settings")
        if resolved_backend != self.config["whisper"]["backend"]:
            self._set_backend_choice(resolved_backend)

        if waveform_style not in {"single", "gradient"}:
            waveform_style = "gradient"
        if waveform_color != self.config["overlay"]["waveform_color"]:
            self.config["overlay"]["waveform_color"] = str(waveform_color)
        if waveform_style != str(self.config["overlay"].get("waveform_style", "gradient")):
            self.config["overlay"]["waveform_style"] = waveform_style
        if str(waveform_gradient_start) != str(
            self.config["overlay"].get("waveform_gradient_start", self.config["overlay"]["waveform_color"])
        ):
            self.config["overlay"]["waveform_gradient_start"] = str(waveform_gradient_start)
        if str(waveform_gradient_end) != str(self.config["overlay"].get("waveform_gradient_end", "#0096FF")):
            self.config["overlay"]["waveform_gradient_end"] = str(waveform_gradient_end)
        if overlay_monitor_index < -1:
            overlay_monitor_index = -1
        if overlay_monitor_index != int(self.config["overlay"].get("monitor_index", -1)):
            self.config["overlay"]["monitor_index"] = int(overlay_monitor_index)
            logging.info("Overlay monitor changed to index=%s", overlay_monitor_index)

        if unload_after_idle_sec != int(self.config["whisper"].get("unload_after_idle_sec", 300)):
            self.config["whisper"]["unload_after_idle_sec"] = unload_after_idle_sec
            if unload_after_idle_sec == 0:
                logging.info("Unload setting changed to never.")
            else:
                self.config["whisper"]["unload_after_idle_prev_sec"] = unload_after_idle_sec
                logging.info("Unload setting changed to %d sec.", unload_after_idle_sec)

        self.config["insertion"]["rescue_enabled"] = rescue_enabled
        self.config["insertion"]["rescue_timeout_sec"] = max(1, rescue_timeout_sec)
        self.config["insertion"]["rescue_never_expire"] = rescue_never_expire
        if not rescue_enabled:
            self._clear_rescue_text()
        else:
            self._reschedule_rescue_expiry()

        normalized_ui_language = normalize_ui_language(str(language_ui))
        if normalized_ui_language != self.config["general"]["language_ui"]:
            self.config["general"]["language_ui"] = normalized_ui_language

        if autostart != bool(self.config["general"].get("autostart", True)):
            if self._set_autostart(autostart):
                self.config["general"]["autostart"] = autostart
            else:
                raise RuntimeError("Autostart could not be updated.")

        if debug_logging != bool(self.config["general"].get("debug_logging", False)):
            self.config["general"]["debug_logging"] = debug_logging
            self._reconfigure_runtime_logging()
            logging.info("Debug logging changed to %s.", debug_logging)

        self.config_manager.save(self.config)
        logging.info("Settings saved to config. Scheduling deferred tray/overlay refresh.")
        QTimer.singleShot(0, self._apply_settings_ui_refresh)

    def _apply_settings_ui_refresh(self) -> None:
        logging.info("Deferred settings UI refresh started.")
        self.tray.set_selected(
            model=self.config["whisper"]["model"],
            transcription_language=self.config["whisper"]["language"],
            backend=self.config["whisper"]["backend"],
            ui_language=self.config["general"]["language_ui"],
        )
        self.overlay.cfg["waveform_color"] = self.config["overlay"]["waveform_color"]
        self.overlay.cfg["waveform_style"] = self.config["overlay"]["waveform_style"]
        self.overlay.cfg["waveform_gradient_start"] = self.config["overlay"]["waveform_gradient_start"]
        self.overlay.cfg["waveform_gradient_end"] = self.config["overlay"]["waveform_gradient_end"]
        self.overlay.cfg["monitor_index"] = int(self.config["overlay"].get("monitor_index", -1))
        self.overlay.place_bottom_center()
        self.tray.refresh_status()
        logging.info("Deferred settings UI refresh completed.")

    def _setup_tray(self) -> None:
        icon_path = str(Path("assets/icon.png"))
        self.tray = AppTray(
            icon_path=icon_path,
            short_status_provider=self._status_text_short,
            full_status_provider=self._status_text_full,
            ui_language_provider=self._ui_language,
            model_status_provider=self._model_status_for_tray,
            on_model=self._on_model_change,
            on_model_install=self._on_model_install_request,
            on_transcription_language=self._on_language_change,
            on_backend=self._on_backend_change,
            audio_devices_provider=self._audio_devices_for_tray,
            on_audio_device=self._on_audio_device_change,
            on_open_settings=self._open_settings,
            unload_available_provider=self._unload_available,
            unload_never_provider=self._unload_never_enabled,
            on_unload_model=self._unload_model_from_tray,
            on_toggle_unload_never=self._toggle_unload_never_from_tray,
            rescue_copy_available_provider=self._rescue_copy_available,
            on_copy_last_dictation=self._copy_last_dictation_to_clipboard,
            on_quit=self.shutdown,
            on_debug_trace=lambda event: self._debug_trace(event),
            on_debug_critical=lambda step: self._debug_set_critical_step(step),
            on_debug_clear=lambda note=None: self._debug_clear_critical_step(note),
        )
        self.tray.set_selected(
            model=self.config["whisper"]["model"],
            transcription_language=self.config["whisper"]["language"],
            backend=self.config["whisper"]["backend"],
            ui_language=self.config["general"]["language_ui"],
        )
        self.tray.refresh_status()
        self._ensure_tray_visible()

    def _ensure_tray_visible(self) -> None:
        if self._shutting_down:
            if self._tray_retry_timer.isActive():
                self._tray_retry_timer.stop()
            return

        if self.tray.isVisible():
            if self._tray_retry_timer.isActive():
                self._tray_retry_timer.stop()
            return

        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray_retry_attempts += 1
            if self._tray_retry_attempts == 1:
                logging.warning("System tray not available yet. Retrying tray icon show.")
            if self._tray_retry_attempts >= self._tray_retry_max_attempts:
                logging.error("System tray unavailable after retries. Tray icon remains hidden.")
                self._tray_retry_timer.stop()
                return
            if not self._tray_retry_timer.isActive():
                self._tray_retry_timer.start()
            return

        self.tray.show()
        if self.tray.isVisible():
            self._tray_retry_attempts = 0
            if self._tray_retry_timer.isActive():
                self._tray_retry_timer.stop()
            logging.info("Tray icon visible.")
            return

        self._tray_retry_attempts += 1
        if self._tray_retry_attempts >= self._tray_retry_max_attempts:
            logging.error("Failed to show tray icon after retries.")
            self._tray_retry_timer.stop()
            return
        if not self._tray_retry_timer.isActive():
            self._tray_retry_timer.start()

    def _run_first_start_if_needed(self) -> None:
        if self._shutting_down:
            return

        marker = get_marker_path()
        wizard_marker = marker.with_name(".first_run_setup_v2_complete")

        # Legacy migration for the old one-time marker.
        legacy_marker = Path(".first_run_complete").resolve()
        old_appdata = os.environ.get("APPDATA")
        old_marker = Path(old_appdata) / "WhisperType" / ".first_run_complete" if old_appdata else None
        if not marker.exists():
            migration_candidates: list[Path] = [legacy_marker]
            if old_marker is not None:
                migration_candidates.append(old_marker)
            for candidate in migration_candidates:
                if not candidate.exists():
                    continue
                marker.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(candidate, marker)
                break

        # Base first-run profile selection (old behavior) if marker does not exist yet.
        if not marker.exists():
            hw = detect_hardware()
            suggested_backend = hw.backend
            cuda_vram_gb = detect_nvidia_vram_gb() if suggested_backend == "cuda" else None
            suggested_model = recommend_model(suggested_backend, cuda_vram_gb=cuda_vram_gb)
            ui_lang = normalize_ui_language(str(self.config.get("general", {}).get("language_ui", "de")))
            default_transcription_lang = "en" if ui_lang == "en" else "de"

            self.config["whisper"]["backend"] = suggested_backend
            self.config["whisper"]["model"] = suggested_model
            if str(self.config["whisper"].get("language", "auto")) == "auto":
                self.config["whisper"]["language"] = default_transcription_lang
            self.config_manager.save(self.config)

            self.transcriber.set_backend(
                suggested_backend,
                self.config["whisper"]["compute_type"],
                download_root=self.config["whisper"]["download_root"],
            )
            self.transcriber.set_model(suggested_model)
            self.transcriber.set_language(str(self.config["whisper"]["language"]))
            self.tray.set_selected(
                model=self.config["whisper"]["model"],
                transcription_language=self.config["whisper"]["language"],
                backend=self.config["whisper"]["backend"],
                ui_language=self.config["general"]["language_ui"],
            )
            self.tray.refresh_status()

            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok", encoding="utf-8")

        # New first-run setup wizard should run once even for upgraded installations.
        if wizard_marker.exists():
            return

        try:
            hw = detect_hardware()
            configured_backend = str(self.config["whisper"]["backend"]).strip().lower()
            detected_backend = hw.backend if hw.backend in {"cpu", "cuda"} else "cpu"
            if configured_backend == "cuda" and detected_backend != "cuda":
                suggested_backend = detected_backend
                self.config["whisper"]["backend"] = suggested_backend
                self.config_manager.save(self.config)
                logging.info(
                    "First-run backend normalized from cuda to %s (hardware detection).",
                    suggested_backend,
                )
            elif configured_backend in {"cpu", "cuda"}:
                suggested_backend = configured_backend
            else:
                suggested_backend = detected_backend
            if configured_backend == "auto":
                self.config["whisper"]["backend"] = suggested_backend
                self.config_manager.save(self.config)

            suggested_model = str(self.config["whisper"]["model"])
            cuda_vram_gb = detect_nvidia_vram_gb() if has_nvidia_gpu() else None

            cuda_ok, _ = check_cuda_runtime_available()
            nvidia_present = has_nvidia_gpu()
            model_status = refresh_model_status(self.config["whisper"]["download_root"])
            recommended_models = self._recommended_first_run_models(suggested_backend, cuda_vram_gb)
            suggested_models = [m for m in recommended_models if not model_status.get(m, False)]
            installed_models = [m for m, installed in model_status.items() if installed]
            logging.info(
                "First-run model status: installed=%s suggested_for_prefetch=%s",
                installed_models,
                suggested_models,
            )

            dialog = FirstRunSetupDialog(
                ui_language=self._ui_language(),
                hardware_device=hw.device_name,
                suggested_backend=suggested_backend,
                suggested_model=suggested_model,
                show_cuda_option=nvidia_present,
                cuda_already_ready=cuda_ok,
                suggested_models=suggested_models,
                installed_models=installed_models,
                backend_hint=suggested_backend,
                download_root=self.config["whisper"]["download_root"],
                install_cuda_cb=install_cuda_runtime_headless,
            )
            dialog.exec()
            payload = dialog.result_payload()

            if bool(payload.get("cuda_success", False)):
                _register_cuda_dll_paths()
                ok, _ = check_cuda_runtime_available()
                if ok:
                    self._set_backend_choice("cuda")
                    preferred_model = recommend_model("cuda", cuda_vram_gb=cuda_vram_gb)
                    self.config["whisper"]["model"] = preferred_model
                    self.config_manager.save(self.config)
                    self.transcriber.set_model(preferred_model)

            failed_models = [str(x) for x in payload.get("failed_models", [])]
            if failed_models:
                failed_text = ", ".join(failed_models)
                self._show_tray_message(self._t("first_run_model_prefetch_failed", models=failed_text))

            if bool(payload.get("restart_recommended", False)):
                QMessageBox.information(
                    None,
                    self._t("cuda_restart_title"),
                    self._t("first_run_restart_recommended"),
                )

            wizard_marker.parent.mkdir(parents=True, exist_ok=True)
            wizard_marker.write_text("ok", encoding="utf-8")
        except Exception:
            logging.exception("First-run optional setup failed.")

    def _status_text_short(self) -> str:
        device_token = self.recorder.get_current_device_token()
        mic_state = (
            self._t("status_mic_default_short")
            if device_token == "default"
            else f"#{device_token}"
        )
        return (
            f"{self.transcriber.device_info()} | "
            f"{self.config['whisper']['model']} | "
            f"{self.config['whisper']['language']} | "
            f"mic:{mic_state}"
        )

    def _on_screen_added(self, screen) -> None:  # noqa: ANN001
        try:
            name = screen.name() if screen is not None else "unknown"
        except Exception:
            name = "unknown"
        logging.info("Screen added: %s", name)
        try:
            self.overlay.place_bottom_center()
        except Exception:
            logging.exception("Failed to reposition overlay after screen add.")

    def _on_screen_removed(self, screen) -> None:  # noqa: ANN001
        try:
            name = screen.name() if screen is not None else "unknown"
        except Exception:
            name = "unknown"
        logging.info("Screen removed: %s", name)
        try:
            preferred = int(self.config.get("overlay", {}).get("monitor_index", -1))
        except Exception:
            preferred = -1
        screens = self.qt_app.screens()
        if preferred >= 0 and preferred >= len(screens):
            logging.info(
                "Preferred monitor index %s currently unavailable; overlay falls back to primary monitor.",
                preferred,
            )
        try:
            self.overlay.place_bottom_center()
        except Exception:
            logging.exception("Failed to reposition overlay after screen removal.")

    def _status_text_full(self) -> str:
        device_token = self.recorder.get_current_device_token()
        mic_state = (
            self._t("status_mic_default_full")
            if device_token == "default"
            else self._t("status_mic_device", token=device_token)
        )
        return (
            f"{self._t('status_backend')}: {self.transcriber.device_info()} | "
            f"{self._t('status_model')}: {self.config['whisper']['model']} | "
            f"{self._t('status_transcription_language')}: {self.config['whisper']['language']} | "
            f"{self._t('status_microphone')}: {mic_state} | "
            f"{self._t('status_hotkey')}: {self.config['hotkey']['combination']}"
        )

    def _audio_devices_for_tray(self) -> tuple[list[dict[str, str | bool]], str]:
        current = self.recorder.get_current_device_token()
        devices: list[dict[str, str | bool]] = [
            {"token": "default", "label": self._t("status_mic_default_full"), "is_default": True}
        ]
        devices.extend(AudioRecorder.list_input_devices())
        return devices, current

    def _model_status_for_tray(self) -> dict[str, bool]:
        return get_model_status(self.config["whisper"]["download_root"])

    def _on_model_install_request(self, model: str) -> None:
        model = str(model)
        if self.recorder.is_running or self._transcription_in_progress:
            self.overlay.show_notice(self._t("model_install_busy"), ms=1100)
            return
        if self._model_install_in_progress:
            self._show_tray_message(self._t("model_install_in_progress"))
            return
        statuses = self._model_status_for_tray()
        if statuses.get(model, False):
            self._on_model_change(model)
            return

        self._model_install_in_progress = True
        self._model_install_target = model
        self.bridge.model_install_started.emit(model)
        worker = threading.Thread(
            target=self._model_install_worker,
            args=(model,),
            daemon=True,
            name=f"model-install-{model}",
        )
        worker.start()

    def _model_install_worker(self, model: str) -> None:
        try:
            ok, reason = ensure_model_installed(
                model=model,
                backend_hint=self.config["whisper"]["backend"],
                download_root=self.config["whisper"]["download_root"],
                progress_cb=lambda percent: self.bridge.model_install_progress.emit(int(percent)),
            )
        except Exception as exc:
            logging.exception("Unhandled exception in model install worker for '%s': %s", model, exc)
            ok, reason = False, str(exc)
        self.bridge.model_install_finished.emit(model, ok, reason)

    def _on_audio_device_change(self, device_token: str) -> None:
        if self.recorder.is_running:
            logging.warning("Audio device switch rejected while recording. Stop recording first.")
            self.tray.refresh_status()
            return

        if not self.recorder.set_device(device_token):
            self.tray.refresh_status()
            return

        self.config["audio"]["device"] = None if device_token == "default" else int(device_token)
        self.config_manager.save(self.config)
        logging.info("Audio device persisted as %s", self.config["audio"]["device"])
        self.tray.refresh_status()

    def _on_model_change(self, model: str) -> None:
        statuses = self._model_status_for_tray()
        if not statuses.get(model, False):
            self._on_model_install_request(model)
            return
        self.config["whisper"]["model"] = model
        self.config_manager.save(self.config)
        self.transcriber.set_model(model)
        self.tray.set_selected(
            model=self.config["whisper"]["model"],
            transcription_language=self.config["whisper"]["language"],
            backend=self.config["whisper"]["backend"],
            ui_language=self.config["general"]["language_ui"],
        )
        self.tray.refresh_status()

    def _on_language_change(self, language: str) -> None:
        self.config["whisper"]["language"] = language
        self.config_manager.save(self.config)
        self.transcriber.set_language(language)
        self.tray.set_selected(
            model=self.config["whisper"]["model"],
            transcription_language=self.config["whisper"]["language"],
            backend=self.config["whisper"]["backend"],
            ui_language=self.config["general"]["language_ui"],
        )
        self.tray.refresh_status()

    def _on_ui_language_change(self, language_ui: str) -> None:
        normalized = normalize_ui_language(language_ui)
        self.config["general"]["language_ui"] = normalized
        self.config_manager.save(self.config)
        self.tray.set_selected(
            model=self.config["whisper"]["model"],
            transcription_language=self.config["whisper"]["language"],
            backend=self.config["whisper"]["backend"],
            ui_language=self.config["general"]["language_ui"],
        )
        self.tray.refresh_status()

    def _on_backend_change(self, backend: str) -> None:
        resolved_backend = self._resolve_backend_request(backend, source="tray")
        if resolved_backend != self.config["whisper"]["backend"]:
            self._set_backend_choice(resolved_backend)
        else:
            self.tray.refresh_status()

    def _on_idle_unload_tick(self) -> None:
        try:
            idle_sec = int(self.config["whisper"].get("unload_after_idle_sec", 300))
            self.transcriber.unload_if_idle(idle_sec)
        except Exception:
            logging.exception("Idle unload check failed.")

    def _check_audio_device_changes(self) -> None:
        """Poll for audio device changes and re-initialise PortAudio when not recording."""
        if self.recorder.is_running:
            return

        try:
            devices = AudioRecorder.list_input_devices()
            fingerprint = frozenset((d.get("token", ""), d.get("label", "")) for d in devices)
        except Exception as exc:
            logging.debug("Audio device check failed: %s", exc)
            return

        if self._audio_device_fingerprint is None:
            self._audio_device_fingerprint = fingerprint
            return

        if fingerprint == self._audio_device_fingerprint:
            return

        logging.info("Audio device list changed – re-initialising PortAudio.")
        self._audio_device_fingerprint = fingerprint

        try:
            import sounddevice as _sd  # noqa: PLC0415
            _sd._terminate()
            _sd._initialize()
            logging.info("PortAudio re-initialised.")
        except Exception as exc:
            logging.warning("PortAudio reinit skipped: %s", exc)

        self.tray.refresh_status()

    def _cancel_transcribing_delay(self) -> None:
        if self._transcribing_delay_timer.isActive():
            self._transcribing_delay_timer.stop()

    def _cancel_transcription_timeout(self) -> None:
        if self._transcription_timeout_timer.isActive():
            self._transcription_timeout_timer.stop()

    def _on_transcribing_delay_elapsed(self) -> None:
        if self._transcription_in_progress and not self._model_loading_active:
            self.overlay.show_transcribing()

    def _on_model_loading_start(self, model: str, mode: str) -> None:
        if not self._transcription_in_progress:
            return
        self._model_loading_active = True
        self._model_loading_model = str(model)
        self._model_loading_mode = str(mode or "download")
        self.overlay.show_model_warmup(self._t("overlay_loading_model", model=model))
        self.overlay.set_loading_progress(0)

    def _on_model_loading_progress(self, progress: int) -> None:
        if not self._transcription_in_progress:
            return
        if not self._model_loading_active:
            return
        self.overlay.set_loading_progress(progress)

    def _on_model_install_started(self, model: str) -> None:
        self.overlay.show_loading(self._t("overlay_downloading_simple"))
        self.overlay.set_loading_progress(0)

    def _on_model_install_progress(self, progress: int) -> None:
        if not self._model_install_in_progress:
            return
        self.overlay.set_loading_progress(progress)

    def _on_model_install_finished(self, model: str, success: bool, reason: str) -> None:
        self._model_install_in_progress = False
        self._model_install_target = None

        if success:
            self.config["whisper"]["model"] = model
            self.config_manager.save(self.config)
            self.transcriber.set_model(model)
            self.overlay.show_done(self.config["overlay"]["display_duration_ms"])
            self._show_tray_message(self._t("model_install_success", model=model))
        else:
            logging.warning("Model install failed for '%s': %s", model, reason)
            self.overlay.show_error(self._t("model_install_failed", model=model), ms=1400)
            self._show_tray_message(self._t("model_install_failed", model=model))

        self.tray.refresh_status()
        self.tray.set_selected(
            model=self.config["whisper"]["model"],
            transcription_language=self.config["whisper"]["language"],
            backend=self.config["whisper"]["backend"],
            ui_language=self.config["general"]["language_ui"],
        )

    def _on_model_loading_done(self) -> None:
        self.overlay.set_loading_progress(100)
        self._model_loading_active = False
        loaded_model = self._model_loading_model
        self._model_loading_model = None
        self._model_loading_mode = "download"
        if loaded_model:
            if loaded_model != str(self.config["whisper"].get("model", "")):
                self.config["whisper"]["model"] = loaded_model
                self.config_manager.save(self.config)
            self.tray.set_selected(
                model=self.config["whisper"]["model"],
                transcription_language=self.config["whisper"]["language"],
                backend=self.config["whisper"]["backend"],
                ui_language=self.config["general"]["language_ui"],
            )
        self.tray.refresh_status()
        if not self._transcription_in_progress:
            return
        self._cancel_transcribing_delay()
        logging.info("Model warmup finish animation started.")
        self.overlay.finish_model_warmup(ms=220, on_finished=self._on_model_warmup_finish_complete)

    def _on_model_warmup_finish_complete(self) -> None:
        logging.info("Model warmup finish animation completed.")
        if self._transcription_in_progress and not self._model_loading_active:
            self.overlay.show_transcribing()

    def _on_transcription_timeout(self) -> None:
        if not self._transcription_in_progress:
            return
        logging.warning("Transcription timeout after %sms", self._transcription_timeout_ms)
        self._discard_timed_out_result = True
        self._model_loading_active = False
        self._model_loading_mode = "download"
        self._transcription_in_progress = False
        self.bridge.transcription_settled.emit()
        self.overlay.show_warning(self._t("overlay_transcription_timeout"), ms=1500)

    def _open_logs(self) -> None:
        raw_log_dir = self.config["general"].get("log_dir") or str(get_log_dir())
        log_dir = resolve_user_path(raw_log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(os, "startfile"):
            os.startfile(str(log_dir))  # type: ignore[attr-defined]
            return
        logging.info("Open logs requested: %s", log_dir)

    def _open_config(self) -> None:
        path = self.config_manager.path.resolve()
        if hasattr(os, "startfile"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        logging.info("Open config requested: %s", path)

    def _start_recording(self) -> None:
        if self.recorder.is_running:
            return
        logging.info("Hotkey down -> start recording")
        self._recording_started_at = time.monotonic()
        self.recorder.start()
        if not self.recorder.is_running:
            logging.info("Recorder start failed / no microphone.")
            self.overlay.show_warning(self._t("overlay_no_microphone"), ms=1200)
            return
        self.overlay.show_recording()

    def _stop_recording(self) -> None:
        if not self.recorder.is_running:
            if self.overlay.state == "recording":
                self.overlay.hide_immediate()
            return
        logging.info("Hotkey up -> stop recording")

        elapsed_ms = (time.monotonic() - self._recording_started_at) * 1000
        if elapsed_ms < self.config["hotkey"]["debounce_ms"]:
            audio = self.recorder.stop()
            _ = audio
            self.overlay.hide_immediate()
            return

        audio = self.recorder.stop()
        if audio.size == 0:
            self.overlay.show_warning(self._t("overlay_no_audio"), ms=1200)
            return

        if self.config["audio"].get("vad_enabled", True):
            vad_result = has_speech(
                audio=audio,
                sample_rate=int(self.config["audio"]["sample_rate"]),
                raw_threshold=float(self.config["audio"].get("vad_threshold", 0.5)),
                min_speech_ms=int(self.config["audio"].get("vad_min_speech_ms", 120)),
            )
            logging.info(
                "VAD precheck: speech=%s threshold=%.5f active_ms=%d duration_ms=%d",
                vad_result.speech,
                vad_result.effective_threshold,
                vad_result.active_ms,
                vad_result.duration_ms,
            )
            if not vad_result.speech:
                self.overlay.show_notice(self._t("overlay_no_speech"), ms=950)
                return

        if not self._transcribe_lock.acquire(blocking=False):
            self.overlay.show_notice(self._t("overlay_busy"), ms=1000)
            return

        self._transcription_in_progress = True
        self._model_loading_active = False
        self._discard_timed_out_result = False
        self._cancel_transcribing_delay()
        self._cancel_transcription_timeout()
        if self._transcription_timeout_ms > 0:
            self._transcription_timeout_timer.start(self._transcription_timeout_ms)
        if self._transcribing_delay_ms <= 0:
            self.overlay.show_transcribing()
        else:
            self._transcribing_delay_timer.start(self._transcribing_delay_ms)

        self._submit_transcription_job(audio)

    def _submit_transcription_job(self, audio: np.ndarray) -> None:
        try:
            future = self.transcriber.transcribe_async(audio)
            future.add_done_callback(self._on_transcription_done)
        except Exception as exc:
            logging.exception("Failed to submit transcription job")
            self._transcription_in_progress = False
            self._model_loading_active = False
            self._discard_timed_out_result = False
            self._cancel_transcribing_delay()
            self._cancel_transcription_timeout()
            self.bridge.transcription_error.emit(str(exc)[:80])
            self.bridge.transcription_settled.emit()
            if self._transcribe_lock.locked():
                self._transcribe_lock.release()

    def _on_transcription_done(self, future) -> None:  # noqa: ANN001
        try:
            text = future.result()
            if self._discard_timed_out_result:
                logging.warning("Discarding late transcription result after timeout.")
            elif text:
                self.bridge.transcription_done.emit(text)
            else:
                self.bridge.transcription_error.emit(self._t("overlay_empty_transcription"))
        except Exception as exc:
            logging.exception("Transcription failed")
            if not self._discard_timed_out_result:
                self.bridge.transcription_error.emit(str(exc)[:80])
        finally:
            self._transcription_in_progress = False
            self._model_loading_active = False
            self._discard_timed_out_result = False
            self.bridge.transcription_settled.emit()
            self._transcribe_lock.release()

    def _rescue_enabled(self) -> bool:
        return bool(self.config.get("insertion", {}).get("rescue_enabled", True))

    def _rescue_copy_available(self) -> bool:
        if not self._rescue_enabled():
            return False
        payload = self._rescue_text.strip()
        if not payload:
            return False
        if self.config.get("insertion", {}).get("rescue_never_expire", False):
            return True
        expires_at = float(self._rescue_expires_at or 0.0)
        return expires_at <= 0.0 or time.monotonic() < expires_at

    def _clear_rescue_text(self) -> None:
        self._rescue_text = ""
        self._rescue_expires_at = 0.0
        if self._rescue_expire_timer.isActive():
            self._rescue_expire_timer.stop()
        if hasattr(self, "tray"):
            self.tray.refresh_status()

    def _reschedule_rescue_expiry(self) -> None:
        if self._rescue_expire_timer.isActive():
            self._rescue_expire_timer.stop()
        if not self._rescue_enabled():
            self._rescue_expires_at = 0.0
            return
        if bool(self.config.get("insertion", {}).get("rescue_never_expire", False)):
            self._rescue_expires_at = 0.0
            return
        timeout_sec = max(1, int(self.config.get("insertion", {}).get("rescue_timeout_sec", 120) or 120))
        self._rescue_expires_at = time.monotonic() + float(timeout_sec)
        self._rescue_expire_timer.start(timeout_sec * 1000)

    def _store_rescue_text(self, text: str) -> None:
        if not self._rescue_enabled():
            self._clear_rescue_text()
            return
        payload = text.strip()
        if not payload:
            return
        self._rescue_text = payload
        self._reschedule_rescue_expiry()
        if hasattr(self, "tray"):
            self.tray.refresh_status()

    def _on_rescue_expired(self) -> None:
        logging.info("Rescue memory expired.")
        self._clear_rescue_text()

    def _copy_last_dictation_to_clipboard(self) -> None:
        if not self._rescue_copy_available():
            self._show_tray_message(self._t("rescue_copy_unavailable"))
            self.tray.refresh_status()
            return
        if self.inserter.copy_to_clipboard(self._rescue_text):
            self._show_tray_message(self._t("rescue_copy_success"))
        else:
            self._show_tray_message(self._t("rescue_copy_unavailable"))
        self.tray.refresh_status()

    def _insert_transcription_payload(self, payload: str) -> None:
        try:
            self.inserter.insert(payload)
        except Exception:
            logging.exception("Failed to insert transcription payload.")
            self.overlay.show_error(self._t("tray_error_prefix"), ms=1400)

    def _apply_transcription(self, text: str) -> None:
        payload = text.strip()
        if not payload:
            self.overlay.show_notice(self._t("overlay_empty_transcription"), ms=1200)
            return
        self._store_rescue_text(payload)
        self.overlay.show_done(self.config["overlay"]["display_duration_ms"])
        QTimer.singleShot(20, lambda payload=payload: self._insert_transcription_payload(payload))

    def shutdown(self) -> None:
        if self._shutting_down:
            return
        self._shutting_down = True
        logging.info("Shutdown requested")

        self._idle_unload_timer.stop()
        self._device_monitor_timer.stop()
        self._tray_retry_timer.stop()
        self._rescue_expire_timer.stop()
        self._clear_rescue_text()
        self._cancel_transcribing_delay()
        self._cancel_transcription_timeout()
        if self._settings_dialog and self._settings_dialog.isVisible():
            try:
                self._settings_dialog.close()
            except Exception:
                logging.exception("Failed to close settings dialog during shutdown.")

        try:
            self.hotkey.stop()
        except Exception:
            logging.exception("Failed to stop hotkey cleanly during shutdown.")

        try:
            self.tray.hide()
        except Exception:
            logging.exception("Failed to hide tray icon during shutdown.")

        try:
            self.transcriber.shutdown()
        except Exception:
            logging.exception("Failed to stop transcriber cleanly during shutdown.")

        try:
            if self._fault_log_handle is not None:
                self._fault_log_handle.flush()
                self._fault_log_handle.close()
                self._fault_log_handle = None
        except Exception:
            logging.debug("Failed to close fault log handle.")
        try:
            self._debug_clear_critical_step("shutdown")
            self._close_debug_trace()
        except Exception:
            logging.debug("Failed to close debug trace cleanly.")

        self.qt_app.quit()

    def run(self) -> int:
        signal.signal(signal.SIGINT, lambda *_: self.shutdown())
        return self.qt_app.exec()


def _setup_cli_logging(config: dict) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    try:
        if bool(config.get("general", {}).get("log_to_file", True)):
            raw_log_dir = config.get("general", {}).get("log_dir") or str(get_log_dir())
            log_dir = resolve_user_path(raw_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"whisply-{time.strftime('%Y%m%d')}.log"
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    except Exception:
        pass

    level = logging.DEBUG if bool(config.get("general", {}).get("debug_logging", False)) else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
        force=True,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--prefetch-model", choices=list(SUPPORTED_MODELS))
    parser.add_argument("--list-model-status", action="store_true")
    parser.add_argument("--json", dest="as_json", action="store_true")
    parser.add_argument("--install-cuda-runtime", action="store_true")
    parser.add_argument("--check-cuda", action="store_true")
    return parser


def _run_cli(args: argparse.Namespace) -> int:
    config = ConfigManager(get_config_path()).load()
    _setup_cli_logging(config)

    download_root = config["whisper"]["download_root"]
    backend_hint = str(config["whisper"].get("backend", "auto"))

    if args.list_model_status:
        status = get_model_status(download_root)
        if args.as_json:
            print(json.dumps(status, ensure_ascii=False))
        else:
            for model in SUPPORTED_MODELS:
                state = "installed" if status.get(model, False) else "missing"
                print(f"{model}: {state}")
        return 0

    if args.prefetch_model:
        logging.info("Prefetch model requested: %s", args.prefetch_model)
        ok, reason = ensure_model_installed(
            model=str(args.prefetch_model),
            backend_hint=backend_hint,
            download_root=download_root,
            progress_cb=lambda value: logging.info("Model prefetch progress: %s%%", value),
        )
        if ok:
            print(f"ok:{args.prefetch_model}:{reason}")
            return 0
        print(f"error:{args.prefetch_model}:{reason}")
        return 1

    if args.install_cuda_runtime:
        logging.info("Headless CUDA runtime install requested.")
        ok, reason = install_cuda_runtime_headless(
            progress_cb=lambda value: logging.info("CUDA runtime install progress: %s%%", value)
        )
        if not ok:
            print(f"error:cuda_runtime_install:{reason}")
            return 1
        _register_cuda_dll_paths()
        cuda_ok, cuda_reason = check_cuda_runtime_available()
        if not cuda_ok:
            print(f"error:cuda_check_after_install:{cuda_reason}")
            return 1
        print("ok:cuda_runtime_install")
        return 0

    if args.check_cuda:
        ok, reason = check_cuda_runtime_available()
        if args.as_json:
            print(json.dumps({"ok": ok, "reason": reason}, ensure_ascii=False))
        else:
            print(f"ok={ok} reason={reason}")
        return 0 if ok else 1

    return 2


if __name__ == "__main__":
    parser = _build_cli_parser()
    parsed_args, unknown_args = parser.parse_known_args()
    has_cli_action = bool(
        parsed_args.prefetch_model
        or parsed_args.list_model_status
        or parsed_args.install_cuda_runtime
        or parsed_args.check_cuda
    )
    if has_cli_action:
        if unknown_args:
            print(f"Unknown arguments: {' '.join(unknown_args)}")
            raise SystemExit(1)
        raise SystemExit(_run_cli(parsed_args))

    app = WhisperTypeApp()
    raise SystemExit(app.run())
