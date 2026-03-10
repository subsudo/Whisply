from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import sys

from PySide6.QtGui import QIcon
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from i18n import normalize_ui_language, tr
from model_store import SUPPORTED_MODELS, ensure_model_installed

log = logging.getLogger(__name__)


_QSS = """
QDialog {
    background: #1f232a;
    color: #e9edf3;
}

QLabel {
    color: #8892a4;
    font-size: 9pt;
}

QGroupBox {
    background: #252b35;
    border: 1px solid #2d3340;
    border-radius: 8px;
    margin-top: 20px;
    padding: 8px 10px 10px 10px;
    font-size: 8pt;
    font-weight: 700;
    color: #6b7585;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
    color: #6b7585;
    font-size: 8pt;
    font-weight: 700;
    background: transparent;
}

QCheckBox {
    color: #8892a4;
    font-size: 9pt;
    spacing: 8px;
    padding: 1px 0;
}
QCheckBox:disabled {
    color: #4a5568;
}
QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #4c566a;
    border-radius: 3px;
    background: #1f232a;
}
QCheckBox::indicator:checked {
    background: #7a8b9c;
    border-color: #7a8b9c;
    image: url("__CHECKMARK_URI__");
}
QCheckBox::indicator:checked:disabled {
    background: #5f6a79;
    border-color: #5f6a79;
    image: url("__CHECKMARK_URI__");
}
QCheckBox::indicator:disabled {
    background: #1a1e26;
    border-color: #2a303a;
}

QProgressBar {
    background: #1f232a;
    border: 1px solid #384151;
    border-radius: 6px;
    text-align: center;
    color: #c8d0dc;
    min-height: 18px;
}
QProgressBar::chunk {
    background: #3a4455;
    border-radius: 5px;
}

QPushButton {
    background: #2a303a;
    border: 1px solid #384151;
    border-radius: 6px;
    padding: 6px 12px;
    color: #b8c0cc;
    font-size: 9pt;
    min-width: 95px;
}
QPushButton:hover {
    background: #323b48;
    border-color: #4d5a70;
}
QPushButton:disabled {
    background: #1e232b;
    border-color: #2a303a;
    color: #4a5568;
}

QFrame[frameShape="4"] {
    background: #2d3340;
    border: none;
    max-height: 1px;
}
"""

_QSS_PRIMARY_BUTTON = """
QPushButton {
    background: #3a4455;
    border: 1px solid #5a6a80;
    border-radius: 6px;
    color: #c8d4e0;
    font-weight: 600;
    padding: 6px 16px;
}
QPushButton:hover {
    background: #454f65;
    border-color: #6a7a90;
}
"""

_QSS_SECONDARY_BUTTON = """
QPushButton {
    background: transparent;
    border-color: transparent;
    color: #6b7585;
    padding: 6px 12px;
}
QPushButton:hover {
    background: #252b35;
    border-color: #384151;
    color: #9aa3b0;
}
"""


class _FirstRunWorker(QThread):
    cuda_progress = Signal(int)
    model_started = Signal(str, int, int)
    model_progress = Signal(int)
    model_done = Signal(str, bool, str)
    done = Signal(bool, bool, list, list)

    def __init__(
        self,
        install_cuda: bool,
        install_cuda_cb: Callable[[Callable[[int], None] | None], tuple[bool, str]],
        models: list[str],
        backend_hint: str,
        download_root: str,
    ) -> None:
        super().__init__()
        self._install_cuda = install_cuda
        self._install_cuda_cb = install_cuda_cb
        self._models = models
        self._backend_hint = backend_hint
        self._download_root = download_root

    def run(self) -> None:
        cuda_ok = True
        try:
            if self._install_cuda:
                cuda_ok, _ = self._install_cuda_cb(lambda p: self.cuda_progress.emit(int(p)))
        except Exception as exc:
            log.exception("First-run CUDA install step failed: %s", exc)
            cuda_ok = False

        installed: list[str] = []
        failed: list[str] = []
        total = len(self._models)
        for idx, model in enumerate(self._models, start=1):
            if self.isInterruptionRequested():
                break
            self.model_started.emit(model, idx, total)
            try:
                ok, reason = ensure_model_installed(
                    model=model,
                    backend_hint=self._backend_hint,
                    download_root=self._download_root,
                    progress_cb=lambda p: self.model_progress.emit(int(p)),
                )
            except Exception as exc:
                log.exception("First-run model install crashed for '%s': %s", model, exc)
                ok, reason = False, str(exc)
            self.model_done.emit(model, ok, reason)
            if ok:
                installed.append(model)
            else:
                failed.append(model)

        self.done.emit(self._install_cuda, cuda_ok, installed, failed)


class FirstRunSetupDialog(QDialog):
    def __init__(
        self,
        ui_language: str,
        hardware_device: str,
        suggested_backend: str,
        suggested_model: str,
        show_cuda_option: bool,
        cuda_already_ready: bool,
        suggested_models: list[str],
        installed_models: list[str],
        backend_hint: str,
        download_root: str,
        install_cuda_cb: Callable[[Callable[[int], None] | None], tuple[bool, str]],
    ) -> None:
        super().__init__()
        self._lang = normalize_ui_language(ui_language)
        self._backend_hint = backend_hint
        self._download_root = download_root
        self._install_cuda_cb = install_cuda_cb
        self._worker: _FirstRunWorker | None = None
        self._preinstalled_models = {str(m).strip().lower() for m in installed_models}

        self.cuda_attempted = False
        self.cuda_success = False
        self.installed_models: list[str] = []
        self.failed_models: list[str] = []
        self.restart_recommended = False
        self._selected_install_models: list[str] = []

        self.setWindowTitle(self._t("first_run_title"))
        self._set_window_icon()
        self.setModal(True)
        self.resize(620, 560)
        self.setStyleSheet(_QSS.replace("__CHECKMARK_URI__", self._checkmark_uri()))

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        title = QLabel(self._t("first_run_title"))
        title.setStyleSheet("font-size: 17px; font-weight: 700; color: #dbe2ee;")
        subtitle = QLabel(self._t("first_run_setup_subtitle"))
        subtitle.setStyleSheet("color: #8b96a8;")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        summary_group = QGroupBox(self._t("settings_group_recognition").upper())
        summary_form = QFormLayout(summary_group)
        summary_form.setContentsMargins(6, 4, 6, 4)
        summary_form.setHorizontalSpacing(16)
        summary_form.setVerticalSpacing(5)
        summary_form.addRow(
            QLabel(self._t("settings_label_backend")),
            QLabel(str(suggested_backend)),
        )
        summary_form.addRow(
            QLabel(self._t("settings_label_model")),
            QLabel(str(suggested_model)),
        )
        summary_form.addRow(
            QLabel(self._t("first_run_label_hardware")),
            QLabel(str(hardware_device)),
        )
        root.addWidget(summary_group)

        options_group = QGroupBox(self._t("settings_group_general").upper())
        options_layout = QVBoxLayout(options_group)
        options_layout.setSpacing(6)

        self.cuda_checkbox = QCheckBox(self._t("first_run_cuda_optional"))
        self.cuda_checkbox.setChecked(show_cuda_option and not cuda_already_ready)
        self.cuda_checkbox.setVisible(show_cuda_option and not cuda_already_ready)
        options_layout.addWidget(self.cuda_checkbox)

        self.cuda_ready_label = QLabel(self._t("first_run_cuda_ready"))
        self.cuda_ready_label.setVisible(cuda_already_ready)
        options_layout.addWidget(self.cuda_ready_label)

        self.cuda_unavailable_label = QLabel(self._t("first_run_cuda_unavailable"))
        self.cuda_unavailable_label.setVisible(not show_cuda_option and not cuda_already_ready)
        options_layout.addWidget(self.cuda_unavailable_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFixedHeight(1)
        options_layout.addWidget(sep)

        models_label = QLabel(self._t("first_run_models_label"))
        models_label.setStyleSheet("color: #a1adbf; font-weight: 600;")
        options_layout.addWidget(models_label)
        self.model_checks: dict[str, QCheckBox] = {}
        for model in SUPPORTED_MODELS:
            is_installed = model in self._preinstalled_models
            label = (
                f"{model} {self._t('first_run_model_installed_suffix')}"
                if is_installed
                else model
            )
            check = QCheckBox(label)
            check.setChecked(is_installed or (model in suggested_models))
            check.setProperty("preinstalled", bool(is_installed))
            if is_installed:
                check.setEnabled(False)
            self.model_checks[model] = check
            options_layout.addWidget(check)

        root.addWidget(options_group)

        progress_group = QGroupBox(self._t("settings_group_overlay").upper())
        progress_layout = QFormLayout(progress_group)
        progress_layout.setContentsMargins(6, 4, 6, 4)
        progress_layout.setHorizontalSpacing(16)
        progress_layout.setVerticalSpacing(8)
        self.cuda_progress = QProgressBar()
        self.cuda_progress.setRange(0, 100)
        self.cuda_progress.setValue(0)
        self.models_progress = QProgressBar()
        self.models_progress.setRange(0, 100)
        self.models_progress.setValue(0)
        self.current_model_progress = QProgressBar()
        self.current_model_progress.setRange(0, 100)
        self.current_model_progress.setValue(0)
        self.current_model_label = QLabel("-")

        progress_layout.addRow(self._t("first_run_progress_cuda"), self.cuda_progress)
        progress_layout.addRow(self._t("first_run_progress_models"), self.models_progress)
        progress_layout.addRow(self._t("first_run_progress_model_current"), self.current_model_progress)
        progress_layout.addRow("", self.current_model_label)
        root.addWidget(progress_group)

        self.status_label = QLabel(self._t("first_run_status_idle"))
        self.status_label.setStyleSheet("color: #9ba8ba;")
        root.addWidget(self.status_label)

        buttons = QDialogButtonBox()
        buttons.setCenterButtons(False)
        self.start_button = QPushButton(self._t("first_run_start_setup"))
        self.skip_button = QPushButton(self._t("first_run_skip_setup"))
        self.start_button.setStyleSheet(_QSS_PRIMARY_BUTTON)
        self.skip_button.setStyleSheet(_QSS_SECONDARY_BUTTON)
        self.start_button.clicked.connect(self._start_setup)
        self.skip_button.clicked.connect(self.reject)
        buttons.addButton(self.skip_button, QDialogButtonBox.RejectRole)
        buttons.addButton(self.start_button, QDialogButtonBox.AcceptRole)
        root.addWidget(buttons)

    def _t(self, key: str, **kwargs: object) -> str:
        return tr(self._lang, key, **kwargs)

    def _set_window_icon(self) -> None:
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        icon_candidates = [
            base_dir / "assets" / "icon.ico",
            Path("assets/icon.ico").resolve(),
            base_dir / "assets" / "icon.png",
            Path("assets/icon.png").resolve(),
        ]
        for icon_path in icon_candidates:
            if not icon_path.exists():
                continue
            self.setWindowIcon(QIcon(str(icon_path)))
            break

    def _checkmark_uri(self) -> str:
        base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
        candidates = [
            base_dir / "assets" / "checkmark.png",
            Path("assets/checkmark.png").resolve(),
            base_dir / "assets" / "checkmark.svg",
            Path("assets/checkmark.svg").resolve(),
        ]
        for path in candidates:
            if path.exists():
                try:
                    return str(path.resolve()).replace("\\", "/")
                except Exception:
                    return str(path).replace("\\", "/")
        return ""

    def _selected_models(self) -> list[str]:
        selected: list[str] = []
        for model, check in self.model_checks.items():
            if bool(check.property("preinstalled")):
                continue
            if check.isEnabled() and check.isChecked():
                selected.append(model)
        return selected

    def _set_options_enabled(self, enabled: bool) -> None:
        self.cuda_checkbox.setEnabled(enabled)
        for check in self.model_checks.values():
            if bool(check.property("preinstalled")):
                check.setEnabled(False)
                continue
            check.setEnabled(enabled)
        self.start_button.setEnabled(enabled)

    def _start_setup(self) -> None:
        selected_models = self._selected_models()
        do_cuda = self.cuda_checkbox.isVisible() and self.cuda_checkbox.isChecked()
        if not do_cuda and not selected_models:
            self.accept()
            return

        self._set_options_enabled(False)
        self.skip_button.setEnabled(False)
        self.status_label.setText(self._t("first_run_status_running"))
        self.current_model_label.setText("-")
        self.cuda_progress.setValue(0)
        self.models_progress.setValue(0)
        self.current_model_progress.setValue(0)
        self._selected_install_models = list(selected_models)

        self._worker = _FirstRunWorker(
            install_cuda=do_cuda,
            install_cuda_cb=self._install_cuda_cb,
            models=selected_models,
            backend_hint=self._backend_hint,
            download_root=self._download_root,
        )
        self._worker.cuda_progress.connect(self.cuda_progress.setValue)
        self._worker.model_started.connect(self._on_model_started)
        self._worker.model_progress.connect(self.current_model_progress.setValue)
        self._worker.model_done.connect(self._on_model_done)
        self._worker.done.connect(self._on_done)
        self._worker.start()

    def _on_model_started(self, model: str, index: int, total: int) -> None:
        self.current_model_label.setText(f"{model} ({index}/{total})")
        self.current_model_progress.setValue(0)

    def _on_model_done(self, model: str, ok: bool, reason: str) -> None:
        if ok:
            if model not in self.installed_models:
                self.installed_models.append(model)
        else:
            if model not in self.failed_models:
                self.failed_models.append(model)
        done = len(self.installed_models) + len(self.failed_models)
        total = max(1, len(self._selected_install_models))
        self.models_progress.setValue(int((done / total) * 100))

    def _on_done(self, cuda_attempted: bool, cuda_ok: bool, installed: list, failed: list) -> None:
        self.cuda_attempted = bool(cuda_attempted)
        self.cuda_success = bool(cuda_ok) if cuda_attempted else False
        self.installed_models = [str(x) for x in installed]
        self.failed_models = [str(x) for x in failed]
        self.restart_recommended = self.cuda_success

        if self.failed_models or (self.cuda_attempted and not self.cuda_success):
            self.status_label.setText(self._t("first_run_status_failed"))
        else:
            self.status_label.setText(self._t("first_run_status_done"))

        self.start_button.hide()
        self.skip_button.setEnabled(True)
        self.skip_button.setText(self._t("common_ok"))
        self.skip_button.clicked.disconnect()
        self.skip_button.clicked.connect(self.accept)

    def result_payload(self) -> dict[str, object]:
        return {
            "cuda_attempted": self.cuda_attempted,
            "cuda_success": self.cuda_success,
            "installed_models": list(self.installed_models),
            "failed_models": list(self.failed_models),
            "restart_recommended": self.restart_recommended,
        }

    def closeEvent(self, event) -> None:  # noqa: ANN001
        if self._worker is not None and self._worker.isRunning():
            self._worker.requestInterruption()
            if not self._worker.wait(3000):
                # Keep dialog alive until worker has stopped to avoid destroying a running thread.
                self.status_label.setText(self._t("first_run_status_running"))
                event.ignore()
                return
        super().closeEvent(event)
