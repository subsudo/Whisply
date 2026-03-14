from __future__ import annotations

from collections.abc import Callable
from html import escape
from pathlib import Path
import logging
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from hotkey import validate_hotkey_combination
from i18n import normalize_ui_language, tr

log = logging.getLogger(__name__)

# ── Stylesheet ────────────────────────────────────────────────────────────────
_QSS = """
QDialog {
    background: #1f232a;
    color: #e9edf3;
}

QGroupBox {
    background: #252b35;
    border: 1px solid #2d3340;
    border-radius: 8px;
    margin-top: 20px;
    padding: 6px 10px 8px 10px;
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

QLabel { color: #8892a4; font-size: 9pt; }
QLabel:disabled { color: #4a5568; }

QLineEdit, QComboBox {
    background: #1f232a;
    border: 1px solid #384151;
    border-radius: 6px;
    padding: 5px 8px;
    color: #d8dde8;
    font-size: 9pt;
}
QLineEdit:hover, QComboBox:hover { border-color: #4d5a70; }
QLineEdit:focus, QComboBox:focus { border-color: #505e78; }
QLineEdit:disabled, QComboBox:disabled {
    background: #1a1e26;
    border-color: #2a303a;
    color: #4a5568;
}

QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    background: #1f232a;
    border: 1px solid #384151;
    color: #d8dde8;
    selection-background-color: #2d3340;
}

QSpinBox {
    background: #1f232a;
    border: 1px solid #384151;
    border-radius: 6px;
    padding: 5px 7px;
    color: #d8dde8;
    font-size: 9pt;
}
QSpinBox:hover { border-color: #4d5a70; }
QSpinBox:disabled { background: #1a1e26; border-color: #2a303a; color: #4a5568; }
QSpinBox::up-button, QSpinBox::down-button { width: 0; }

QCheckBox { color: #8892a4; font-size: 9pt; spacing: 8px; }
QCheckBox:disabled { color: #4a5568; }
QCheckBox::indicator {
    width: 14px; height: 14px;
    border: 1px solid #4c566a;
    border-radius: 3px;
    background: #1f232a;
}
QCheckBox::indicator:checked {
    background: #7a8b9c;
    border-color: #7a8b9c;
    image: url("__CHECKMARK_URI__");
}
QCheckBox::indicator:checked:hover { background: #8a9bac; }
QCheckBox::indicator:checked:disabled {
    background: #5f6a79;
    border-color: #5f6a79;
    image: url("__CHECKMARK_URI__");
}
QCheckBox::indicator:hover { border-color: #6b7a8a; }
QCheckBox::indicator:disabled { background: #1a1e26; border-color: #2a303a; }

QPushButton {
    background: #2a303a;
    border: 1px solid #384151;
    border-radius: 6px;
    padding: 5px 12px;
    color: #b8c0cc;
    font-size: 9pt;
}
QPushButton:hover { background: #323b48; border-color: #4d5a70; }
QPushButton:disabled { background: #1e232b; border-color: #2a303a; color: #4a5568; }

QPushButton[infoButton="true"] {
    min-width: 16px;
    max-width: 16px;
    min-height: 16px;
    max-height: 16px;
    padding: 0px;
    border-radius: 8px;
    background: #2a303a;
    border: 1px solid #445066;
    color: #9fb0c4;
    font-size: 8pt;
    font-weight: 700;
}
QPushButton[infoButton="true"]:hover {
    background: #323b48;
    border-color: #5b6a80;
    color: #d4dfec;
}

QFrame[frameShape="4"] {
    background: #2d3340;
    border: none;
    max-height: 1px;
}
"""

_QSS_SAVE = """
QPushButton {
    background: #3a4455;
    border: 1px solid #5a6a80;
    border-radius: 6px;
    color: #c8d4e0;
    font-weight: 600;
    padding: 6px 20px;
    min-width: 80px;
    font-size: 9pt;
}
QPushButton:hover { background: #454f65; border-color: #6a7a90; }
"""

_QSS_CANCEL = """
QPushButton {
    background: transparent;
    border-color: transparent;
    color: #6b7585;
    padding: 6px 16px;
    font-size: 9pt;
}
QPushButton:hover { background: #252b35; border-color: #384151; color: #9aa3b0; }
"""

_QSS_CUDA_BADGE = (
    "color: #8da5be; padding: 2px 9px; border-radius: 10px; "
    "background-color: #1e2a3a; border: 1px solid #2c3a52; font-size: 9pt;"
)

_FORM_LABEL_WIDTH = 148


class SettingsDialog(QDialog):
    def __init__(
        self,
        config: dict,
        on_save: Callable[[dict[str, object]], None],
        cuda_status_provider: Callable[[], dict[str, str | bool]] | None = None,
        on_cuda_download: Callable[[], bool] | None = None,
        model_status_provider: Callable[[], dict[str, bool]] | None = None,
        on_open_logs: Callable[[], None] | None = None,
        on_open_config: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_save = on_save
        self._cuda_status_provider = cuda_status_provider
        self._on_cuda_download = on_cuda_download
        self._model_status_provider = model_status_provider
        self._on_open_logs = on_open_logs
        self._on_open_config = on_open_config
        self._ui_language = normalize_ui_language(config.get("general", {}).get("language_ui", "de"))
        self.setWindowTitle(self._t("settings_title"))
        self._set_window_icon()
        self.setModal(True)
        self.resize(500, 520)
        self.setStyleSheet(_QSS.replace("__CHECKMARK_URI__", self._checkmark_uri()))

        def tooltip(text: str) -> str:
            safe = escape(str(text or ""))
            return (
                "<div style='max-width: 260px; white-space: normal; "
                "color: #d8dde8; background: #252b35;'>"
                f"{safe}</div>"
            )

        def mk_form_label(text: str) -> QLabel:
            label = QLabel(text)
            label.setFixedWidth(_FORM_LABEL_WIDTH)
            return label

        def mk_form_label_with_info(text: str, tooltip_text: str | None = None) -> QWidget:
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(6)
            label = QLabel(text)
            label.setFixedWidth(_FORM_LABEL_WIDTH - (22 if tooltip_text else 0))
            layout.addWidget(label)
            if tooltip_text:
                info_btn = QPushButton("i")
                info_btn.setProperty("infoButton", True)
                info_btn.setFocusPolicy(Qt.NoFocus)
                info_btn.setToolTip(tooltip(tooltip_text))
                info_btn.setCursor(Qt.PointingHandCursor)
                layout.addWidget(info_btn)
            layout.addStretch()
            container.setFixedWidth(_FORM_LABEL_WIDTH)
            return container

        def add_row(form: QFormLayout, label_text: str, field: QWidget, tooltip: str | None = None) -> None:
            if tooltip:
                form.addRow(mk_form_label_with_info(label_text, tooltip), field)
            else:
                form.addRow(mk_form_label(label_text), field)

        whisper_cfg = config["whisper"]
        hotkey_cfg  = config["hotkey"]
        overlay_cfg = config["overlay"]
        general_cfg = config["general"]
        insertion_cfg = config["insertion"]

        # ── Erkennung ─────────────────────────────────────────────────────────
        self.model_combo = QComboBox()
        model_status: dict[str, bool] = {}
        if self._model_status_provider:
            try:
                model_status = self._model_status_provider()
            except Exception:
                model_status = {}
        selected_model = str(whisper_cfg.get("model", "small")).strip().lower()
        for model_name in ["small", "medium", "large-v3", "large-v3-turbo"]:
            installed = bool(model_status.get(model_name, False))
            label = model_name if installed else f"{model_name} {self._t('tray_model_install_suffix')}"
            self.model_combo.addItem(label, model_name)
        combo_index = self.model_combo.findData(selected_model)
        if combo_index < 0:
            combo_index = 0
        self.model_combo.setCurrentIndex(combo_index)

        self.transcription_language_combo = QComboBox()
        self.transcription_language_combo.addItems(["auto", "de", "en", "fr", "es", "it"])
        self.transcription_language_combo.setCurrentText(str(whisper_cfg.get("language", "auto")))

        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["auto", "cuda", "cpu"])
        self.backend_combo.setCurrentText(str(whisper_cfg.get("backend", "auto")))
        self.backend_combo.setToolTip(tooltip(self._t("settings_backend_tooltip")))

        # ── Eingabe ───────────────────────────────────────────────────────────
        self.hotkey_input = QLineEdit()
        self.hotkey_input.setText(str(hotkey_cfg.get("combination", "win+ctrl")))
        self.hotkey_input.setPlaceholderText(self._t("settings_hotkey_placeholder"))
        self.hotkey_default_button = QPushButton(self._t("settings_hotkey_reset_default"))
        self.hotkey_default_button.clicked.connect(self._reset_hotkey_default)

        hotkey_row = QFrame()
        hotkey_layout = QHBoxLayout(hotkey_row)
        hotkey_layout.setContentsMargins(0, 0, 0, 0)
        hotkey_layout.setSpacing(8)
        hotkey_layout.addWidget(self.hotkey_input, 1)
        hotkey_layout.addWidget(self.hotkey_default_button)

        self.hotkey_mode_combo = QComboBox()
        self.hotkey_mode_combo.addItem(self._t("settings_hotkey_mode_hold"), "hold")
        self.hotkey_mode_combo.addItem(self._t("settings_hotkey_mode_press"), "press")
        self.hotkey_mode_combo.setToolTip(tooltip(self._t("settings_hotkey_mode_tooltip")))
        current_hotkey_mode = str(hotkey_cfg.get("mode", "press")).strip().lower()
        if current_hotkey_mode not in {"hold", "press"}:
            current_hotkey_mode = "press"
        for idx in range(self.hotkey_mode_combo.count()):
            if self.hotkey_mode_combo.itemData(idx) == current_hotkey_mode:
                self.hotkey_mode_combo.setCurrentIndex(idx)
                break

        # ── Allgemein ─────────────────────────────────────────────────────────
        rescue_enabled = bool(insertion_cfg.get("rescue_enabled", True))
        rescue_timeout = int(insertion_cfg.get("rescue_timeout_sec", 120) or 120)
        rescue_never = bool(insertion_cfg.get("rescue_never_expire", False))

        self.rescue_enabled_checkbox = QCheckBox(self._t("settings_label_rescue_enabled"))
        self.rescue_enabled_checkbox.setChecked(rescue_enabled)
        self.rescue_enabled_checkbox.toggled.connect(self._on_rescue_enabled_toggled)
        self.rescue_enabled_checkbox.setToolTip(tooltip(self._t("settings_rescue_enabled_tooltip")))

        self.rescue_timeout_spin = QSpinBox()
        self.rescue_timeout_spin.setRange(1, 999999)
        self.rescue_timeout_spin.setToolTip(tooltip(self._t("settings_rescue_timeout_tooltip")))
        self.rescue_timeout_spin.setSingleStep(10)
        self.rescue_timeout_spin.setValue(max(1, rescue_timeout))
        self.rescue_timeout_spin.setSuffix(f" {self._t('settings_unload_seconds_suffix')}")

        self.rescue_never_checkbox = QCheckBox(self._t("settings_label_rescue_never"))
        self.rescue_never_checkbox.setChecked(rescue_never)
        self.rescue_never_checkbox.toggled.connect(self._on_rescue_never_toggled)
        self.rescue_never_checkbox.setToolTip(tooltip(self._t("settings_rescue_never_tooltip")))

        rescue_enabled_row = QWidget()
        rescue_enabled_layout = QHBoxLayout(rescue_enabled_row)
        rescue_enabled_layout.setContentsMargins(9, 0, 0, 0)
        rescue_enabled_layout.setSpacing(0)
        rescue_enabled_layout.addWidget(self.rescue_enabled_checkbox)
        rescue_enabled_layout.addStretch()
        rescue_enabled_row.setMinimumHeight(30)

        rescue_timeout_row = QFrame()
        rescue_timeout_layout = QHBoxLayout(rescue_timeout_row)
        rescue_timeout_layout.setContentsMargins(0, 0, 0, 0)
        rescue_timeout_layout.setSpacing(8)
        rescue_timeout_layout.addWidget(self.rescue_timeout_spin)
        rescue_timeout_layout.addWidget(self.rescue_never_checkbox)
        rescue_timeout_layout.addStretch()
        rescue_timeout_row.setMinimumHeight(30)
        rescue_timeout_row.setToolTip(tooltip(self._t("settings_rescue_timeout_tooltip")))

        self.ui_language_combo = QComboBox()
        self.ui_language_combo.addItem(self._t("language_name_de"), "de")
        self.ui_language_combo.addItem(self._t("language_name_en"), "en")
        self.ui_language_combo.setToolTip(tooltip(self._t("settings_ui_language_tooltip")))
        selected_ui = normalize_ui_language(str(general_cfg.get("language_ui", self._ui_language)))
        for idx in range(self.ui_language_combo.count()):
            if self.ui_language_combo.itemData(idx) == selected_ui:
                self.ui_language_combo.setCurrentIndex(idx)
                break

        self.autostart_checkbox = QCheckBox(self._t("settings_label_autostart"))
        self.autostart_checkbox.setChecked(bool(general_cfg.get("autostart", True)))

        self.debug_logging_checkbox = QCheckBox(self._t("settings_debug_logging"))
        self.debug_logging_checkbox.setChecked(bool(general_cfg.get("debug_logging", False)))
        self.debug_logging_checkbox.setToolTip(tooltip(self._t("settings_debug_logging_tooltip")))

        unload_after_idle_sec = int(whisper_cfg.get("unload_after_idle_sec", 300))
        self.unload_never_checkbox = QCheckBox(self._t("settings_label_unload_never"))
        self.unload_never_checkbox.setChecked(unload_after_idle_sec == 0)
        self.unload_never_checkbox.toggled.connect(self._on_unload_never_toggled)
        self.unload_never_checkbox.setToolTip(tooltip(self._t("settings_unload_never_tooltip")))

        self.unload_delay_spin = QSpinBox()
        self.unload_delay_spin.setRange(10, 86400)
        self.unload_delay_spin.setToolTip(tooltip(self._t("settings_unload_tooltip")))
        self.unload_delay_spin.setSingleStep(10)
        self.unload_delay_spin.setValue(300 if unload_after_idle_sec == 0 else unload_after_idle_sec)
        self.unload_delay_spin.setSuffix(f" {self._t('settings_unload_seconds_suffix')}")
        self.unload_delay_spin.setEnabled(unload_after_idle_sec != 0)

        # Spinner + "Nie entladen" on one horizontal line
        unload_row = QFrame()
        unload_layout = QHBoxLayout(unload_row)
        unload_layout.setContentsMargins(0, 0, 0, 0)
        unload_layout.setSpacing(8)
        unload_layout.addWidget(self.unload_delay_spin)
        unload_layout.addWidget(self.unload_never_checkbox)
        unload_layout.addStretch()
        unload_row.setToolTip(tooltip(self._t("settings_unload_tooltip")))

        # Autostart checkbox – shifted 9 px right to align with dropdown text
        autostart_container = QWidget()
        autostart_layout = QHBoxLayout(autostart_container)
        autostart_layout.setContentsMargins(9, 0, 0, 0)
        autostart_layout.setSpacing(0)
        autostart_layout.addWidget(self.autostart_checkbox)
        autostart_layout.addStretch()
        autostart_container.setMinimumHeight(30)
        unload_row.setMinimumHeight(30)

        # ── Overlay ───────────────────────────────────────────────────────────
        self.waveform_style_combo = QComboBox()
        self.waveform_style_combo.addItem(self._t("settings_waveform_style_gradient"), "gradient")
        self.waveform_style_combo.addItem(self._t("settings_waveform_style_single"), "single")
        style = str(overlay_cfg.get("waveform_style", "gradient")).strip().lower()
        if style not in {"gradient", "single"}:
            style = "gradient"
        for idx in range(self.waveform_style_combo.count()):
            if self.waveform_style_combo.itemData(idx) == style:
                self.waveform_style_combo.setCurrentIndex(idx)
                break
        self.waveform_style_combo.currentIndexChanged.connect(self._on_waveform_style_changed)
        self.waveform_style_combo.setToolTip(tooltip(self._t("settings_waveform_style_tooltip")))

        self.waveform_reset_button = QPushButton(self._t("settings_reset_default_colors"))
        self.waveform_reset_button.clicked.connect(self._reset_waveform_defaults)

        self.overlay_monitor_combo = QComboBox()
        self._populate_overlay_monitor_combo(int(overlay_cfg.get("monitor_index", -1)))
        self.overlay_monitor_combo.setToolTip(tooltip(self._t("settings_overlay_monitor_tooltip")))

        style_row = QFrame()
        style_layout = QHBoxLayout(style_row)
        style_layout.setContentsMargins(0, 0, 0, 0)
        style_layout.setSpacing(8)
        style_layout.addWidget(self.waveform_style_combo, 1)
        style_layout.addWidget(self.waveform_reset_button)

        self.waveform_primary_input = QLineEdit()
        self.waveform_primary_input.setReadOnly(True)
        self.waveform_primary_input.setMaxLength(7)
        self.waveform_primary_input.setFixedWidth(100)
        self.waveform_primary_preview = QFrame()
        self.waveform_primary_preview.setFixedSize(24, 24)
        self.waveform_primary_preview.setFrameShape(QFrame.StyledPanel)
        self.waveform_primary_button = QPushButton(self._t("settings_choose_color"))
        self.waveform_primary_button.clicked.connect(self._choose_waveform_primary_color)

        self.waveform_secondary_input = QLineEdit()
        self.waveform_secondary_input.setReadOnly(True)
        self.waveform_secondary_input.setMaxLength(7)
        self.waveform_secondary_input.setFixedWidth(100)
        self.waveform_secondary_preview = QFrame()
        self.waveform_secondary_preview.setFixedSize(24, 24)
        self.waveform_secondary_preview.setFrameShape(QFrame.StyledPanel)
        self.waveform_secondary_button = QPushButton(self._t("settings_choose_color"))
        self.waveform_secondary_button.clicked.connect(self._choose_waveform_secondary_color)

        primary_color = str(
            overlay_cfg.get(
                "waveform_gradient_start",
                overlay_cfg.get("waveform_color", overlay_cfg.get("color_recording", "#56F64E")),
            )
        )
        secondary_color = str(overlay_cfg.get("waveform_gradient_end", "#0096FF"))
        self._set_waveform_primary_color(primary_color)
        self._set_waveform_secondary_color(secondary_color)

        # Color rows: swatch + hex input + button (identical layout for both)
        def _make_color_row(preview: QFrame, inp: QLineEdit, btn: QPushButton) -> QFrame:
            row = QFrame()
            lay = QHBoxLayout(row)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(8)
            lay.addWidget(preview)
            lay.addWidget(inp)
            lay.addWidget(btn, 1)
            return row

        self.waveform_primary_widget   = _make_color_row(
            self.waveform_primary_preview,   self.waveform_primary_input,   self.waveform_primary_button
        )
        self.waveform_secondary_widget = _make_color_row(
            self.waveform_secondary_preview, self.waveform_secondary_input, self.waveform_secondary_button
        )

        # ── Layout ────────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Erkennung card — includes CUDA status (same card as in mockup)
        recognition_group = QGroupBox(self._t("settings_group_recognition").upper())
        recognition_form  = QFormLayout(recognition_group)
        recognition_form.setSpacing(4)
        add_row(recognition_form, self._t("settings_label_model"), self.model_combo)
        model_hint = QLabel(self._t("settings_model_hint"))
        model_hint.setStyleSheet("font-size: 8.5pt; color: #3f4b5f;")
        recognition_form.addRow(mk_form_label(""), model_hint)
        add_row(
            recognition_form,
            self._t("settings_label_transcription_language"),
            self.transcription_language_combo,
        )
        add_row(recognition_form, self._t("settings_label_backend"), self.backend_combo)
        input_group = QGroupBox(self._t("settings_group_input").upper())
        input_form  = QFormLayout(input_group)
        input_form.setSpacing(4)
        add_row(
            input_form,
            self._t("settings_label_hotkey"),
            hotkey_row,
            self._t("settings_hotkey_info_tooltip"),
        )
        add_row(input_form, self._t("settings_label_hotkey_mode"), self.hotkey_mode_combo)
        add_row(
            input_form,
            self._t("settings_label_rescue_memory"),
            rescue_enabled_row,
            self._t("settings_rescue_memory_info_tooltip"),
        )
        add_row(input_form, self._t("settings_label_rescue_timeout"), rescue_timeout_row)

        general_group = QGroupBox(self._t("settings_group_general").upper())
        general_form  = QFormLayout(general_group)
        general_form.setVerticalSpacing(9)
        general_form.setHorizontalSpacing(4)
        add_row(general_form, self._t("settings_label_ui_language"), self.ui_language_combo)
        add_row(general_form, self._t("settings_label_autostart_row"), autostart_container)
        add_row(general_form, self._t("settings_label_unload_behavior"), unload_row)
        self.open_config_button = QPushButton(self._t("settings_open_button"))
        self.open_config_button.clicked.connect(self._handle_open_config)
        self.open_config_button.setToolTip(tooltip(self._t("settings_open_config_tooltip")))
        self.open_config_button.setEnabled(self._on_open_config is not None)
        add_row(general_form, self._t("settings_open_config"), self.open_config_button)

        overlay_group = QGroupBox(self._t("settings_group_overlay").upper())
        overlay_form  = QFormLayout(overlay_group)
        overlay_form.setSpacing(4)
        add_row(overlay_form, self._t("settings_label_overlay_monitor"), self.overlay_monitor_combo)
        add_row(overlay_form, self._t("settings_label_waveform_style"), style_row)
        add_row(overlay_form, self._t("settings_label_waveform_primary"), self.waveform_primary_widget)
        self.waveform_secondary_label = QLabel(self._t("settings_label_waveform_secondary"))
        self.waveform_secondary_label.setFixedWidth(_FORM_LABEL_WIDTH)
        overlay_form.addRow(self.waveform_secondary_label, self.waveform_secondary_widget)

        root.addWidget(recognition_group)
        root.addWidget(input_group)
        root.addWidget(general_group)
        root.addWidget(overlay_group)

        # Footer buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        save_btn   = buttons.button(QDialogButtonBox.Save)
        cancel_btn = buttons.button(QDialogButtonBox.Cancel)
        if save_btn:
            save_btn.setText(self._t("common_save"))
            save_btn.setStyleSheet(_QSS_SAVE)
        if cancel_btn:
            cancel_btn.setText(self._t("common_cancel"))
            cancel_btn.setStyleSheet(_QSS_CANCEL)

        footer_row = QHBoxLayout()
        footer_row.setContentsMargins(0, 0, 0, 0)
        footer_row.setSpacing(8)
        self.logs_button = QPushButton(self._t("settings_open_logs"))
        self.logs_button.setToolTip(tooltip(self._t("settings_open_logs_tooltip")))
        self.logs_button.clicked.connect(self._handle_open_logs)
        self.logs_button.setEnabled(self._on_open_logs is not None)
        footer_row.addWidget(self.logs_button)
        footer_row.addWidget(self.debug_logging_checkbox)
        footer_row.addStretch()
        footer_row.addWidget(buttons)
        root.addLayout(footer_row)

        self._on_waveform_style_changed()
        self._update_rescue_controls()
        log.info("SettingsDialog initialized: ui_language=%s model=%s backend=%s hotkey_mode=%s", self._ui_language, self.model_combo.currentData(), self.backend_combo.currentText(), self.hotkey_mode_combo.currentData())

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_unload_never_toggled(self, checked: bool) -> None:
        self.unload_delay_spin.setEnabled(not checked)

    def _on_rescue_enabled_toggled(self, checked: bool) -> None:
        self._update_rescue_controls()

    def _on_rescue_never_toggled(self, checked: bool) -> None:
        self._update_rescue_controls()

    def _update_rescue_controls(self) -> None:
        enabled = self.rescue_enabled_checkbox.isChecked()
        self.rescue_timeout_spin.setEnabled(enabled and not self.rescue_never_checkbox.isChecked())
        self.rescue_never_checkbox.setEnabled(enabled)

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

    def _reset_hotkey_default(self) -> None:
        self.hotkey_input.setText("win+ctrl")

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

    def _on_waveform_style_changed(self) -> None:
        style    = str(self.waveform_style_combo.currentData())
        gradient = style == "gradient"
        self.waveform_secondary_widget.setEnabled(gradient)
        self.waveform_secondary_input.setEnabled(gradient)
        self.waveform_secondary_button.setEnabled(gradient)
        self.waveform_secondary_preview.setEnabled(gradient)
        self.waveform_secondary_label.setEnabled(gradient)

    def _populate_overlay_monitor_combo(self, selected_index: int) -> None:
        self.overlay_monitor_combo.clear()
        self.overlay_monitor_combo.addItem(self._t("settings_overlay_monitor_primary"), -1)

        for idx, screen in enumerate(QGuiApplication.screens()):
            geom = screen.geometry()
            name = screen.name() or f"Display {idx + 1}"
            label = f"{idx + 1}: {name} ({geom.width()}x{geom.height()})"
            self.overlay_monitor_combo.addItem(label, idx)

        target = selected_index if selected_index >= 0 else -1
        combo_index = self.overlay_monitor_combo.findData(target)
        if combo_index < 0:
            combo_index = 0
        self.overlay_monitor_combo.setCurrentIndex(combo_index)

    def _reset_waveform_defaults(self) -> None:
        self.waveform_style_combo.setCurrentText(self._t("settings_waveform_style_gradient"))
        self._set_waveform_primary_color("#56F64E")
        self._set_waveform_secondary_color("#0096FF")
        self._on_waveform_style_changed()

    def _set_waveform_primary_color(self, color_hex: str) -> None:
        color = QColor(color_hex)
        if not color.isValid():
            color = QColor("#56F64E")
        normalized = color.name().upper()
        self.waveform_primary_input.setText(normalized)
        if not hasattr(self, "waveform_primary_preview"):
            return
        self.waveform_primary_preview.setStyleSheet(
            f"background:{normalized}; border:1px solid #7b8598; border-radius:4px;"
        )

    def _set_waveform_secondary_color(self, color_hex: str) -> None:
        color = QColor(color_hex)
        if not color.isValid():
            color = QColor("#0096FF")
        normalized = color.name().upper()
        self.waveform_secondary_input.setText(normalized)
        if not hasattr(self, "waveform_secondary_preview"):
            return
        self.waveform_secondary_preview.setStyleSheet(
            f"background:{normalized}; border:1px solid #7b8598; border-radius:4px;"
        )

    def _choose_waveform_primary_color(self) -> None:
        initial = QColor(self.waveform_primary_input.text())
        picked  = QColorDialog.getColor(initial, self, self._t("settings_color_dialog_title"))
        if not picked.isValid():
            return
        self._set_waveform_primary_color(picked.name())

    def _choose_waveform_secondary_color(self) -> None:
        initial = QColor(self.waveform_secondary_input.text())
        picked  = QColorDialog.getColor(initial, self, self._t("settings_color_dialog_title"))
        if not picked.isValid():
            return
        self._set_waveform_secondary_color(picked.name())

    def showEvent(self, event) -> None:  # noqa: N802, ANN001
        super().showEvent(event)
        log.info("SettingsDialog showEvent: visible=%s size=%sx%s", self.isVisible(), self.width(), self.height())

    def closeEvent(self, event) -> None:  # noqa: N802, ANN001
        log.info("SettingsDialog closeEvent")
        super().closeEvent(event)

    def _handle_open_config(self) -> None:
        if not self._on_open_config:
            return
        log.info("SettingsDialog: open config requested")
        try:
            self._on_open_config()
        except Exception as exc:
            QMessageBox.warning(self, self._t("settings_title"), str(exc))

    def _handle_open_logs(self) -> None:
        if not self._on_open_logs:
            return
        log.info("SettingsDialog: open logs requested")
        try:
            self._on_open_logs()
        except Exception as exc:
            QMessageBox.warning(self, self._t("settings_title"), str(exc))

    def _refresh_cuda_status(self) -> None:
        if not self._cuda_status_provider:
            self.cuda_status_value.setText(self._t("settings_cuda_state_unknown"))
            self.cuda_download_button.hide()
            return

        try:
            state = self._cuda_status_provider()
        except Exception:
            self.cuda_status_value.setText(self._t("settings_cuda_state_unknown"))
            self.cuda_download_button.hide()
            return

        text         = str(state.get("text", self._t("settings_cuda_state_unknown")))
        downloadable = bool(state.get("downloadable", False))
        self.cuda_status_value.setText(text)
        if downloadable and self._on_cuda_download:
            self.cuda_download_button.show()
            self.cuda_download_button.setEnabled(True)
        else:
            self.cuda_download_button.hide()

    def _handle_cuda_download(self) -> None:
        if not self._on_cuda_download:
            return
        self.cuda_download_button.setEnabled(False)
        try:
            success = bool(self._on_cuda_download())
        except Exception as exc:
            QMessageBox.warning(self, self._t("settings_title"), str(exc))
            success = False
        finally:
            self.cuda_download_button.setEnabled(True)
        if success:
            self.backend_combo.setCurrentText("cuda")
        self._refresh_cuda_status()

    def _save(self) -> None:
        ok, normalized_or_error = validate_hotkey_combination(self.hotkey_input.text())
        if not ok:
            QMessageBox.warning(self, self._t("settings_invalid_hotkey_title"), normalized_or_error)
            return

        primary_hex   = self.waveform_primary_input.text().strip()
        secondary_hex = self.waveform_secondary_input.text().strip()
        if not QColor(primary_hex).isValid():
            QMessageBox.warning(
                self,
                self._t("settings_invalid_color_title"),
                self._t("settings_invalid_color_body"),
            )
            return
        if not QColor(secondary_hex).isValid():
            QMessageBox.warning(
                self,
                self._t("settings_invalid_color_title"),
                self._t("settings_invalid_color_body"),
            )
            return

        waveform_style = str(self.waveform_style_combo.currentData())
        payload = {
            "model":                  str(self.model_combo.currentData() or self.model_combo.currentText()),
            "language":               self.transcription_language_combo.currentText(),
            "backend":                self.backend_combo.currentText(),
            "hotkey":                 normalized_or_error,
            "hotkey_mode":            str(self.hotkey_mode_combo.currentData()),
            "rescue_enabled":         self.rescue_enabled_checkbox.isChecked(),
            "rescue_timeout_sec":     int(self.rescue_timeout_spin.value()),
            "rescue_never_expire":    self.rescue_never_checkbox.isChecked(),
            "language_ui":            str(self.ui_language_combo.currentData()),
            "autostart":              self.autostart_checkbox.isChecked(),
            "debug_logging":          self.debug_logging_checkbox.isChecked(),
            "waveform_style":         waveform_style,
            "waveform_color":         QColor(primary_hex).name().upper(),
            "waveform_gradient_start": QColor(primary_hex).name().upper(),
            "waveform_gradient_end":   QColor(secondary_hex).name().upper(),
            "overlay_monitor_index":  int(self.overlay_monitor_combo.currentData()),
            "unload_after_idle_sec":  0 if self.unload_never_checkbox.isChecked()
                                      else int(self.unload_delay_spin.value()),
        }

        log.info("SettingsDialog save requested: model=%s backend=%s hotkey_mode=%s ui_language=%s autostart=%s unload_after_idle_sec=%s rescue_enabled=%s rescue_never=%s overlay_monitor_index=%s", payload["model"], payload["backend"], payload["hotkey_mode"], payload["language_ui"], payload["autostart"], payload["unload_after_idle_sec"], payload["rescue_enabled"], payload["rescue_never_expire"], payload["overlay_monitor_index"])
        try:
            self._on_save(payload)
        except Exception as exc:
            QMessageBox.critical(self, self._t("settings_save_failed_title"), str(exc))
            return

        self.accept()

    def _t(self, key: str, **kwargs: object) -> str:
        return tr(self._ui_language, key, **kwargs)
