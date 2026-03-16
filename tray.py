from __future__ import annotations

from collections.abc import Callable
import logging
from pathlib import Path
import sys

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QCursor, QFontMetrics, QIcon, QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMenu,
    QSizePolicy,
    QSystemTrayIcon,
    QWidget,
    QWidgetAction,
)

from i18n import normalize_ui_language, tr

AudioDeviceItem = dict[str, str | bool]
log = logging.getLogger(__name__)

_MENU_QSS = """
QMenu {
    background-color: #1a1e27;
    border: 1px solid #2d3340;
    border-radius: 10px;
    padding: 0px;
    color: #e9edf3;
    font-family: "Segoe UI";
}
QMenu::separator {
    height: 1px;
    background: #252b35;
    margin: 3px 0px;
}
QMenu::item {
    padding: 5px 14px;
    min-height: 31px;
    color: #c8d0dc;
    background: transparent;
}
QMenu::item:selected {
    background: #232830;
}

QWidget#trayRow {
    background: transparent;
}
QWidget#trayRow[clickable="true"]:hover {
    background: #232830;
}
QWidget#trayRow[role="quit"]:hover {
    background: rgba(239, 68, 68, .06);
}
QWidget#trayRow:disabled,
QWidget#trayRow:disabled:hover,
QWidget#trayRow[role="quit"]:disabled:hover {
    background: transparent;
}

QLabel#trayLabelStatus {
    color: #444e5c;
    font-size: 11.5px;
    font-variant-numeric: tabular-nums;
}
QLabel#trayLabelSection {
    color: #323d4d;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}
QLabel#trayLabelMain {
    color: #c8d0dc;
    font-size: 13px;
}
QLabel#trayValue {
    color: #4a5568;
    font-size: 12px;
}
QLabel#trayIndicator {
    color: #aeb8c6;
    font-size: 12px;
    font-weight: 700;
}
QWidget#trayRow:disabled QLabel#trayIndicator {
    color: #566171;
}
QLabel#trayPill {
    color: #aeb8c6;
    font-size: 11px;
    padding: 2px 8px;
    border: 1px solid #384151;
    border-radius: 9px;
    background: #252b35;
}
QWidget#trayRow[clickable="true"]:hover QLabel#trayPill {
    color: #d8dee8;
    background: #2b313c;
    border-color: #4d5a70;
}
QWidget#trayRow:disabled QLabel#trayPill {
    color: #566171;
    background: #1d222b;
    border-color: #2a303a;
}
QLabel#trayArrow {
    color: #333d4d;
    font-size: 10px;
}
QLabel#trayLabelAction {
    color: #8892a4;
    font-size: 12.5px;
}
QWidget#trayRow[clickable="true"]:hover QLabel#trayLabelAction {
    color: #c8d0dc;
}
QWidget#trayRow:disabled QLabel#trayLabelAction {
    color: #4a5568;
}
QLabel#trayLabelQuit {
    color: #6b3a3a;
    font-size: 12.5px;
}
QWidget#trayRow[role="quit"]:hover QLabel#trayLabelQuit {
    color: #e05555;
}
QLabel#trayLabelFooter {
    color: #3f4856;
    font-size: 10.5px;
}
"""

_SUBMENU_QSS = """
QMenu {
    background-color: #1a1e27;
    border: 1px solid #2d3340;
    border-radius: 8px;
    padding: 2px 0px;
    color: #c8d0dc;
    font-family: "Segoe UI";
    font-size: 9pt;
}
QMenu::separator {
    height: 1px;
    background: #252b35;
    margin: 2px 0px;
}
QMenu::item {
    padding: 1px 9px;
    min-height: 18px;
    color: #c8d0dc;
    background: transparent;
}
QMenu::item:selected {
    background: #232830;
}
QMenu::item:checked {
    background: #252b35;
}
"""


def _resolve_tray_icon(icon_path: str) -> QIcon:
    candidates: list[Path] = []
    if icon_path:
        raw = Path(icon_path)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(Path.cwd() / raw)

    meipass = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    exe_dir = Path(sys.executable).resolve().parent
    module_dir = Path(__file__).resolve().parent

    candidates.extend(
        [
            meipass / "assets" / "icon.ico",
            meipass / "assets" / "icon.png",
            exe_dir / "assets" / "icon.ico",
            exe_dir / "assets" / "icon.png",
            module_dir / "assets" / "icon.ico",
            module_dir / "assets" / "icon.png",
            exe_dir / "icon.ico",
            exe_dir / "icon.png",
            Path(sys.executable).resolve(),
        ]
    )

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            if not candidate.exists():
                continue
            icon = QIcon(str(candidate))
            if not icon.isNull():
                log.info("Tray icon loaded from: %s", candidate)
                return icon
        except Exception:
            continue

    log.warning("Tray icon could not be loaded from any known path.")
    return QIcon()


class _TrayRowWidget(QWidget):
    def __init__(
        self,
        role: str,
        label: str = "",
        value: str = "",
        show_arrow: bool = False,
        clickable: bool = False,
        on_click: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_click = on_click
        self._role = role
        self._raw_label = str(label)
        self._raw_value = str(value)
        self._value_mode = "plain"
        self.setObjectName("trayRow")
        self.setProperty("role", role)
        self.setProperty("clickable", "true" if clickable else "false")
        if clickable:
            self.setCursor(Qt.PointingHandCursor)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 0, 14, 0)
        layout.setSpacing(6)

        self.label = QLabel(label)
        self.label.setWordWrap(False)
        self.label.setMinimumWidth(0)
        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.label, 1)

        self.value = QLabel(value)
        self.value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value.setMinimumWidth(0)
        self.value.setFixedWidth(96)
        self.value.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        self.value.setObjectName("trayValue")
        self.value.setVisible(bool(value))
        layout.addWidget(self.value, 0)

        self.arrow = QLabel("›")
        self.arrow.setVisible(show_arrow)
        self.arrow.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.arrow.setFixedWidth(8)
        self.arrow.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        layout.addWidget(self.arrow, 0)

        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        layout.setStretch(2, 0)

        if role == "status":
            self.setMinimumHeight(31)
            self.label.setObjectName("trayLabelStatus")
            self.label.setWordWrap(True)
            self.value.hide()
            self.arrow.hide()
        elif role == "section":
            self.setFixedHeight(20)
            self.label.setObjectName("trayLabelSection")
            self.value.hide()
            self.arrow.hide()
        elif role == "action":
            self.setFixedHeight(31)
            self.label.setObjectName("trayLabelAction")
            self.arrow.hide()
        elif role == "quit":
            self.setFixedHeight(31)
            self.label.setObjectName("trayLabelQuit")
            self.value.hide()
            self.arrow.hide()
        elif role == "footer":
            self.setFixedHeight(18)
            self.label.setObjectName("trayLabelFooter")
            self.value.hide()
            self.arrow.hide()
        else:
            self.setFixedHeight(31)
            self.label.setObjectName("trayLabelMain")
            self.value.setObjectName("trayValue")
            self.arrow.setObjectName("trayArrow")
        self._apply_elided_texts()

    def set_label(self, text: str) -> None:
        self._raw_label = str(text or "")
        self._apply_elided_texts()

    def set_value(self, text: str) -> None:
        clean = str(text or "").strip()
        self._raw_value = clean
        self._update_value_width()
        self._apply_elided_texts()
        self.value.setVisible(bool(clean))

    def set_value_pill(self, text: str) -> None:
        self._value_mode = "pill"
        self.value.setObjectName("trayPill")
        self.value.setAlignment(Qt.AlignCenter)
        self._repolish_value()
        self.set_value(text)

    def set_value_indicator(self, text: str) -> None:
        self._value_mode = "indicator"
        self.value.setObjectName("trayIndicator")
        self.value.setAlignment(Qt.AlignCenter)
        self._repolish_value()
        self._raw_value = str(text or "")
        self._update_value_width()
        self._apply_elided_texts()
        self.value.setVisible(bool(self._raw_value))

    def clear_value(self) -> None:
        self._raw_value = ""
        self.value.hide()
        self._apply_elided_texts()

    def _repolish_value(self) -> None:
        style = self.value.style()
        style.unpolish(self.value)
        style.polish(self.value)
        self.value.update()

    def _update_value_width(self) -> None:
        if not self._raw_value:
            return
        if self._value_mode == "pill":
            fm = QFontMetrics(self.value.font())
            width = max(64, fm.horizontalAdvance(self._raw_value) + 18)
            self.value.setFixedWidth(width)
        elif self._value_mode == "indicator":
            self.value.setFixedWidth(18)
        else:
            self.value.setFixedWidth(96)

    def _apply_elided_texts(self) -> None:
        if self._role == "status":
            self.label.setText(self._raw_label)
            self._update_status_height()
            return

        label_metrics = QFontMetrics(self.label.font())
        label_width = max(10, self.label.width() - 2)
        self.label.setText(label_metrics.elidedText(self._raw_label, Qt.ElideRight, label_width))

        if self.value.isVisible() or self._raw_value:
            value_metrics = QFontMetrics(self.value.font())
            value_width = max(10, self.value.width() - 2)
            self.value.setText(value_metrics.elidedText(self._raw_value, Qt.ElideRight, value_width))

    def resizeEvent(self, event) -> None:  # noqa: N802, ANN001
        super().resizeEvent(event)
        self._apply_elided_texts()

    def _update_status_height(self) -> None:
        content_width = max(40, self.width() - 28)
        fm = QFontMetrics(self.label.font())
        wrapped = fm.boundingRect(
            0,
            0,
            content_width,
            200,
            Qt.TextWordWrap,
            self._raw_label,
        )
        desired = max(31, wrapped.height() + 12)
        self.setMinimumHeight(desired)
        self.setMaximumHeight(desired)

    def set_enabled(self, enabled: bool) -> None:
        self.setEnabled(enabled)
        self.label.setEnabled(enabled)
        self.value.setEnabled(enabled)
        self.arrow.setEnabled(enabled)
        clickable = enabled and self._on_click is not None
        self.setProperty("clickable", "true" if clickable else "false")
        if clickable:
            self.setCursor(Qt.PointingHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        for widget in (self, self.label, self.value, self.arrow):
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
            widget.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._role == "footer" and event.button() == Qt.LeftButton:
            event.accept()
            return
        if (
            self.isEnabled()
            and event.button() == Qt.LeftButton
            and self._on_click is not None
            and self.rect().contains(event.pos())
        ):
            self._on_click()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._role == "footer" and event.button() == Qt.LeftButton:
            event.accept()
            return
        super().mousePressEvent(event)


class AppTray(QSystemTrayIcon):
    def __init__(
        self,
        icon_path: str,
        short_status_provider: Callable[[], str],
        full_status_provider: Callable[[], str],
        ui_language_provider: Callable[[], str],
        model_status_provider: Callable[[], dict[str, bool]],
        on_model: Callable[[str], None],
        on_model_install: Callable[[str], None],
        on_transcription_language: Callable[[str], None],
        on_backend: Callable[[str], None],
        audio_devices_provider: Callable[[], tuple[list[AudioDeviceItem], str]],
        on_audio_device: Callable[[str], None],
        on_open_settings: Callable[[], None],
        unload_available_provider: Callable[[], bool],
        unload_never_provider: Callable[[], bool],
        on_unload_model: Callable[[], None],
        on_toggle_unload_never: Callable[[], None],
        rescue_copy_available_provider: Callable[[], bool],
        on_copy_last_dictation: Callable[[], None],
        on_quit: Callable[[], None],
        on_debug_trace: Callable[[str], None] | None = None,
        on_debug_critical: Callable[[str], None] | None = None,
        on_debug_clear: Callable[[str | None], None] | None = None,
    ) -> None:
        super().__init__(_resolve_tray_icon(icon_path))
        self._short_status_provider = short_status_provider
        self._full_status_provider = full_status_provider
        self._ui_language_provider = ui_language_provider
        self._model_status_provider = model_status_provider
        self._on_model = on_model
        self._on_model_install = on_model_install
        self._on_transcription_language = on_transcription_language
        self._on_backend = on_backend
        self._audio_devices_provider = audio_devices_provider
        self._on_audio_device = on_audio_device
        self._on_open_settings = on_open_settings
        self._unload_available_provider = unload_available_provider
        self._unload_never_provider = unload_never_provider
        self._on_unload_model = on_unload_model
        self._on_toggle_unload_never = on_toggle_unload_never
        self._rescue_copy_available_provider = rescue_copy_available_provider
        self._on_copy_last_dictation = on_copy_last_dictation
        self._on_quit = on_quit
        self._on_debug_trace = on_debug_trace
        self._on_debug_critical = on_debug_critical
        self._on_debug_clear = on_debug_clear
        self._ui_language = normalize_ui_language(self._ui_language_provider())

        self._current_model = ""
        self._current_transcription_language = ""
        self._current_backend = ""
        self._current_microphone = ""

        self._model_actions: dict[str, QAction] = {}
        self._transcription_language_actions: dict[str, QAction] = {}
        self._backend_actions: dict[str, QAction] = {}
        self._audio_actions: dict[str, QAction] = {}
        self._audio_group = QActionGroup(self)
        self._audio_group.setExclusive(True)

        self.menu = QMenu()
        self.menu.setStyleSheet(_MENU_QSS)
        self.menu.setMinimumWidth(250)
        self.menu.setMaximumWidth(250)
        self.menu.aboutToShow.connect(self._on_menu_about_to_show)
        self.menu.aboutToHide.connect(self._on_menu_about_to_hide)

        # Hidden submenus (opened on row click)
        self._model_menu = QMenu(self.menu)
        self._model_menu.setStyleSheet(_SUBMENU_QSS)
        self._transcription_language_menu = QMenu(self.menu)
        self._transcription_language_menu.setStyleSheet(_SUBMENU_QSS)
        self._backend_menu = QMenu(self.menu)
        self._backend_menu.setStyleSheet(_SUBMENU_QSS)
        self._audio_menu = QMenu(self.menu)
        self._audio_menu.setStyleSheet(_SUBMENU_QSS)

        self._build_submenus()
        self._build_main_menu()
        self._rebuild_audio_menu()

        self.setContextMenu(self.menu)
        self.activated.connect(self._handle_activated)
        self._retranslate()

    def _debug_trace(self, event: str) -> None:
        try:
            if self._on_debug_trace is not None:
                self._on_debug_trace(event)
        except Exception:
            log.exception("Tray debug trace callback failed: %s", event)

    def _debug_critical(self, step: str) -> None:
        try:
            if self._on_debug_critical is not None:
                self._on_debug_critical(step)
        except Exception:
            log.exception("Tray debug critical callback failed: %s", step)

    def _debug_clear(self, note: str | None = None) -> None:
        try:
            if self._on_debug_clear is not None:
                self._on_debug_clear(note)
        except Exception:
            log.exception("Tray debug clear callback failed.")

    def _on_menu_about_to_show(self) -> None:
        log.debug("Tray menu aboutToShow.")
        self._debug_critical("tray_menu_about_to_show")
        self._debug_trace("tray_menu_about_to_show")

    def _on_menu_about_to_hide(self) -> None:
        log.debug("Tray menu aboutToHide.")
        self._debug_trace("tray_menu_about_to_hide")
        self._debug_clear("tray_menu_hidden")

    def _build_submenus(self) -> None:
        model_group = QActionGroup(self)
        model_group.setExclusive(True)
        for model in ["small", "medium", "large-v3", "large-v3-turbo"]:
            action = QAction(model, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, m=model: self._queue_callback(
                    f"model:{m}",
                    lambda m=m: self._on_model_action(m),
                    close_menu=False,
                )
            )
            model_group.addAction(action)
            self._model_menu.addAction(action)
            self._model_actions[model] = action

        tr_lang_group = QActionGroup(self)
        tr_lang_group.setExclusive(True)
        for lang in ["auto", "de", "en", "fr", "es", "it"]:
            action = QAction(lang, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, l=lang: self._queue_callback(
                    f"transcription_language:{l}",
                    lambda l=l: self._on_transcription_language(l),
                    close_menu=False,
                )
            )
            tr_lang_group.addAction(action)
            self._transcription_language_menu.addAction(action)
            self._transcription_language_actions[lang] = action

        backend_group = QActionGroup(self)
        backend_group.setExclusive(True)
        for backend in ["auto", "cuda", "cpu"]:
            action = QAction(backend, self)
            action.setCheckable(True)
            action.triggered.connect(
                lambda checked=False, b=backend: self._queue_callback(
                    f"backend:{b}",
                    lambda b=b: self._on_backend(b),
                    close_menu=False,
                )
            )
            backend_group.addAction(action)
            self._backend_menu.addAction(action)
            self._backend_actions[backend] = action

    def _add_row(self, widget: QWidget) -> QWidgetAction:
        action = QWidgetAction(self.menu)
        action.setDefaultWidget(widget)
        self.menu.addAction(action)
        return action

    def _build_main_menu(self) -> None:
        self._section_row = _TrayRowWidget(
            role="section",
            label=self._t("tray_menu_configuration_header"),
        )
        self._add_row(self._section_row)

        self._model_row = _TrayRowWidget(
            role="config",
            label=self._t("tray_menu_model"),
            show_arrow=True,
            clickable=True,
            on_click=lambda: self._open_submenu(self._model_menu, self._model_row_action),
        )
        self._model_row_action = self._add_row(self._model_row)

        self._transcription_row = _TrayRowWidget(
            role="config",
            label=self._t("tray_menu_transcription_language"),
            show_arrow=True,
            clickable=True,
            on_click=lambda: self._open_submenu(
                self._transcription_language_menu,
                self._transcription_row_action,
            ),
        )
        self._transcription_row_action = self._add_row(self._transcription_row)

        self._backend_row = _TrayRowWidget(
            role="config",
            label=self._t("tray_menu_backend"),
            show_arrow=True,
            clickable=True,
            on_click=lambda: self._open_submenu(self._backend_menu, self._backend_row_action),
        )
        self._backend_row_action = self._add_row(self._backend_row)

        self._audio_row = _TrayRowWidget(
            role="config",
            label=self._t("tray_menu_microphone"),
            show_arrow=True,
            clickable=True,
            on_click=lambda: self._open_submenu(self._audio_menu, self._audio_row_action),
        )
        self._audio_row_action = self._add_row(self._audio_row)

        self.menu.addSeparator()

        self._settings_row = _TrayRowWidget(
            role="action",
            label=self._t("tray_menu_settings"),
            clickable=True,
            on_click=lambda: self._invoke_callback("open_settings", self._on_open_settings, close_menu=True),
        )
        self._add_row(self._settings_row)

        self._unload_row = _TrayRowWidget(
            role="action",
            label=self._t("tray_menu_unload_model"),
            clickable=True,
            on_click=lambda: self._invoke_callback("unload_model", self._on_unload_model, close_menu=False),
        )
        self._unload_row_action = self._add_row(self._unload_row)

        self._unload_never_row = _TrayRowWidget(
            role="action",
            label=self._t("tray_menu_unload_never"),
            clickable=True,
            on_click=lambda: self._invoke_callback("toggle_unload_never", self._on_toggle_unload_never, close_menu=False),
        )
        self._unload_never_row_action = self._add_row(self._unload_never_row)

        self._rescue_copy_row = _TrayRowWidget(
            role="action",
            label=self._t("tray_menu_last_dictation"),
            clickable=True,
            on_click=lambda: self._invoke_callback("copy_last_dictation", self._on_copy_last_dictation, close_menu=True),
        )
        self._rescue_copy_row.set_value_pill(self._t("tray_action_copy"))
        self._rescue_copy_row_action = self._add_row(self._rescue_copy_row)

        self.menu.addSeparator()

        self._quit_row = _TrayRowWidget(
            role="quit",
            label=self._t("tray_menu_quit"),
            clickable=True,
            on_click=lambda: self._invoke_callback("quit", self._on_quit, close_menu=True),
        )
        self._add_row(self._quit_row)

        self._footer_row = _TrayRowWidget(
            role="footer",
            label="",
            clickable=False,
        )
        self._footer_row_action = self._add_row(self._footer_row)
        self._footer_row_action.setEnabled(False)

    def _open_submenu(self, submenu: QMenu, anchor_action: QAction) -> None:
        row_rect = self.menu.actionGeometry(anchor_action)
        if not row_rect.isValid():
            pos = self.menu.mapToGlobal(QPoint(self.menu.width() - 1, 0))
            submenu.popup(pos)
            return
        pos = self.menu.mapToGlobal(QPoint(self.menu.width() - 1, row_rect.top()))
        submenu.popup(pos)

    def refresh_status(self) -> None:
        self._ui_language = normalize_ui_language(self._ui_language_provider())
        self._refresh_model_actions()
        self._rebuild_audio_menu()
        self._update_unload_row()
        self._update_unload_never_row()
        self._update_rescue_copy_row()
        self._retranslate()
        full_status = self._full_status_provider()
        self.setToolTip(f"{self._t('app_name')}\n{full_status}")

    def set_selected(
        self,
        model: str,
        transcription_language: str,
        backend: str,
        ui_language: str,
    ) -> None:
        self._ui_language = normalize_ui_language(ui_language)
        self._current_model = model
        self._current_transcription_language = transcription_language
        self._current_backend = backend

        model_action = self._model_actions.get(model)
        if model_action:
            model_action.setChecked(True)
        tr_action = self._transcription_language_actions.get(transcription_language)
        if tr_action:
            tr_action.setChecked(True)
        backend_action = self._backend_actions.get(backend)
        if backend_action:
            backend_action.setChecked(True)
        self._retranslate()

    def _handle_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason == QSystemTrayIcon.Trigger:
            self._invoke_callback("tray_click_open_settings", self._on_open_settings)

    def _rebuild_audio_menu(self) -> None:
        self._audio_menu.clear()
        self._audio_actions.clear()
        self._audio_group = QActionGroup(self)
        self._audio_group.setExclusive(True)

        try:
            devices, current_token = self._audio_devices_provider()
        except Exception as exc:
            error_action = QAction(f"{self._t('tray_error_prefix')}: {exc}", self)
            error_action.setEnabled(False)
            self._audio_menu.addAction(error_action)
            self._current_microphone = self._t("status_mic_default_short")
            return

        for item in devices:
            token = str(item.get("token", ""))
            label = str(item.get("label", "Unknown device"))
            if not token:
                continue
            action = QAction(label, self)
            action.setCheckable(True)
            action.setChecked(token == current_token)
            action.triggered.connect(
                lambda checked=False, t=token: self._queue_callback(
                    f"audio_device:{t}",
                    lambda t=t: self._on_audio_device(t),
                    close_menu=False,
                )
            )
            self._audio_group.addAction(action)
            self._audio_menu.addAction(action)
            self._audio_actions[token] = action

        selected = self._audio_actions.get(current_token)
        if selected is not None:
            self._current_microphone = selected.text()
        elif current_token == "default":
            self._current_microphone = self._t("status_mic_default_short")
        else:
            self._current_microphone = self._t("status_mic_device", token=current_token)

    def _retranslate(self) -> None:
        self._section_row.set_label(self._t("tray_menu_configuration_header"))
        self._model_row.set_label(self._t("tray_menu_model"))
        self._transcription_row.set_label(self._t("tray_menu_transcription_language"))
        self._backend_row.set_label(self._t("tray_menu_backend"))
        self._audio_row.set_label(self._t("tray_menu_microphone"))
        self._settings_row.set_label(self._t("tray_menu_settings"))
        self._unload_row.set_label(self._t("tray_menu_unload_model"))
        self._update_unload_never_row()
        self._rescue_copy_row.set_label(self._t("tray_menu_last_dictation"))
        self._rescue_copy_row.set_value_pill(self._t("tray_action_copy"))
        self._quit_row.set_label(self._t("tray_menu_quit"))
        self._footer_row.set_label("")

        self._model_row.set_value(self._current_model)
        self._transcription_row.set_value(self._current_transcription_language)
        self._backend_row.set_value(self._current_backend)
        self._audio_row.set_value(self._current_microphone)

        self._refresh_model_actions()

    def _model_status(self) -> dict[str, bool]:
        try:
            return self._model_status_provider()
        except Exception:
            log.exception("Could not read model status")
            return {}

    def _update_unload_row(self) -> None:
        available = False
        try:
            available = bool(self._unload_available_provider())
        except Exception:
            log.exception("Could not read unload availability")
        self._unload_row_action.setEnabled(available)
        self._unload_row.set_enabled(available)

    def _update_unload_never_row(self) -> None:
        enabled = False
        try:
            enabled = bool(self._unload_never_provider())
        except Exception:
            log.exception("Could not read unload-never state")
        suffix = " \u2713" if enabled else ""
        self._unload_never_row.set_label(self._t("tray_menu_unload_never") + suffix)
        self._unload_never_row.clear_value()
        self._unload_never_row_action.setEnabled(True)
        self._unload_never_row.set_enabled(True)

    def _update_rescue_copy_row(self) -> None:
        available = False
        try:
            available = bool(self._rescue_copy_available_provider())
        except Exception:
            log.exception("Could not read rescue-copy availability")
        self._rescue_copy_row_action.setEnabled(available)
        self._rescue_copy_row.set_enabled(available)
        self._rescue_copy_row.set_value_pill(self._t("tray_action_copy"))

    def _refresh_model_actions(self) -> None:
        statuses = self._model_status()
        for model, action in self._model_actions.items():
            installed = bool(statuses.get(model, False))
            if installed:
                action.setText(model)
            else:
                action.setText(f"{model} {self._t('tray_model_install_suffix')}")
            action.setCheckable(installed)
            if not installed:
                action.setChecked(False)

    def _on_model_action(self, model: str) -> None:
        installed = bool(self._model_status().get(model, False))
        if installed:
            self._on_model(model)
            return
        self._on_model_install(model)

    def _t(self, key: str, **kwargs: object) -> str:
        return tr(self._ui_language, key, **kwargs)

    def _invoke_callback(self, label: str, callback: Callable[[], None], close_menu: bool = True) -> None:
        self._queue_callback(label, callback, close_menu=close_menu)

    def _queue_callback(self, label: str, callback: Callable[[], None], close_menu: bool = True) -> None:
        try:
            log.debug("Queueing tray callback: %s (close_menu=%s)", label, close_menu)
            submenu_open = any(
                menu.isVisible()
                for menu in (
                    self._model_menu,
                    self._transcription_language_menu,
                    self._backend_menu,
                    self._audio_menu,
                )
            )
            if close_menu:
                reopen_pos = QPoint(self.menu.pos()) if self.menu.isVisible() else QPoint(QCursor.pos())
                self.menu.close()
                self._model_menu.close()
                self._transcription_language_menu.close()
                self._backend_menu.close()
                self._audio_menu.close()
                QTimer.singleShot(0, lambda: self._run_callback(label, callback, close_menu, reopen_pos))
                return

            if submenu_open:
                reopen_pos = QPoint(self.menu.pos()) if self.menu.isVisible() else QPoint(QCursor.pos())
                self._model_menu.close()
                self._transcription_language_menu.close()
                self._backend_menu.close()
                self._audio_menu.close()
                QTimer.singleShot(0, lambda: self._run_callback(label, callback, False, reopen_pos))
                return

            QTimer.singleShot(0, lambda: self._run_callback(label, callback, False, QPoint()))
        except Exception:
            log.exception("Tray callback queueing failed: %s", label)

    def _run_callback(self, label: str, callback: Callable[[], None], close_menu: bool, reopen_pos: QPoint) -> None:
        try:
            log.debug("Running tray callback: %s", label)
            callback()
            log.debug("Tray callback completed: %s", label)
            if not close_menu and not reopen_pos.isNull():
                QTimer.singleShot(0, lambda pos=QPoint(reopen_pos): self._reopen_main_menu(pos))
        except Exception:
            log.exception("Tray callback failed: %s", label)

    def _reopen_main_menu(self, pos: QPoint) -> None:
        try:
            self.refresh_status()
            self.menu.popup(pos)
        except Exception:
            log.exception("Tray menu reopen failed.")
