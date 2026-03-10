from __future__ import annotations

import ctypes
import logging
import time

import pyperclip


log = logging.getLogger(__name__)

VK_CONTROL = 0x11
VK_V = 0x56
KEYEVENTF_KEYUP = 0x0002


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("flags", ctypes.c_uint),
        ("hwndActive", ctypes.c_void_p),
        ("hwndFocus", ctypes.c_void_p),
        ("hwndCapture", ctypes.c_void_p),
        ("hwndMenuOwner", ctypes.c_void_p),
        ("hwndMoveSize", ctypes.c_void_p),
        ("hwndCaret", ctypes.c_void_p),
        ("rcCaret_left", ctypes.c_long),
        ("rcCaret_top", ctypes.c_long),
        ("rcCaret_right", ctypes.c_long),
        ("rcCaret_bottom", ctypes.c_long),
    ]


_LIKELY_EDIT_CLASSES = {
    "edit",
    "richedit20w",
    "richedit50w",
    "scintilla",
    "chrome_renderwidgethosthwnd",
    "mozillawindowclass",
    "windows.ui.core.corewindow",
    "directuihwnd",
}


def _send_ctrl_v() -> None:
    user32 = ctypes.windll.user32
    user32.keybd_event(VK_CONTROL, 0, 0, 0)
    user32.keybd_event(VK_V, 0, 0, 0)
    user32.keybd_event(VK_V, 0, KEYEVENTF_KEYUP, 0)
    user32.keybd_event(VK_CONTROL, 0, KEYEVENTF_KEYUP, 0)


def _focused_hwnd() -> int:
    user32 = ctypes.windll.user32
    foreground = int(user32.GetForegroundWindow() or 0)
    if not foreground:
        return 0

    thread_id = int(user32.GetWindowThreadProcessId(ctypes.c_void_p(foreground), None) or 0)
    if not thread_id:
        return 0

    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(GUITHREADINFO)
    if not user32.GetGUIThreadInfo(thread_id, ctypes.byref(info)):
        return 0
    return int(info.hwndFocus or info.hwndCaret or 0)


def _class_name(hwnd: int) -> str:
    if not hwnd:
        return ""
    buffer = ctypes.create_unicode_buffer(256)
    try:
        length = ctypes.windll.user32.GetClassNameW(ctypes.c_void_p(hwnd), buffer, len(buffer))
    except Exception:
        return ""
    if length <= 0:
        return ""
    return buffer.value[:length]


class TextInserter:
    def __init__(
        self,
        paste_delay_ms: int = 50,
        restore_clipboard: bool = True,
        append_trailing_space: bool = True,
    ) -> None:
        self.paste_delay_ms = paste_delay_ms
        self.restore_clipboard = restore_clipboard
        self.append_trailing_space = append_trailing_space

    def prepare_text(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        if self.append_trailing_space and not text[-1].isspace():
            text = f"{text} "
        return text

    def copy_to_clipboard(self, text: str) -> bool:
        payload = self.prepare_text(text)
        if not payload:
            return False
        try:
            pyperclip.copy(payload)
            return True
        except Exception:
            log.exception("Failed to copy pending transcription to clipboard.")
            return False

    def can_insert_now(self) -> bool:
        hwnd = _focused_hwnd()
        if not hwnd:
            log.info("Insert target check: no focused control.")
            return False
        class_name = _class_name(hwnd).strip().lower()
        allowed = class_name in _LIKELY_EDIT_CLASSES or "richedit" in class_name
        log.info(
            "Insert target check: hwnd=%s class=%r editable=%s",
            hwnd,
            class_name,
            allowed,
        )
        return allowed

    def _insert_clipboard(self, text: str) -> None:
        original: str | None = None
        if self.restore_clipboard:
            try:
                original = pyperclip.paste()
            except Exception:
                original = None

        try:
            pyperclip.copy(text)
            time.sleep(0.02)
            _send_ctrl_v()
            time.sleep(self.paste_delay_ms / 1000.0)
        finally:
            if self.restore_clipboard and original is not None:
                time.sleep(0.05)
                try:
                    pyperclip.copy(original)
                except Exception:
                    pass

    def insert(self, text: str) -> None:
        payload = self.prepare_text(text)
        if not payload:
            return

        log.info("Inserting transcription via method=clipboard")
        self._insert_clipboard(payload)
