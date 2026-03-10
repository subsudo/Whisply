from __future__ import annotations

import ctypes
import logging
import threading
import time
from collections.abc import Callable

import keyboard

log = logging.getLogger(__name__)

KEY_MAP = {
    "win": "windows",
    "windows": "windows",
    "left windows": "windows",
    "right windows": "windows",
    "linke windows": "windows",
    "rechte windows": "windows",
    "window": "windows",
    "super": "windows",
    "space": "space",
    "alt": "alt",
    "left alt": "alt",
    "right alt": "alt",
    "linke alt": "alt",
    "rechte alt": "alt",
    "ctrl": "ctrl",
    "left ctrl": "ctrl",
    "right ctrl": "ctrl",
    "strg": "ctrl",
    "left strg": "ctrl",
    "right strg": "ctrl",
    "linke strg": "ctrl",
    "rechte strg": "ctrl",
    "shift": "shift",
    "left shift": "shift",
    "right shift": "shift",
    "umschalt": "shift",
    "left umschalt": "shift",
    "right umschalt": "shift",
    "linke umschalt": "shift",
    "rechte umschalt": "shift",
    "section": "§",
    "paragraph": "§",
}

MODIFIER_KEYS = {"windows", "alt", "ctrl", "shift"}
SAFE_TRIGGER_KEYS = {"space", *(f"f{i}" for i in range(1, 25))}
HOOK_ALIASES = {
    "windows": ["left windows", "right windows", "windows"],
    "ctrl": ["left ctrl", "right ctrl", "ctrl"],
    "alt": ["left alt", "right alt", "alt"],
    "shift": ["left shift", "right shift", "shift"],
}
VK_MODIFIER_CODES = {
    "windows": (0x5B, 0x5C),  # VK_LWIN, VK_RWIN
    "ctrl": (0xA2, 0xA3),  # VK_LCONTROL, VK_RCONTROL
    "alt": (0xA4, 0xA5),  # VK_LMENU, VK_RMENU
    "shift": (0xA0, 0xA1),  # VK_LSHIFT, VK_RSHIFT
}


def _normalize_key_name(key_name: str) -> str:
    return KEY_MAP.get(key_name.strip().lower(), key_name.strip().lower())


def _is_vk_down(vk_code: int) -> bool:
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk_code) & 0x8000)
    except Exception:
        return False


def normalize_hotkey_combination(combination: str) -> str:
    raw_parts = [p for p in (part.strip() for part in combination.split("+")) if p]
    if not raw_parts:
        raise ValueError("Hotkey darf nicht leer sein.")

    normalized_parts: list[str] = []
    seen: set[str] = set()
    for part in raw_parts:
        normalized = _normalize_key_name(part)
        if normalized in seen:
            continue
        normalized_parts.append(normalized)
        seen.add(normalized)

    if len(normalized_parts) == 1:
        single_key = normalized_parts[0]
        if single_key in MODIFIER_KEYS:
            raise ValueError("Ein einzelner Hotkey darf keine Modifikatortaste sein.")
        return "win" if single_key == "windows" else single_key

    if len(normalized_parts) < 2:
        raise ValueError("Hotkey darf nicht leer sein.")

    if not any(key in MODIFIER_KEYS for key in normalized_parts):
        raise ValueError("Hotkey braucht mindestens eine Modifikatortaste (win/alt/ctrl/shift).")

    trigger_keys = [key for key in normalized_parts if key not in MODIFIER_KEYS]
    if len(trigger_keys) == 0:
        ordered_modifiers_only = [k for k in ("windows", "ctrl", "alt", "shift") if k in normalized_parts]
        human_readable_modifiers = [("win" if key == "windows" else key) for key in ordered_modifiers_only]
        return "+".join(human_readable_modifiers)

    if len(trigger_keys) != 1:
        raise ValueError("Hotkey darf hoechstens eine Ausloesetaste enthalten (z.B. space oder F-Taste).")

    trigger = trigger_keys[0]
    if trigger not in SAFE_TRIGGER_KEYS:
        raise ValueError("Bei Mehrfach-Hotkeys nur 'space' oder F1-F24 als Ausloesetaste, damit normale Tasten nicht blockiert werden.")

    ordered_modifiers = [k for k in ("windows", "ctrl", "alt", "shift") if k in normalized_parts]
    canonical_parts = [*ordered_modifiers, trigger]
    human_readable_parts = [("win" if key == "windows" else key) for key in canonical_parts]
    return "+".join(human_readable_parts)


def validate_hotkey_combination(combination: str) -> tuple[bool, str]:
    try:
        return True, normalize_hotkey_combination(combination)
    except ValueError as exc:
        return False, str(exc)


class HoldHotkey:
    def __init__(
        self,
        combination: str,
        on_down: Callable[[], None],
        on_up: Callable[[], None],
        debounce_ms: int = 200,
        debug_trace: bool = False,
        debug_global: bool = False,
    ) -> None:
        normalized = normalize_hotkey_combination(combination)
        self.keys = {_normalize_key_name(key) for key in normalized.split("+")}
        self.combination = normalized
        self.on_down = on_down
        self.on_up = on_up
        self._modifier_only = len(self.keys) > 1 and self.keys.issubset(MODIFIER_KEYS)
        self._debug_trace = debug_trace
        self._debug_global = debug_global
        self.debounce_s = debounce_ms / 1000.0
        self._last_event = 0.0
        self._pressed: set[str] = set()
        self._active = False
        self._hooks: list = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._last_snapshot: set[str] = set()

    def start(self) -> None:
        suppress = not self._modifier_only
        log.info(
            "Hotkey keys to match: %s (suppress=%s, modifier_only=%s, debug_trace=%s, debug_global=%s)",
            self.keys,
            suppress,
            self._modifier_only,
            self._debug_trace,
            self._debug_global,
        )
        if self._modifier_only:
            log.info("Modifier-only hotkey detected; using WinAPI modifier polling.")
            self._stop_event.clear()
            self._poll_thread = threading.Thread(target=self._modifier_poll_loop, name="hotkey-modifier-poll", daemon=True)
            self._poll_thread.start()
            if self._debug_global:
                self._hooks.append(keyboard.hook(self._on_global_event, suppress=False))
            return

        for key in self.keys:
            hook_keys = HOOK_ALIASES.get(key, [key])
            for hook_key in hook_keys:
                try:
                    if self._debug_trace:
                        log.info("Register hook: %s", hook_key)
                    h1 = keyboard.on_press_key(hook_key, self._on_key_down, suppress=suppress)
                    h2 = keyboard.on_release_key(hook_key, self._on_key_up, suppress=suppress)
                    self._hooks.extend([h1, h2])
                except ValueError as exc:
                    log.warning("Skipping unmapped hook key '%s': %s", hook_key, exc)
        if self._debug_global:
            if self._debug_trace:
                log.info("Register global keyboard hook for debugging")
            self._hooks.append(keyboard.hook(self._on_global_event, suppress=False))

    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=0.2)
            self._poll_thread = None
        for h in self._hooks:
            keyboard.unhook(h)
        self._hooks.clear()
        self._last_snapshot.clear()

    def _current_modifier_snapshot(self) -> set[str]:
        snapshot: set[str] = set()
        for key in self.keys:
            for vk_code in VK_MODIFIER_CODES.get(key, ()):
                if _is_vk_down(vk_code):
                    snapshot.add(key)
                    break
        return snapshot

    def _modifier_poll_loop(self) -> None:
        while not self._stop_event.wait(0.015):
            now = time.monotonic()
            with self._lock:
                self._pressed = self._current_modifier_snapshot()
                if self._debug_trace and self._pressed != self._last_snapshot:
                    log.info("Modifier snapshot changed: %s", sorted(self._pressed))
                    self._last_snapshot = set(self._pressed)

                if not self._active and self.keys.issubset(self._pressed):
                    if now - self._last_event >= self.debounce_s:
                        self._active = True
                        self._last_event = now
                        log.info("Hotkey activated")
                        self.on_down()
                elif self._active and not self.keys.issubset(self._pressed):
                    if now - self._last_event >= self.debounce_s:
                        self._active = False
                        self._last_event = now
                        log.info("Hotkey deactivated")
                        self.on_up()

    def _on_key_down(self, event) -> None:  # noqa: ANN001
        raw = (event.name or "").lower()
        name = KEY_MAP.get(raw, raw)
        if name not in self.keys:
            return
        now = time.monotonic()

        with self._lock:
            self._pressed.add(name)
            if self._debug_trace:
                log.info(
                    "Hotkey debug DOWN raw=%r normalized=%r scan_code=%s pressed=%s target=%s active=%s",
                    raw,
                    name,
                    getattr(event, "scan_code", "?"),
                    sorted(self._pressed),
                    sorted(self.keys),
                    self._active,
                )
            if not self._active and self.keys.issubset(self._pressed):
                if now - self._last_event >= self.debounce_s:
                    self._active = True
                    self._last_event = now
                    log.info("Hotkey activated")
                    self.on_down()
                elif self._debug_trace:
                    log.info(
                        "Hotkey activation blocked by debounce (elapsed_ms=%d < debounce_ms=%d)",
                        int((now - self._last_event) * 1000),
                        int(self.debounce_s * 1000),
                    )

    def _on_key_up(self, event) -> None:  # noqa: ANN001
        raw = (event.name or "").lower()
        name = KEY_MAP.get(raw, raw)
        if name not in self.keys:
            return
        now = time.monotonic()

        with self._lock:
            self._pressed.discard(name)
            if self._debug_trace:
                log.info(
                    "Hotkey debug UP raw=%r normalized=%r scan_code=%s pressed=%s target=%s active=%s",
                    raw,
                    name,
                    getattr(event, "scan_code", "?"),
                    sorted(self._pressed),
                    sorted(self.keys),
                    self._active,
                )
            if self._active and not self.keys.issubset(self._pressed):
                if now - self._last_event >= self.debounce_s:
                    self._active = False
                    self._last_event = now
                    log.info("Hotkey deactivated")
                    self.on_up()
                elif self._debug_trace:
                    log.info(
                        "Hotkey deactivation blocked by debounce (elapsed_ms=%d < debounce_ms=%d)",
                        int((now - self._last_event) * 1000),
                        int(self.debounce_s * 1000),
                    )

    def _on_global_event(self, event) -> None:  # noqa: ANN001
        raw = (event.name or "").lower()
        normalized = KEY_MAP.get(raw, raw)
        event_type = getattr(event, "event_type", "?")
        scan_code = getattr(event, "scan_code", "?")
        if self._debug_global:
            log.info("Global key event type=%s raw=%r normalized=%r scan_code=%s", event_type, raw, normalized, scan_code)

        _ = normalized
