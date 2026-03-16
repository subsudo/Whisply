from __future__ import annotations

from collections.abc import Callable
import math

from PySide6.QtCore import QPoint, QRect, QTimer, Qt
from PySide6.QtGui import QColor, QFontMetrics, QLinearGradient, QPainter, QPainterPath
from PySide6.QtWidgets import QApplication, QWidget

# ── Bar count & idle animation ────────────────────────────────────────────────
_BARS = 12

# A bar is considered "idle" when its display level is below this threshold.
_ANIMATION_IDLE_CUTOFF = 0.018

# Idle breathing: a slow sine wave ripples across the dot row while listening.
# Amplitude is in pixels (±px height change on each dot).
_IDLE_BREATHE_PX         = 1.5    # ±px oscillation amplitude
_IDLE_BREATHE_SPEED      = 0.022  # rad/tick  ≈ 0.67 Hz at 30 fps
_IDLE_BREATHE_PHASE_STEP = 0.45   # rad per bar  → wave sweeps left→right

# ── Per-bar smoothing ─────────────────────────────────────────────────────────
# Outer bars (bass / treble) react slightly faster; mid-range lingers a touch.
_BAR_ATTACK = (0.88, 0.90, 0.91, 0.92, 0.92, 0.93, 0.93, 0.94, 0.94, 0.95, 0.96, 0.97)
_BAR_DECAY  = (0.28, 0.30, 0.32, 0.33, 0.33, 0.35, 0.35, 0.38, 0.38, 0.42, 0.44, 0.46)

# ── Per-bar normalisation ─────────────────────────────────────────────────────
_INITIAL_BAR_PEAK = 300.0   # starting adaptive peak (raw FFT magnitude units)
_BAND_COMPRESS    = 2.4     # soft-knee exponent (higher → more compression)
_MAX_VISUAL_LEVEL = 0.88    # clamp for normalised bar height

# ── Colour defaults ───────────────────────────────────────────────────────────
_DEFAULT_GRADIENT_LEFT  = "#56F64E"
_DEFAULT_GRADIENT_RIGHT = "#0096FF"

# Edge envelope: prevent extreme bars from peaking as high as center bars.
_EDGE_ENVELOPE = (
    0.52, 0.64, 0.76, 0.87, 0.95, 1.00,
    1.00, 0.95, 0.87, 0.76, 0.64, 0.52,
)


class OverlayWidget(QWidget):
    def __init__(self, config: dict):
        super().__init__()
        self.cfg    = config
        self.state  = "idle"
        self.message = ""
        self._message_level = "notice"
        self._phase  = 0.0          # used for transcribing / model-warmup animation
        self._fade   = 0.0
        self._fade_target = 0.0
        self._base_width = int(config["width"])
        self._base_height = int(config["height"])

        self._error_fade_timer:       QTimer | None = None
        self._warmup_finish_timer:    QTimer | None = None
        self._warmup_finish_progress  = 0.0
        self._warmup_finish_step      = 0.0
        self._warmup_finish_remaining_steps = 0
        self._warmup_finish_callback: Callable[[], None] | None = None
        self._loading_completion_callback: Callable[[], None] | None = None
        self._loading_progress = 0
        self._display_loading_progress = 0.0

        # ── per-bar state (reset on show_recording) ───────────────────────────
        self._bar_targets        = [0.0] * _BARS   # normalised/compressed target
        self._bar_display_levels = [0.0] * _BARS   # smoothed display value
        self._bar_peaks          = [_INITIAL_BAR_PEAK] * _BARS  # adaptive peak
        self._idle_phase         = 0.0             # drives breathing wave

        self.setFixedSize(self._base_width, self._base_height)
        self.setWindowFlags(
            Qt.FramelessWindowHint
            | Qt.Tool
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus
            | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setWindowOpacity(0.0)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.setInterval(33)

    # ── positioning ──────────────────────────────────────────────────────────

    def place_bottom_center(self) -> None:
        screen = self._target_screen()
        if screen is None:
            return
        geo = screen.availableGeometry()
        x = geo.x() + (geo.width() - self.width()) // 2
        bottom_offset = int(self.cfg.get("bottom_offset", 100))
        y = geo.y() + geo.height() - self.height() - bottom_offset
        self.move(QPoint(x, y))

    def _target_screen(self):
        app = QApplication.instance()
        screens = app.screens() if app is not None else []
        if not screens:
            handle = self.windowHandle()
            return self.screen() or (handle.screen() if handle else None)

        try:
            preferred_index = int(self.cfg.get("monitor_index", -1))
        except (TypeError, ValueError):
            preferred_index = -1

        if preferred_index >= 0 and preferred_index < len(screens):
            return screens[preferred_index]

        primary = app.primaryScreen() if app is not None else None
        return primary or screens[0]

    # ── state transitions ────────────────────────────────────────────────────

    def show_recording(self) -> None:
        self._cancel_error_timer()
        self._cancel_model_warmup_finish()
        self._loading_completion_callback = None
        self.state   = "recording"
        self.message = ""
        self._reset_overlay_size()
        self._bar_targets        = [0.0] * _BARS
        self._bar_display_levels = [0.0] * _BARS
        self._bar_peaks          = [_INITIAL_BAR_PEAK] * _BARS
        self._idle_phase         = 0.0
        self.place_bottom_center()
        self._begin_fade_in()

    def show_transcribing(self) -> None:
        self._cancel_error_timer()
        self._cancel_model_warmup_finish()
        self._loading_completion_callback = None
        self.state   = "transcribing"
        self.message = ""
        self._reset_overlay_size()
        self._phase  = 0.0
        self.place_bottom_center()
        self._begin_fade_in()

    def show_loading(self, msg: str) -> None:
        self._cancel_error_timer()
        self._cancel_model_warmup_finish()
        self._loading_completion_callback = None
        self.state   = "loading"
        self.message = msg
        self._reset_overlay_size()
        self._loading_progress = 0
        self._display_loading_progress = 0.0
        self.place_bottom_center()
        self._begin_fade_in()

    def show_model_warmup(self, msg: str) -> None:
        self._cancel_error_timer()
        self._cancel_model_warmup_finish()
        self._loading_completion_callback = None
        self.state   = "model_warmup"
        self.message = msg
        self._reset_overlay_size()
        self._loading_progress = 0
        self._display_loading_progress = 0.0
        self._phase  = 0.0
        self.place_bottom_center()
        self._begin_fade_in()

    def set_loading_progress(self, percent: int) -> None:
        self._loading_progress = max(0, min(int(percent), 100))
        if self.state in {"loading", "model_warmup"} and self.isVisible():
            self.update()

    def show_done(self, ms: int = 300) -> None:
        _ = ms
        self._cancel_error_timer()
        self._cancel_model_warmup_finish()
        self._loading_completion_callback = None
        if self.state == "transcribing":
            self.hide_immediate()
            return
        if self.state == "loading":
            self.set_loading_progress(100)
            if self._display_loading_progress >= 99.5:
                self._begin_fade_out()
            else:
                self._loading_completion_callback = self._begin_fade_out
            return
        self._begin_fade_out()

    def show_error(self, msg: str, ms: int = 2000) -> None:
        self._show_message(msg=msg, ms=ms, level="error")

    def show_warning(self, msg: str, ms: int = 1800) -> None:
        self._show_message(msg=msg, ms=ms, level="warning")

    def show_notice(self, msg: str, ms: int = 1600) -> None:
        self._show_message(msg=msg, ms=ms, level="notice")

    def _show_message(self, msg: str, ms: int, level: str) -> None:
        self._cancel_error_timer()
        self._cancel_model_warmup_finish()
        self._loading_completion_callback = None
        self.state   = "message"
        self.message = msg
        self._message_level = str(level or "notice")
        self._resize_for_message(msg)
        self.place_bottom_center()
        self._begin_fade_in()
        if self._error_fade_timer is None:
            self._error_fade_timer = QTimer(self)
            self._error_fade_timer.setSingleShot(True)
            self._error_fade_timer.timeout.connect(self._begin_fade_out)
        self._error_fade_timer.start(max(0, int(ms)))

    def set_audio_levels(self, levels: list[float]) -> None:
        """Receive 12 per-band FFT levels; update adaptive peaks and targets."""
        for i in range(min(len(levels), _BARS)):
            raw = max(0.0, float(levels[i]))
            # Slow-release adaptive peak preserves local loudness context.
            if raw > self._bar_peaks[i]:
                self._bar_peaks[i] = raw
            else:
                self._bar_peaks[i] = max(raw, self._bar_peaks[i] * 0.9985)
            # Normalise then soft-knee compress to 0 … _MAX_VISUAL_LEVEL.
            norm       = raw / max(1.0, self._bar_peaks[i])
            compressed = 1.0 - math.exp(-norm * _BAND_COMPRESS)
            self._bar_targets[i] = max(0.0, min(compressed * _MAX_VISUAL_LEVEL, _MAX_VISUAL_LEVEL))

    def finish_model_warmup(self, ms: int = 220, on_finished: Callable[[], None] | None = None) -> None:
        _ = ms
        if self.state != "model_warmup":
            if on_finished is not None:
                on_finished()
            return

        self._cancel_model_warmup_finish()
        self._warmup_finish_callback = on_finished
        self._loading_completion_callback = self._complete_model_warmup_finish
        self.set_loading_progress(100)
        if self._display_loading_progress >= 99.5:
            self._complete_loading_completion()

    def _complete_model_warmup_finish(self) -> None:
        callback = self._warmup_finish_callback
        self._warmup_finish_callback = None
        if callback is not None:
            callback()

    def _on_model_warmup_finish_tick(self) -> None:
        if self.state != "model_warmup":
            self._cancel_model_warmup_finish()
            return

        self._warmup_finish_remaining_steps -= 1
        self._warmup_finish_progress        += self._warmup_finish_step

        if self._warmup_finish_remaining_steps <= 0 or self._warmup_finish_progress >= 100.0:
            self.set_loading_progress(100)
            callback = self._warmup_finish_callback
            self._cancel_model_warmup_finish()
            if callback is not None:
                callback()
            return

        self.set_loading_progress(int(self._warmup_finish_progress))

    # ── fade helpers ─────────────────────────────────────────────────────────

    def _begin_fade_in(self) -> None:
        self._ensure_timer_running()
        if not self.isVisible():
            self._fade = 0.0
            self.setWindowOpacity(0.0)
            self.show()
        self._fade_target = 1.0

    def _begin_fade_out(self) -> None:
        if not self.isVisible() or self._fade <= 0.0:
            self.hide_immediate()
            return
        self._ensure_timer_running()
        self._fade_target = 0.0

    def hide_immediate(self) -> None:
        self._cancel_error_timer()
        self._cancel_model_warmup_finish()
        self._loading_completion_callback = None
        self._fade        = 0.0
        self._fade_target = 0.0
        self._reset_overlay_size()
        self._bar_display_levels = [0.0] * _BARS
        self._bar_targets        = [0.0] * _BARS
        self._loading_progress   = 0
        self._display_loading_progress = 0.0
        self.setWindowOpacity(0.0)
        if self.isVisible():
            self.hide()
        self._stop_timer_if_hidden()

    def _ensure_timer_running(self) -> None:
        if not self._timer.isActive():
            self._timer.start()

    def _stop_timer_if_hidden(self) -> None:
        if not self.isVisible() and self._fade_target <= 0.0 and self._fade <= 0.0:
            if self._timer.isActive():
                self._timer.stop()

    def _cancel_error_timer(self) -> None:
        if self._error_fade_timer and self._error_fade_timer.isActive():
            self._error_fade_timer.stop()

    def _cancel_model_warmup_finish(self) -> None:
        if self._warmup_finish_timer and self._warmup_finish_timer.isActive():
            self._warmup_finish_timer.stop()
        self._warmup_finish_progress        = float(self._loading_progress)
        self._warmup_finish_step            = 0.0
        self._warmup_finish_remaining_steps = 0
        self._warmup_finish_callback        = None

    def _complete_loading_completion(self) -> None:
        callback = self._loading_completion_callback
        if callback is None:
            return
        # Consume the callback before invoking it so repeated ticks cannot re-fire it.
        self._loading_completion_callback = None
        callback()

    def _reset_overlay_size(self) -> None:
        self.setFixedSize(self._base_width, self._base_height)

    def _resize_for_message(self, msg: str) -> None:
        font = self.font()
        font.setPointSize(9)
        font.setBold(True)
        metrics = QFontMetrics(font)
        target_width = max(self._base_width, min(420, metrics.horizontalAdvance(str(msg or "")) + 40))
        self.setFixedSize(target_width, self._base_height)

    def _advance_loading_display_progress(self) -> None:
        raw = float(max(0, min(self._loading_progress, 100)))
        display = float(self._display_loading_progress)

        if raw >= 100.0:
            if display < 85.0:
                step = 0.90
            elif display < 97.0:
                step = 0.65
            else:
                step = 0.35
            self._display_loading_progress = min(100.0, display + step)
            return

        if display < 30.0:
            base_step = 0.38
        elif display < 60.0:
            base_step = 0.30
        elif display < 85.0:
            base_step = 0.22
        elif display < 95.0:
            base_step = 0.14
        else:
            base_step = 0.06

        gap_boost = min(0.30, max(0.0, raw - display) * 0.03)
        self._display_loading_progress = min(99.0, display + base_step + gap_boost)

    def _advance_model_warmup_display_progress(self) -> None:
        target = float(max(0, min(self._loading_progress, 100)))
        self._display_loading_progress = target

    # ── tick / animation loop ────────────────────────────────────────────────

    def _tick(self) -> None:
        if not self.isVisible() and self._fade_target <= 0.0 and self._fade <= 0.0:
            self._stop_timer_if_hidden()
            return

        if self.state == "transcribing":
            self._phase += 0.112

        if self.state == "loading":
            self._advance_loading_display_progress()
            if self._loading_completion_callback is not None and self._loading_progress >= 100 and self._display_loading_progress >= 99.5:
                self._complete_loading_completion()
        elif self.state == "model_warmup":
            self._advance_model_warmup_display_progress()
            if self._loading_completion_callback is not None and self._display_loading_progress >= 99.5:
                self._complete_loading_completion()

        if self.state == "recording":
            is_idle = max(self._bar_targets) < _ANIMATION_IDLE_CUTOFF
            if is_idle:
                # Smoothly decay bars → zero; advance breathing phase.
                for i in range(_BARS):
                    self._bar_display_levels[i] *= 0.70
                    if self._bar_display_levels[i] < 0.001:
                        self._bar_display_levels[i] = 0.0
                self._idle_phase += _IDLE_BREATHE_SPEED
            else:
                # Per-bar attack / decay smoothing.
                for i in range(_BARS):
                    tgt = self._bar_targets[i]
                    cur = self._bar_display_levels[i]
                    if tgt > cur:
                        self._bar_display_levels[i] = min(1.0, cur + (tgt - cur) * _BAR_ATTACK[i])
                    else:
                        self._bar_display_levels[i] = max(0.0, cur + (tgt - cur) * _BAR_DECAY[i])
                self._idle_phase = 0.0  # reset so next idle always starts smoothly

        # Fade animation
        cfg_opacity = self.cfg.get("opacity", 1.0)
        if self._fade < self._fade_target:
            self._fade = min(self._fade + 0.25, self._fade_target)
            self.setWindowOpacity(self._fade * cfg_opacity)
        elif self._fade > self._fade_target:
            self._fade = max(self._fade - 0.16, self._fade_target)
            self.setWindowOpacity(self._fade * cfg_opacity)
            if self._fade <= 0.0:
                self.hide_immediate()
                return

        if self.isVisible():
            self.update()
        else:
            self._stop_timer_if_hidden()

    # ── paint dispatch ───────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802, ANN001
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.state == "recording":
            self._paint_waveform(painter)
        elif self.state == "loading":
            self._paint_loading(painter)
        elif self.state == "model_warmup":
            self._paint_model_warmup(painter)
        elif self.state == "transcribing":
            self._paint_transcribing(painter)
        elif self.state == "message":
            self._paint_message(painter)

    # ── recording: FFT spectrum bars ─────────────────────────────────────────

    def _paint_waveform(self, painter: QPainter) -> None:
        pill_rect    = self._paint_pill_background(painter, alpha=145)
        content_rect = pill_rect.adjusted(8, 5, -8, -5)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._wave_gradient(content_rect))

        w       = max(1, content_rect.width())
        h       = max(1, content_rect.height())
        spacing = w / _BARS
        bar_w   = max(3, int(spacing * 0.55))

        is_idle = max(self._bar_display_levels) < _ANIMATION_IDLE_CUTOFF

        for i in range(_BARS):
            dot_size = float(bar_w)
            if is_idle:
                # Idle: round dots with a gentle breathing wave.
                breathe = _IDLE_BREATHE_PX * math.sin(
                    self._idle_phase + i * _IDLE_BREATHE_PHASE_STEP
                )
                bar_h  = max(1.0, dot_size + breathe)
                draw_w = bar_w
            else:
                # Active: bar height driven directly by per-band FFT level.
                level  = self._bar_display_levels[i]
                edge_gain = _EDGE_ENVELOPE[i]
                bar_h  = max(dot_size, min(h * 0.92 * level * edge_gain, h * 0.92))
                draw_w = bar_w

            x      = int(content_rect.left() + i * spacing + (spacing - draw_w) * 0.5)
            y      = int(content_rect.top() + h * 0.50 - bar_h / 2)
            corner = min(draw_w / 2.0, 5.0)
            painter.drawRoundedRect(x, y, int(draw_w), int(bar_h), corner, corner)

    # ── transcribing / model-warmup ──────────────────────────────────────────

    def _paint_transcribing(self, painter: QPainter) -> None:
        pill_rect    = self._paint_pill_background(painter, alpha=145)
        content_rect = pill_rect.adjusted(8, 5, -8, -5)
        self._draw_transcribing_bars(
            painter=painter,
            content_rect=content_rect,
            animated=True,
            fill_ratio=1.0,
            paint_base=False,
        )

    def _paint_model_warmup(self, painter: QPainter) -> None:
        pill_rect    = self._paint_pill_background(painter, alpha=145)
        content_rect = pill_rect.adjusted(8, 5, -8, -5)
        fill_ratio   = max(0.0, min(self._display_loading_progress / 100.0, 1.0))
        self._draw_transcribing_bars(
            painter=painter,
            content_rect=content_rect,
            animated=False,
            fill_ratio=fill_ratio,
            paint_base=True,
        )

    def _transcribing_bar_rects(self, content_rect: QRect, phase: float) -> list[tuple[int, int, int, int]]:
        w       = max(1, content_rect.width())
        h       = max(1, content_rect.height())
        spacing = w / _BARS
        bar_w   = max(3, int(spacing * 0.55))
        rects: list[tuple[int, int, int, int]] = []

        for i in range(_BARS):
            drift = math.sin(phase * 2.6 - i * 0.72)
            amp   = 0.52 + 0.30 * drift
            edge_gain = _EDGE_ENVELOPE[i]
            bar_h = max(3.0, min(h * 0.88 * amp * edge_gain, h * 0.92))
            draw_h = max(3, int(round(bar_h)))
            x = int(round(content_rect.left() + i * spacing + (spacing - bar_w) * 0.5))
            y = int(round(content_rect.top() + h * 0.50 - draw_h / 2.0))
            rects.append((x, y, bar_w, draw_h))

        return rects

    def _draw_transcribing_bars(
        self,
        painter: QPainter,
        content_rect: QRect,
        animated: bool,
        fill_ratio: float,
        paint_base: bool,
    ) -> None:
        painter.setPen(Qt.NoPen)

        w       = max(1, content_rect.width())
        bar_rects = self._transcribing_bar_rects(
            content_rect=content_rect,
            phase=self._phase if animated else 0.0,
        )
        bar_w = bar_rects[0][2] if bar_rects else max(3, int((w / _BARS) * 0.55))
        corner  = min(bar_w / 2.0, 5.0)

        if paint_base:
            painter.setBrush(QColor(255, 255, 255, 38))
            for x, y, rect_w, rect_h in bar_rects:
                painter.drawRoundedRect(x, y, rect_w, rect_h, corner, corner)

        fill_px = int(w * max(0.0, min(fill_ratio, 1.0)))
        if fill_px <= 0:
            return

        gradient = self._wave_gradient(content_rect)
        fill_end = content_rect.left() + fill_px

        painter.save()
        painter.setBrush(gradient)

        for x, y, rect_w, rect_h in bar_rects:
            if animated:
                painter.drawRoundedRect(x, y, rect_w, rect_h, corner, corner)
                continue

            filled_w = max(0, min(rect_w, fill_end - x))
            if filled_w <= 0:
                continue
            clip_path = QPainterPath()
            clip_path.addRoundedRect(float(x), float(y), float(rect_w), float(rect_h), corner, corner)
            painter.save()
            painter.setClipPath(clip_path)
            painter.fillRect(x, y, int(filled_w), rect_h, gradient)
            painter.restore()
        painter.restore()

    # ── loading bar ──────────────────────────────────────────────────────────

    def _paint_loading(self, painter: QPainter) -> None:
        pill_rect  = self._paint_pill_background(painter, alpha=145)
        text_color, _ = self._wave_colors()
        text_rect  = pill_rect.adjusted(16, 8, -16, -28)
        painter.setPen(text_color)
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        msg = self.message or "Loading model"
        msg = metrics.elidedText(msg, Qt.ElideRight, max(10, text_rect.width() - 52))
        display_progress = int(round(max(0.0, min(self._display_loading_progress, 100.0))))
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, msg)
        painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignRight, f"{display_progress}%")

        bar_rect = QRect(
            pill_rect.left() + 24,
            pill_rect.bottom() - 26,
            max(8, pill_rect.width() - 48),
            11,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 36))
        painter.drawRoundedRect(bar_rect, 4, 4)

        if bar_rect.width() <= 4:
            return
        fill_w = int((bar_rect.width() * max(0.0, min(self._display_loading_progress, 100.0))) / 100.0)
        if fill_w <= 0:
            return
        seg_rect = QRect(bar_rect.left(), bar_rect.top(), fill_w, bar_rect.height())
        painter.setBrush(self._wave_gradient(bar_rect))
        painter.drawRoundedRect(seg_rect, 4, 4)

    # ── colour helpers ───────────────────────────────────────────────────────

    def _wave_colors(self) -> tuple[QColor, QColor]:
        style = str(self.cfg.get("waveform_style", "gradient")).strip().lower()
        primary_raw = str(
            self.cfg.get(
                "waveform_gradient_start",
                self.cfg.get("waveform_color", _DEFAULT_GRADIENT_LEFT),
            )
        )
        secondary_raw = str(self.cfg.get("waveform_gradient_end", _DEFAULT_GRADIENT_RIGHT))

        primary = QColor(primary_raw)
        if not primary.isValid():
            primary = QColor(_DEFAULT_GRADIENT_LEFT)

        if style == "single":
            single_raw = str(self.cfg.get("waveform_color", primary.name()))
            single     = QColor(single_raw)
            if not single.isValid():
                single = primary
            return single, single

        secondary = QColor(secondary_raw)
        if not secondary.isValid():
            secondary = QColor(_DEFAULT_GRADIENT_RIGHT)
        return primary, secondary

    def _wave_gradient(self, rect: QRect) -> QLinearGradient:
        left_color, right_color = self._wave_colors()
        gradient = QLinearGradient(rect.left(), rect.top(), rect.right(), rect.top())
        gradient.setColorAt(0.0, left_color)
        gradient.setColorAt(1.0, right_color)
        return gradient

    # ── pill background ──────────────────────────────────────────────────────

    def _paint_pill_background(self, painter: QPainter, alpha: int) -> QRect:
        rect   = self.rect().adjusted(6, 4, -6, -4)
        radius = rect.height() / 2
        bg = QColor(7, 10, 15, max(0, min(alpha, 255)))
        edge = QColor(42, 50, 64, 48)
        painter.setPen(edge)
        painter.setBrush(bg)
        painter.drawRoundedRect(rect, radius, radius)
        return rect

    # ── message ──────────────────────────────────────────────────────────────

    def _paint_message(self, painter: QPainter) -> None:
        pill_rect = self._paint_pill_background(painter, alpha=145)
        if self._message_level == "error":
            color = QColor(self.cfg["color_error"])
        elif self._message_level == "warning":
            color = QColor(str(self.cfg.get("color_transcribing", "#FFC107")))
        else:
            color = QColor(str(self.cfg.get("color_done", "#2196F3")))
        painter.setPen(color)
        font = painter.font()
        font.setPointSize(9)
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        msg = metrics.elidedText(self.message, Qt.ElideRight, max(20, pill_rect.width() - 26))
        painter.drawText(pill_rect.adjusted(12, 0, -12, 0), Qt.AlignCenter, msg)
