import os
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from first_run_dialog import FirstRunSetupDialog, _FirstRunWorker
from main import WhisperTypeApp, _is_cuda_fallback_oom, _should_persist_first_run_wizard
from overlay import OverlayWidget
from recorder import AudioRecorder
from settings_dialog import SettingsDialog
from transcriber import Transcriber
from tray import AppTray


class _FakeSignal:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def emit(self, *args) -> None:
        self.calls.append(args)


class _FakeTimer:
    def __init__(self, active: bool = False) -> None:
        self.active = active

    def isActive(self) -> bool:  # noqa: N802
        return self.active

    def stop(self) -> None:
        self.active = False


class _FailingTranscriber:
    def transcribe_async(self, audio: np.ndarray):  # noqa: ANN001
        raise RuntimeError("submit failed")


class _FakeButton:
    def __init__(self) -> None:
        self.enabled = True
        self.visible = True

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def show(self) -> None:
        self.visible = True

    def hide(self) -> None:
        self.visible = False


class _FakeLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _DirectEmit:
    def __init__(self, callback) -> None:  # noqa: ANN001
        self._callback = callback

    def emit(self, *args) -> None:  # noqa: ANN001
        self._callback(*args)


class _ImmediateThread:
    def __init__(self, target=None, name=None, daemon=None) -> None:  # noqa: ANN001
        self._target = target

    def start(self) -> None:
        if self._target is not None:
            self._target()


class _FakeCloseEvent:
    def __init__(self) -> None:
        self.ignored = False

    def ignore(self) -> None:
        self.ignored = True


class _FakeWorker:
    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.interruption_requested = False

    def isRunning(self) -> bool:  # noqa: N802
        return self.running

    def requestInterruption(self) -> None:
        self.interruption_requested = True


class _DoneCollector:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def emit(self, *args) -> None:
        self.calls.append(args)


class StabilityRound1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qt_app = QApplication.instance() or QApplication([])

    def test_submit_failure_releases_lock_and_resets_state(self) -> None:
        app = WhisperTypeApp.__new__(WhisperTypeApp)
        app.transcriber = _FailingTranscriber()
        app._transcribe_lock = threading.Lock()
        app._transcribe_lock.acquire()
        app._transcription_in_progress = True
        app._model_loading_active = True
        app._discard_timed_out_result = True
        app._transcribing_delay_timer = _FakeTimer(active=True)
        app._transcription_timeout_timer = _FakeTimer(active=True)
        app.bridge = types.SimpleNamespace(
            transcription_error=_FakeSignal(),
            transcription_settled=_FakeSignal(),
        )
        app._on_transcription_done = lambda future: None

        WhisperTypeApp._submit_transcription_job(app, np.zeros(8, dtype=np.float32))

        self.assertFalse(app._transcribe_lock.locked())
        self.assertFalse(app._transcription_in_progress)
        self.assertFalse(app._model_loading_active)
        self.assertFalse(app._discard_timed_out_result)
        self.assertFalse(app._transcribing_delay_timer.isActive())
        self.assertFalse(app._transcription_timeout_timer.isActive())
        self.assertEqual(app.bridge.transcription_settled.calls, [()])
        self.assertEqual(len(app.bridge.transcription_error.calls), 1)
        self.assertIn("submit failed", app.bridge.transcription_error.calls[0][0])

    def test_cuda_oom_detection_is_explicit_and_stable(self) -> None:
        self.assertTrue(_is_cuda_fallback_oom("CUDA out of memory while loading model"))
        self.assertTrue(_is_cuda_fallback_oom("cuda alloc failed for workspace"))
        self.assertFalse(_is_cuda_fallback_oom("alloc failed in generic buffer"))

    def test_transcriber_keeps_model_unloaded_when_cuda_and_cpu_load_fail(self) -> None:
        cuda_backend = Mock()
        cuda_backend.get_device_info.return_value = "cuda"
        cuda_backend.load_model.side_effect = RuntimeError("cuda boom")

        cpu_backend = Mock()
        cpu_backend.get_device_info.return_value = "cpu"
        cpu_backend.load_model.side_effect = RuntimeError("cpu boom")

        with patch("transcriber.create_backend", side_effect=[cuda_backend, cpu_backend]), patch(
            "transcriber.mark_installed"
        ) as mark_installed:
            transcriber = Transcriber(backend_name="cuda", download_root=None)
            with self.assertRaisesRegex(RuntimeError, "cpu boom"):
                transcriber._transcribe_with_lazy_load(np.zeros(32, dtype=np.float32))

        self.assertFalse(transcriber._model_loaded)
        mark_installed.assert_not_called()

    def test_overlay_completion_callback_fires_only_once(self) -> None:
        config = {
            "width": 220,
            "height": 72,
            "bottom_offset": 100,
            "opacity": 1.0,
            "monitor_index": -1,
            "waveform_gradient_left": "#56F64E",
            "waveform_gradient_right": "#0096FF",
        }
        overlay = OverlayWidget(config)
        overlay._fade = 1.0
        overlay._fade_target = 1.0

        loading_calls: list[str] = []
        overlay.state = "loading"
        overlay._loading_progress = 100
        overlay._display_loading_progress = 99.5
        overlay._loading_completion_callback = lambda: loading_calls.append("loading")
        overlay._tick()
        overlay._tick()

        warmup_calls: list[str] = []
        overlay.state = "model_warmup"
        overlay._loading_progress = 100
        overlay._display_loading_progress = 99.5
        overlay._loading_completion_callback = lambda: warmup_calls.append("warmup")
        overlay._tick()
        overlay._tick()

        self.assertEqual(loading_calls, ["loading"])
        self.assertEqual(warmup_calls, ["warmup"])
        overlay.hide_immediate()
        overlay.deleteLater()

    def test_overlay_message_timer_is_single_and_disarmed_on_state_change(self) -> None:
        config = {
            "width": 220,
            "height": 72,
            "bottom_offset": 100,
            "opacity": 1.0,
            "monitor_index": -1,
            "waveform_gradient_left": "#56F64E",
            "waveform_gradient_right": "#0096FF",
        }
        overlay = OverlayWidget(config)
        overlay.show_notice("first", ms=1000)
        timer = overlay._error_fade_timer
        first_generation = overlay._message_fade_generation

        overlay.show_warning("second", ms=1000)
        self.assertIs(timer, overlay._error_fade_timer)
        self.assertGreater(overlay._message_fade_generation, first_generation)

        overlay.state = "loading"
        overlay._fade_target = 1.0
        overlay._on_error_fade_timeout()
        self.assertEqual(overlay._fade_target, 1.0)
        overlay.hide_immediate()
        overlay.deleteLater()

    def test_overlay_audio_levels_handle_cross_thread_updates(self) -> None:
        config = {
            "width": 220,
            "height": 72,
            "bottom_offset": 100,
            "opacity": 1.0,
            "monitor_index": -1,
            "waveform_gradient_left": "#56F64E",
            "waveform_gradient_right": "#0096FF",
        }
        overlay = OverlayWidget(config)
        overlay.state = "recording"
        overlay._fade = 1.0
        overlay._fade_target = 1.0

        stop_event = threading.Event()
        errors: list[Exception] = []

        def writer() -> None:
            try:
                while not stop_event.is_set():
                    overlay.set_audio_levels([0.1] * 12)
                    overlay.set_audio_levels([0.8] * 12)
            except Exception as exc:  # pragma: no cover - defensive capture
                errors.append(exc)

        thread = threading.Thread(target=writer, daemon=True)
        thread.start()
        try:
            for _ in range(200):
                overlay._tick()
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

        self.assertEqual(errors, [])
        overlay.hide_immediate()
        overlay.deleteLater()

    def test_first_run_close_requests_interrupt_without_blocking_wait(self) -> None:
        dialog = FirstRunSetupDialog.__new__(FirstRunSetupDialog)
        dialog._worker = _FakeWorker(running=True)
        dialog._close_requested = False
        dialog._setup_aborted = False
        dialog.start_button = _FakeButton()
        dialog.skip_button = _FakeButton()
        dialog.status_label = _FakeLabel()
        dialog._set_options_enabled = Mock()
        dialog._t = lambda key, **kwargs: "aborted"

        event = _FakeCloseEvent()
        FirstRunSetupDialog.closeEvent(dialog, event)

        self.assertTrue(dialog._close_requested)
        self.assertTrue(dialog._setup_aborted)
        self.assertTrue(dialog._worker.interruption_requested)
        dialog._set_options_enabled.assert_called_once_with(False)
        self.assertFalse(dialog.start_button.enabled)
        self.assertFalse(dialog.skip_button.enabled)
        self.assertEqual(dialog.status_label.text, "aborted")
        self.assertTrue(event.ignored)

    def test_first_run_worker_marks_aborted_when_interrupted(self) -> None:
        collector = _DoneCollector()
        thread = _FirstRunWorker(
            install_cuda=False,
            install_cuda_cb=lambda cb=None: (True, "ok"),
            models=["medium"],
            backend_hint="cpu",
            download_root=".",
        )
        thread.done.connect(collector.emit)
        with patch.object(thread, "isInterruptionRequested", return_value=True), patch(
            "first_run_dialog.ensure_model_installed"
        ) as ensure_model_installed:
            thread.run()

        self.assertEqual(len(collector.calls), 1)
        self.assertTrue(collector.calls[0][-1])
        ensure_model_installed.assert_not_called()

    def test_first_run_payload_reports_aborted_setup(self) -> None:
        dialog = FirstRunSetupDialog.__new__(FirstRunSetupDialog)
        dialog.cuda_attempted = False
        dialog.cuda_success = False
        dialog.installed_models = []
        dialog.failed_models = []
        dialog.restart_recommended = False
        dialog._setup_started = True
        dialog._setup_finished = False
        dialog._setup_aborted = True

        payload = FirstRunSetupDialog.result_payload(dialog)

        self.assertTrue(payload["setup_started"])
        self.assertFalse(payload["setup_finished"])
        self.assertTrue(payload["setup_aborted"])

    def test_first_run_wizard_marker_not_persisted_for_aborted_payload(self) -> None:
        self.assertFalse(_should_persist_first_run_wizard({"setup_aborted": True}))
        self.assertTrue(_should_persist_first_run_wizard({"setup_aborted": False}))
        self.assertTrue(_should_persist_first_run_wizard({}))

    def test_list_input_devices_survives_missing_default_input(self) -> None:
        devices = [
            {"name": "Speaker", "max_input_channels": 0, "hostapi": 0},
            {"name": "USB Mic", "max_input_channels": 1, "hostapi": 0},
        ]
        hostapis = [{"name": "WASAPI"}]

        def fake_query_devices(device=None, kind=None):  # noqa: ANN001
            if kind == "input":
                raise RuntimeError("Error querying device -1")
            if device is None:
                return devices
            return devices[int(device)]

        with patch("recorder.sd.query_devices", side_effect=fake_query_devices), patch(
            "recorder.sd.query_hostapis", return_value=hostapis
        ):
            result = AudioRecorder.list_input_devices()

        self.assertEqual(
            result,
            [
                {
                    "token": "1",
                    "label": "USB Mic (WASAPI)",
                    "is_default": False,
                }
            ],
        )

    def test_recorder_start_falls_back_to_first_input_when_default_is_invalid(self) -> None:
        devices = [
            {"name": "Speaker", "max_input_channels": 0, "hostapi": 0},
            {"name": "USB Mic", "max_input_channels": 1, "hostapi": 0},
        ]
        opened_devices: list[int | None] = []

        def fake_query_devices(device=None, kind=None):  # noqa: ANN001
            if kind == "input":
                raise RuntimeError("Error querying device -1")
            if device is None:
                return devices
            return devices[int(device)]

        class _FakeStream:
            def __init__(self, **kwargs) -> None:  # noqa: ANN003
                opened_devices.append(kwargs.get("device"))

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def close(self) -> None:
                return None

        with patch("recorder.sd.query_devices", side_effect=fake_query_devices), patch(
            "recorder.sd.query_hostapis", return_value=[{"name": "WASAPI"}]
        ), patch("recorder.sd.InputStream", side_effect=lambda **kwargs: _FakeStream(**kwargs)):
            recorder = AudioRecorder()
            recorder.start()

        self.assertTrue(recorder.is_running)
        self.assertEqual(opened_devices, [1])

    def test_recorder_tries_next_input_device_after_invalid_fallback_candidate(self) -> None:
        devices = [
            {"name": "Speaker", "max_input_channels": 0, "hostapi": 0},
            {"name": "Ghost Mic", "max_input_channels": 1, "hostapi": 0},
            {"name": "Real Mic", "max_input_channels": 1, "hostapi": 0},
        ]
        opened_devices: list[int | None] = []

        def fake_query_devices(device=None, kind=None):  # noqa: ANN001
            if kind == "input":
                raise RuntimeError("Error querying device -1")
            if device is None:
                return devices
            return devices[int(device)]

        class _FakeStream:
            def __init__(self, **kwargs) -> None:  # noqa: ANN003
                device = kwargs.get("device")
                opened_devices.append(device)
                if device == 1:
                    raise RuntimeError("Error opening InputStream: Invalid device [PaErrorCode -9996]")

            def start(self) -> None:
                return None

            def stop(self) -> None:
                return None

            def close(self) -> None:
                return None

        with patch("recorder.sd.query_devices", side_effect=fake_query_devices), patch(
            "recorder.sd.query_hostapis", return_value=[{"name": "WASAPI"}]
        ), patch("recorder.sd.InputStream", side_effect=lambda **kwargs: _FakeStream(**kwargs)), patch(
            "recorder.AudioRecorder._reinitialize_portaudio"
        ) as reinit:
            recorder = AudioRecorder()
            recorder.start()

        self.assertTrue(recorder.is_running)
        self.assertEqual(opened_devices, [1, 2])
        reinit.assert_not_called()

    def test_tray_refresh_status_uses_cached_audio_snapshot(self) -> None:
        audio_provider = Mock(
            return_value=(
                [{"token": "default", "label": "Default microphone", "is_default": True}],
                "default",
            )
        )
        tray = AppTray(
            icon_path="",
            short_status_provider=lambda: "short",
            full_status_provider=lambda: "full",
            ui_language_provider=lambda: "en",
            model_status_provider=lambda: {"medium": True},
            on_model=lambda model: None,
            on_model_install=lambda model: None,
            on_transcription_language=lambda lang: None,
            on_backend=lambda backend: None,
            audio_devices_provider=audio_provider,
            on_audio_device=lambda token: None,
            on_open_settings=lambda: None,
            unload_available_provider=lambda: True,
            unload_never_provider=lambda: False,
            on_unload_model=lambda: None,
            on_toggle_unload_never=lambda: None,
            rescue_copy_available_provider=lambda: False,
            on_copy_last_dictation=lambda: None,
            on_quit=lambda: None,
        )

        self.assertEqual(audio_provider.call_count, 1)
        tray.refresh_status()
        tray.refresh_status()
        self.assertEqual(audio_provider.call_count, 1)
        tray.hide()
        tray.deleteLater()

    def test_settings_cuda_status_refresh_runs_async_and_applies_result(self) -> None:
        dialog = SettingsDialog.__new__(SettingsDialog)
        dialog._cuda_status_provider = Mock(return_value={"text": "CUDA ready", "downloadable": True})
        dialog._on_cuda_download = lambda: True
        dialog._cuda_refresh_generation = 0
        dialog._cuda_refresh_in_flight = False
        dialog.cuda_status_value = _FakeLabel()
        dialog.cuda_download_button = _FakeButton()
        dialog._t = lambda key, **kwargs: "unknown"
        dialog._signals = types.SimpleNamespace(
            cuda_status_ready=_DirectEmit(lambda generation, state: SettingsDialog._apply_cuda_status_result(dialog, generation, state))
        )

        with patch("settings_dialog.threading.Thread", _ImmediateThread):
            SettingsDialog._refresh_cuda_status(dialog)

        self.assertFalse(dialog._cuda_refresh_in_flight)
        self.assertEqual(dialog.cuda_status_value.text, "CUDA ready")
        self.assertTrue(dialog.cuda_download_button.visible)
        self.assertTrue(dialog.cuda_download_button.enabled)


if __name__ == "__main__":
    unittest.main()
