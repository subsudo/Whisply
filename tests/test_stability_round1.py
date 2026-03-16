import os
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from main import WhisperTypeApp, _is_cuda_fallback_oom
from overlay import OverlayWidget
from transcriber import Transcriber


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


if __name__ == "__main__":
    unittest.main()
