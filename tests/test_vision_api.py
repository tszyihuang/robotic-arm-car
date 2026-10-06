"""摄像头、模型复用和后台推理的直接调用验证，不打开真实摄像头。"""
import contextlib
import io
import threading
import time
import unittest
from unittest.mock import Mock, patch

from vision.api import Vision


class VisionApiTests(unittest.TestCase):
    def test_scan_and_observation_share_camera_and_object_model(self):
        camera = Mock(condition=threading.Condition(), index=10)
        camera.next_frame.side_effect = [(object(), i, time.time()) for i in range(11, 17)]
        predictor = Mock()
        predictor.predict.return_value = {"detections": [
            {"name": name, "box": [i * 100, 10, i * 100 + 40, 50]}
            for i, name in enumerate(("红球", "绿球", "蓝球"))]}
        vision = Vision()
        vision._object_model = predictor
        with patch("vision.api.CameraStream", return_value=camera) as factory, \
                patch("vision.api.scan_qrcode", return_value="211") as scan:
            self.assertEqual(vision.scan_qrcode(), "211")
            self.assertEqual(vision.observe_target("ball", "green"), "middle")
            self.assertEqual(vision.observe_target("ball", "blue"), "right")
            self.assertEqual(factory.call_count, 1)
            self.assertIs(scan.call_args.args[0], camera)
            self.assertEqual(predictor.predict.call_count, 6)
        vision.close()
        camera.close.assert_called_once()

    def test_background_boundary_is_direct_and_camera_failure_reaches_controller(self):
        camera = Mock()
        now = time.time()
        camera.next_frame.side_effect = [(object(), 1, now), OSError("摄像头断开")]
        geometry = {"size": [1280, 720], "left": {"a": -0.3, "b": 496},
                    "right": {"a": 0.3, "b": 784}, "timing_ms": {"total": 1}}
        vision = Vision()
        vision._boundary_model = Mock()
        vision._boundary_model.predict.return_value = geometry
        with patch("vision.api.CameraStream", return_value=camera):
            vision.set_models(boundary=True, objects=False)
        vision._thread.join(timeout=1)
        self.assertFalse(vision._thread.is_alive())
        self.assertEqual(vision.sample().seq, 1)
        self.assertIs(vision.sample().info, geometry)
        self.assertEqual(vision.sample().state, "error")
        self.assertIn("摄像头断开", vision.sample().error)
        self.assertFalse(vision.wait_ready(0, log=lambda _: None))
        vision.close()

    def test_model_switch_off_keeps_camera_and_discards_boundary(self):
        camera = Mock()
        vision = Vision()
        vision.camera = camera
        with patch("vision.api.threading.Thread"):
            vision.set_models(boundary=False, objects=False)
        self.assertIs(vision.camera, camera)
        camera.close.assert_not_called()
        self.assertEqual(vision.sample().state, "off")
        vision.close()
        camera.close.assert_called_once()

    def test_camera_close_failure_does_not_skip_inference_thread_cleanup(self):
        vision = Vision()
        vision.camera, vision._thread = Mock(), Mock()
        original = OSError("摄像头释放失败")
        vision.camera.close.side_effect = original
        with contextlib.redirect_stderr(io.StringIO()) as output, self.assertRaises(OSError) as failed:
            vision.close()
        self.assertIs(failed.exception, original)
        vision._thread.join.assert_called_once()
        self.assertIn("摄像头释放失败", output.getvalue())
