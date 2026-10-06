"""摄像头、模型复用和后台推理的直接调用验证，不打开真实摄像头。"""
import contextlib
import io
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from base.control import MotionCancelled
from vision.api import Vision


class VisionApiTests(unittest.TestCase):
    def test_preview_waits_for_both_models_and_uses_their_source_frame(self):
        vision = Vision()
        source = np.full((80, 100, 3), 30, dtype=np.uint8)
        latest = np.full_like(source, 220)
        camera = Mock(error='', frame=latest, capture_stamp=time.time(),
                      condition=threading.Condition())
        entered, release, consumer_started, delivered = [threading.Event() for _ in range(4)]
        stamp = time.time()
        geometry = {'left': {'points': [[10, 70], [30, 10]]}, 'right': None,
                    'timing_ms': {'total': 1}}
        objects = {'detections': [{'box': [50, 30, 80, 60], 'class_id': 0}]}

        def capture(**kwargs):
            if camera.next_frame.call_count == 1:
                return source, 1, stamp
            vision._inference_stop.wait(2)
            raise MotionCancelled('结束测试')

        def detect(frame):
            self.assertIs(frame, source)
            entered.set()
            if not release.wait(2):
                raise RuntimeError('测试推理未释放')
            return objects

        def snapshot():
            consumer_started.set()
            try:
                results.append(vision.snapshot(after_stamp=0.0))
            except Exception as exc:
                results.append(exc)
            finally:
                delivered.set()

        camera.next_frame.side_effect = capture
        vision._boundary_model = Mock()
        vision._boundary_model.predict.return_value = geometry
        vision._object_model = Mock()
        vision._object_model.predict.side_effect = detect
        results = []
        consumer = threading.Thread(target=snapshot)
        try:
            with patch('vision.api.CameraStream', return_value=camera):
                vision.set_models(boundary=True, objects=True)
            self.assertTrue(entered.wait(1))
            consumer.start()
            self.assertTrue(consumer_started.wait(1))
            self.assertFalse(delivered.wait(0.05), '物体模型完成前不能发布画面')
            self.assertIsNone(vision._preview)
            release.set()
            self.assertTrue(delivered.wait(1))
            self.assertIsInstance(results[0], tuple)
            image, returned_stamp = results[0]
            decoded = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(returned_stamp, stamp)
            self.assertLess(int(decoded[75, 95].mean()), 50, '不能用较新的原图配旧结果')
            self.assertGreater(int(decoded[40, 50, 1]), 150, '应绘制本帧检测框')
            self.assertGreater(int(decoded[70, 10, 1]), 150, '应绘制本帧边界')
            self.assertTrue(np.all(source == 30), '绘图不能修改推理原图')
        finally:
            release.set()
            vision.close()
            if consumer.ident is not None:
                consumer.join(timeout=2)

    def test_stream_waits_for_next_inference_and_skips_older_completed_frames(self):
        vision = Vision()
        vision.camera = Mock(error='', frame=np.full((20, 20, 3), 240, np.uint8),
                             capture_stamp=103.0, condition=threading.Condition())
        vision._enabled = {'boundary': False, 'objects': True}
        source = np.full((20, 20, 3), 30, np.uint8)
        vision._preview = (source, 100.0, None, {'detections': []})
        started, delivered = threading.Event(), threading.Event()
        results = []

        def snapshot():
            started.set()
            results.append(vision.snapshot(after_stamp=100.0))
            delivered.set()

        consumer = threading.Thread(target=snapshot)
        try:
            consumer.start()
            self.assertTrue(started.wait(1))
            self.assertFalse(delivered.wait(0.05), '仅有新采集帧时不能更新')
            with vision.condition:
                vision._preview = (source, 101.0, None, {'detections': []})
                vision._preview = (source, 102.0, None, {'detections': []})
                vision.condition.notify_all()
            self.assertTrue(delivered.wait(1))
            self.assertEqual(results[0][1], 102.0)
            vision.camera.next_frame.assert_not_called()
        finally:
            vision.close()
            consumer.join(timeout=2)

    def test_turning_models_off_wakes_preview_and_returns_unannotated_live_frame(self):
        vision = Vision()
        raw = np.full((40, 60, 3), 120, np.uint8)
        vision.camera = Mock(error='', frame=raw, capture_stamp=time.time(),
                             condition=threading.Condition())
        vision._enabled = {'boundary': False, 'objects': True}
        vision._preview = (raw, 1.0, None, {'detections': []})
        vision._thread = Mock()
        vision.camera.next_frame.return_value = (raw, 2, 2.0)
        started = threading.Event()
        results = []

        def snapshot():
            started.set()
            results.append(vision.snapshot(after_stamp=1.0))

        consumer = threading.Thread(target=snapshot)
        try:
            consumer.start()
            self.assertTrue(started.wait(1))
            vision.set_models(boundary=False, objects=False)
            consumer.join(timeout=1)
            self.assertFalse(consumer.is_alive())
            self.assertIsNone(vision._preview)
            self.assertEqual(results[0][1], 2.0)
            decoded = cv2.imdecode(np.frombuffer(results[0][0], np.uint8), cv2.IMREAD_COLOR)
            self.assertTrue(np.all(decoded == 120))
        finally:
            vision.close()
            consumer.join(timeout=2)

    def test_inference_result_is_discarded_if_models_change_mid_frame(self):
        vision = Vision()
        vision.camera = Mock(error='')
        vision.camera.next_frame.side_effect = [(object(), 1, time.time()), MotionCancelled('结束测试')]
        vision._enabled = {'boundary': False, 'objects': True}

        def detect(frame):
            with vision.condition:
                vision._model_generation += 1
                vision._enabled = {'boundary': False, 'objects': False}
            vision._inference_stop.set()
            return {'detections': []}

        vision._object_model = Mock()
        vision._object_model.predict.side_effect = detect
        vision._infer()
        self.assertIsNone(vision._preview)
        self.assertIsNone(vision._objects)
        vision.close()

    def test_scan_uses_1080p_and_restores_720p_on_success_error_or_cancel(self):
        for error in (None, TimeoutError('扫码超时'), OSError('断开'),
                      MotionCancelled('取消'), KeyboardInterrupt()):
            with self.subTest(error=error):
                vision = Vision()
                camera = Mock()
                with patch('vision.api.CameraStream', return_value=camera), \
                        patch('vision.api.scan_qrcode', return_value='211', side_effect=error) as scan:
                    if error is None:
                        self.assertEqual(vision.scan_qrcode(), '211')
                    else:
                        with self.assertRaises(type(error)) as failed:
                            vision.scan_qrcode()
                        self.assertIs(failed.exception, error)
                self.assertEqual([call.args for call in camera.set_resolution.call_args_list],
                                 [(1920, 1080), (1280, 720)])
                self.assertFalse(vision._scanning.is_set())
                self.assertIsNone(vision.sample().info)
                vision.close()

    def test_unsupported_scan_resolution_restores_camera_without_decoding(self):
        vision = Vision()
        camera = Mock()
        camera.set_resolution.side_effect = [RuntimeError('不支持 1080p'), None]
        with patch('vision.api.CameraStream', return_value=camera), \
                patch('vision.api.scan_qrcode') as scan, self.assertRaisesRegex(RuntimeError, '1080p'):
            vision.scan_qrcode()
        scan.assert_not_called()
        self.assertEqual(camera.set_resolution.call_args.args, (1280, 720))
        self.assertFalse(vision._scanning.is_set())
        vision.close()

    def test_resolution_restore_failure_keeps_original_scan_error(self):
        vision = Vision()
        camera = Mock()
        original = TimeoutError('扫码超时')
        camera.set_resolution.side_effect = [None, OSError('恢复失败')]
        with patch('vision.api.CameraStream', return_value=camera), \
                patch('vision.api.scan_qrcode', side_effect=original), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(TimeoutError) as failed:
            vision.scan_qrcode()
        self.assertIs(failed.exception, original)
        self.assertFalse(vision._scanning.is_set())
        vision.close()

    def test_inference_discards_frames_captured_before_resolution_switch(self):
        vision = Vision(clock=lambda: 21.0)
        vision._frame_cutoff = 20.0
        camera = Mock()
        old, new = object(), object()
        camera.next_frame.side_effect = [(old, 1, 19.0), (new, 2, 21.0), OSError('结束采集')]
        vision._boundary_model = Mock()
        vision._boundary_model.predict.return_value = {
            'size': [1280, 720], 'left': {'a': -0.3, 'b': 496},
            'right': {'a': 0.3, 'b': 784}, 'timing_ms': {'total': 1},
        }
        with patch('vision.api.CameraStream', return_value=camera):
            vision.set_models(boundary=True, objects=False)
        vision._thread.join(timeout=1)
        self.assertFalse(vision._thread.is_alive())
        vision._boundary_model.predict.assert_called_once_with(new)
        self.assertEqual(vision.sample().frames, 2)
        vision.close()

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
