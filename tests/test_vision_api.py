"""摄像头、模型复用和后台推理的直接调用验证，不打开真实摄像头。"""
import contextlib
import io
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from base.control import MotionCancelled
from vision.api import Vision


class ContinuousCamera:
    """持续提供新帧；关闭能唤醒后台，不接触实际摄像头。"""

    def __init__(self):
        self.condition = threading.Condition()
        self.index = 0
        self.error = ''
        self.stopped = threading.Event()
        self.frame = np.zeros((240, 600, 3), np.uint8)
        self.frame[60:180, 30:150] = (0, 255, 0)
        self.frame[60:180, 230:350] = (0, 0, 255)
        self.frame[60:180, 430:550] = (255, 0, 0)
        self.capture_stamp = time.time()
        self.set_resolution = Mock()

    def next_frame(self, *, stop_event=None, **kwargs):
        if self.stopped.wait(0.01) or (stop_event is not None and stop_event.is_set()):
            raise MotionCancelled('结束采集')
        with self.condition:
            self.index += 1
            self.capture_stamp = time.time()
            return self.frame, self.index, self.capture_stamp

    def close(self):
        self.stopped.set()


class VisionApiTests(unittest.TestCase):
    def test_start_loads_once_in_background_and_repeated_calls_keep_models_and_thread(self):
        camera = ContinuousCamera()
        vision = Vision()
        entered, release = threading.Event(), threading.Event()
        thread_names = []
        boundary, objects = Mock(), Mock()
        boundary.predict.return_value = {
            'size': [1280, 720], 'left': {'a': -0.3, 'b': 496},
            'right': {'a': 0.3, 'b': 784}, 'timing_ms': {'total': 1}}
        objects.predict.return_value = {'detections': [
            {'name': name, 'box': [i * 100, 10, i * 100 + 40, 50]}
            for names in [('红球', '绿球', '蓝球'), ('圆柱', '圆锥', '腰鼓')]
            for i, name in enumerate(names)]}

        def load_boundary(*args, **kwargs):
            thread_names.append(threading.current_thread().name)
            entered.set()
            if not release.wait(2):
                raise RuntimeError('测试模型加载未释放')
            return boundary

        boundary_factory = Mock(side_effect=load_boundary)
        object_factory = Mock(return_value=objects)
        modules = {'vision.yolo_boundary': SimpleNamespace(BoundaryPredictor=boundary_factory),
                   'vision.yolo_objects': SimpleNamespace(ObjectPredictor=object_factory)}
        try:
            with patch.dict('sys.modules', modules), \
                    patch('vision.api.CameraStream', return_value=camera) as camera_factory, \
                    patch('vision.api.scan_qrcode', return_value='211'):
                vision.start()
                self.assertTrue(entered.wait(1), '加载应在独立线程开始')
                self.assertEqual(thread_names, ['vision-inference'])
                self.assertIsNone(vision._object_model)
                worker = vision._thread
                generation = vision._model_generation
                vision.start()
                self.assertEqual(vision._model_generation, generation)
                release.set()
                self.assertTrue(vision.wait_ready(1, log=lambda _: None))
                vision.start()
                vision.enable_boundary()
                self.assertEqual(vision.observe_target('ball', 'green'), 'middle')
                self.assertEqual(vision.observe_target('object', 'drum'), 'right')
                vision.set_models(boundary=False, objects=False)
                vision.start()
                self.assertTrue(vision.wait_ready(1, log=lambda _: None))
                self.assertEqual(vision.scan_qrcode(), '211')
                self.assertTrue(vision.wait_ready(1, log=lambda _: None))
                self.assertIs(vision._thread, worker)
                self.assertIs(vision._boundary_model, boundary)
                self.assertIs(vision._object_model, objects)
                camera_factory.assert_called_once()
                boundary_factory.assert_called_once()
                object_factory.assert_called_once()
        finally:
            release.set()
            vision.close()
        self.assertFalse(worker.is_alive())

    def test_color_target_detection_runs_in_background_without_loading_yolo(self):
        camera, vision = ContinuousCamera(), Vision()
        from vision.targets import colored_targets
        threads = []

        def detect(frame, min_area_ratio):
            threads.append(threading.current_thread().name)
            return colored_targets(frame, min_area_ratio)

        try:
            with patch('vision.api.CameraStream', return_value=camera), \
                    patch('vision.api.colored_targets', side_effect=detect):
                self.assertEqual(vision.observe_target('target', 'red'), 'middle')
                self.assertEqual(vision.observe_target('target', 'blue'), 'right')
                self.assertEqual(set(threads), {'vision-inference'})
                self.assertIsNone(vision._boundary_model)
                self.assertIsNone(vision._object_model)
                self.assertFalse(vision._target_requested.is_set())
        finally:
            vision.close()

    def test_model_load_failure_reaches_waiters_and_survives_scan_resolution_changes(self):
        camera, vision = ContinuousCamera(), Vision()
        factory = Mock(side_effect=OSError('权重加载失败'))
        try:
            with patch.dict('sys.modules', {'vision.yolo_objects': SimpleNamespace(ObjectPredictor=factory)}), \
                    patch('vision.api.CameraStream', return_value=camera):
                with self.assertRaisesRegex(RuntimeError, '权重加载失败'):
                    vision.observe_target('ball', 'green')
                vision._thread.join(timeout=1)
                self.assertEqual(vision.status()['state'], 'error')
                vision._set_resolution(camera, 1280, 720)
                self.assertEqual(vision.status()['error'], '权重加载失败')
                self.assertFalse(vision.wait_ready(0, log=lambda _: None))
                factory.assert_called_once()
        finally:
            vision.close()

    def test_close_during_camera_startup_joins_without_holding_connection_lock(self):
        camera, vision = ContinuousCamera(), Vision()
        entered, release, done = [threading.Event() for _ in range(3)]

        def open_camera(*args):
            entered.set()
            if not release.wait(2):
                raise RuntimeError('测试摄像头启动未释放')
            return camera

        def close():
            try:
                vision.close()
            finally:
                done.set()

        closer = threading.Thread(target=close, daemon=True)
        with patch('vision.api.CameraStream', side_effect=open_camera):
            vision.set_models(boundary=False, objects=False)
            try:
                self.assertTrue(entered.wait(1))
                closer.start()
                release.set()
                self.assertTrue(done.wait(1), '关闭不能与后台启动互相等待')
                self.assertFalse(vision._thread.is_alive())
                self.assertTrue(camera.stopped.is_set())
            finally:
                release.set()
                if closer.ident is not None:
                    closer.join(timeout=1)
                vision.close()

    def test_cached_observation_rejects_duplicate_old_and_stale_frames_and_wakes_on_close(self):
        vision = Vision()
        vision.camera = Mock(error='')
        detections = {'detections': []}
        try:
            now = time.time()
            for index, stamp in [(5, now), (6, now - 2), (6, now - 0.1)]:
                with self.subTest(index=index, stamp=stamp):
                    vision._objects = (detections, index, stamp)
                    with self.assertRaises(TimeoutError):
                        vision._wait_candidates('ball', after=5, after_stamp=now - 0.05, timeout=0.01)
            vision._objects = (detections, 6, now)
            self.assertEqual(vision._wait_candidates('ball', after=5, after_stamp=now - 0.05, timeout=0.1),
                             ([], 6, now))
            delivered, errors = threading.Event(), []

            def wait():
                try:
                    vision._wait_candidates('ball', after=6, after_stamp=now, timeout=10)
                except Exception as exc:
                    errors.append(exc)
                finally:
                    delivered.set()

            waiter = threading.Thread(target=wait)
            waiter.start()
            vision.close()
            self.assertTrue(delivered.wait(1))
            waiter.join(timeout=1)
            self.assertIsInstance(errors[0], MotionCancelled)
        finally:
            vision.close()

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
        camera = ContinuousCamera()
        predictor = Mock()
        thread_names = []
        detections = {"detections": [
            {"name": name, "box": [i * 100, 10, i * 100 + 40, 50]}
            for i, name in enumerate(("红球", "绿球", "蓝球"))]}

        def predict(frame):
            thread_names.append(threading.current_thread().name)
            return detections

        predictor.predict.side_effect = predict
        vision = Vision()
        vision._object_model = predictor
        try:
            with patch("vision.api.CameraStream", return_value=camera) as factory, \
                    patch("vision.api.scan_qrcode", return_value="211") as scan:
                self.assertEqual(vision.scan_qrcode(), "211")
                self.assertEqual(vision.observe_target("ball", "green"), "middle")
                self.assertEqual(vision.observe_target("ball", "blue"), "right")
                self.assertEqual(factory.call_count, 1)
                self.assertIs(scan.call_args.args[0], camera)
                self.assertGreaterEqual(predictor.predict.call_count, 6)
                self.assertEqual(set(thread_names), {"vision-inference"})
        finally:
            vision.close()
        self.assertTrue(camera.stopped.is_set())
        self.assertFalse(vision._thread.is_alive())

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
