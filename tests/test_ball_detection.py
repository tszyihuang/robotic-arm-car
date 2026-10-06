"""小球排序、稳定新帧、后台推理复用及主线终端输出。"""
import contextlib
import io
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from debug.dry_run import DryVision
from tasks import detect_ball
from tasks.runner import Step, execute, load_main
from config import ROOT
from tests.test_vision_api import ContinuousCamera
from vision.api import Vision
from vision.targets import observe


def detections(names):
    rows = [{"name": name, "box": [i * 100, 10, i * 100 + 40, 50]}
            for i, name in enumerate(names)]
    # 模型结果顺序不保证与画面左右顺序一致，并包含非小球类别。
    return list(reversed(rows)) + [{"name": "圆柱", "box": [5, 5, 25, 25]}]


class BallDetectionTests(unittest.TestCase):
    def test_debug_stream_returns_each_inference_without_a_default_rate_limit(self):
        for args in ([], ["--hz", "2"]):
            with self.subTest(args=args):
                predictor = Mock(names={0: "红球"}, threshold=0.25)
                camera = Mock(device=0, resolution=(1280, 720))
                frames = [object(), object()]
                camera.next_frame.side_effect = [(frame, i, time.time())
                                                  for i, frame in enumerate(frames, 1)] + [KeyboardInterrupt()]
                order = []

                def predict(frame):
                    order.append(('infer', frame))
                    return {'frame': frame}

                def publish(info, **kwargs):
                    order.append(('publish', info['frame']))

                predictor.predict.side_effect = predict
                with patch('vision.yolo_objects.ObjectPredictor', return_value=predictor), \
                        patch('vision.camera.CameraStream', return_value=camera), \
                        patch('tasks.detect_ball.print_debug', side_effect=publish), \
                        patch('tasks.detect_ball.time.sleep') as sleep, \
                        contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(detect_ball.main(args), 0)
                self.assertEqual(order, [('infer', frames[0]), ('publish', frames[0]),
                                         ('infer', frames[1]), ('publish', frames[1])])
                self.assertEqual(sleep.call_count, 2 if args else 0)
                camera.close.assert_called_once()

    def test_debug_prints_partial_raw_detections_without_assigning_positions(self):
        info = {"detections": [{"name": "绿球", "confidence": 0.9, "box": [10, 20, 50, 80]},
                               {"name": "圆柱", "confidence": 0.8, "box": [70, 20, 110, 80]}],
                "timing_ms": {"total": 32}}
        with contextlib.redirect_stdout(io.StringIO()) as output:
            detect_ball.print_debug(info, frame_index=7)
        text = output.getvalue()
        for expected in ("小球数量：1", "YOLO：绿球", "YOLO：圆柱", "置信度 0.900",
                         "中心 (30.0, 50.0)", "位置待定"):
            self.assertIn(expected, text)
        self.assertNotIn("小球位置：", text)

    def test_debug_once_sorts_balls_and_closes_camera_on_success_or_failure(self):
        for error in (None, RuntimeError("推理失败"), KeyboardInterrupt()):
            with self.subTest(error=error):
                predictor = Mock(names={0: "红球", 1: "绿球", 2: "蓝球"}, threshold=0.25)
                info = {"detections": detections(("绿球", "蓝球", "红球")),
                        "timing_ms": {"total": 32}}
                for row in info["detections"]:
                    row["confidence"] = 0.9
                predictor.predict.return_value = info
                predictor.predict.side_effect = error
                camera = Mock(device=0, resolution=(1280, 720))
                camera.next_frame.return_value = (object(), 1, time.time())
                with patch("vision.yolo_objects.ObjectPredictor", return_value=predictor), \
                        patch("vision.camera.CameraStream", return_value=camera), \
                        contextlib.redirect_stdout(io.StringIO()) as output, \
                        contextlib.redirect_stderr(io.StringIO()) as errors:
                    result = detect_ball.main(["--once"])
                camera.close.assert_called_once()
                if error is None:
                    self.assertEqual(result, 0)
                    for line in ("小球位置：左，颜色：绿色", "小球位置：中，颜色：蓝色",
                                 "小球位置：右，颜色：红色"):
                        self.assertIn(line, output.getvalue())
                elif isinstance(error, KeyboardInterrupt):
                    self.assertEqual(result, 0)
                    self.assertIn("检测调试已结束", output.getvalue())
                else:
                    self.assertEqual(result, 1)
                    self.assertIn("推理失败", errors.getvalue())

    def test_layout_waits_for_complete_separated_and_stable_new_frames(self):
        layouts = [("红球", "绿球"), ("红球", "绿球", "蓝球"),
                   ("绿球", "红球", "蓝球"), ("绿球", "红球", "蓝球"),
                   ("red_ball", "green_ball", "blue_ball"),
                   ("红球", "绿球", "蓝球"), ("红球", "绿球", "蓝球")]
        outputs = [{"detections": detections(names)} for names in layouts]
        # 完整但重叠的帧也不能计入稳定次数。
        outputs[1]["detections"][0]["box"] = outputs[1]["detections"][1]["box"]
        camera = Mock(condition=threading.Condition(), index=10)
        camera.next_frame.side_effect = [(object(), i, time.time()) for i in range(11, 18)]
        predictor = Mock()
        predictor.predict.side_effect = outputs
        result = observe(camera, "ball", None, predictor)
        self.assertEqual(result["frame_index"], 17)
        self.assertEqual([row["value"] for row in result["candidates"]], ["red", "green", "blue"])
        self.assertEqual([call.kwargs["after"] for call in camera.next_frame.call_args_list],
                         list(range(10, 17)))

    def test_task_reads_existing_background_model_and_prints_colors(self):
        camera, vision = ContinuousCamera(), Vision()
        objects = Mock()
        objects.predict.return_value = {"detections": detections(("绿球", "蓝球", "红球"))}
        inference_threads = []
        objects.predict.side_effect = lambda frame: (
            inference_threads.append(threading.current_thread().name) or objects.predict.return_value)
        factory = Mock(return_value=objects)
        try:
            with patch.dict("sys.modules", {"vision.yolo_objects": SimpleNamespace(ObjectPredictor=factory)}), \
                    patch("vision.api.CameraStream", return_value=camera), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                vision.set_models(boundary=False, objects=True)
                result = execute(Step(1, "detect-balls", ()), None, None, vision)
                self.assertEqual(result, [{"position": "left", "color": "green"},
                                          {"position": "middle", "color": "blue"},
                                          {"position": "right", "color": "red"}])
                self.assertEqual(output.getvalue().splitlines(), ["小球位置：左，颜色：绿色",
                                                                 "小球位置：中，颜色：蓝色",
                                                                 "小球位置：右，颜色：红色"])
                factory.assert_called_once()
                self.assertTrue(inference_threads)
                self.assertEqual(set(inference_threads), {"vision-inference"})
        finally:
            vision.close()

    def test_missing_balls_timeout_without_printing_positions(self):
        camera, vision = ContinuousCamera(), Vision()
        vision.config["observe_timeout"] = 0.05
        objects = Mock()
        objects.predict.return_value = {"detections": detections(("红球", "绿球"))}
        try:
            with patch.dict("sys.modules", {"vision.yolo_objects": SimpleNamespace(
                    ObjectPredictor=Mock(return_value=objects))}), \
                    patch("vision.api.CameraStream", return_value=camera), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(TimeoutError) as failed:
                    detect_ball.run(vision)
                self.assertEqual(output.getvalue(), "")
                self.assertIn("需要完整看见三个候选，当前识别到 2 个", str(failed.exception))
        finally:
            vision.close()

    def test_stale_result_timeout_reports_age_and_actual_yolo_classes(self):
        vision = Vision(clock=lambda: 100.0)
        vision.camera = SimpleNamespace(error="")
        vision._objects = ({"detections": detections(("圆锥", "圆柱", "腰鼓"))}, 5, 98.0)
        with self.assertRaises(TimeoutError) as failed:
            vision._wait_candidates("ball", after=1, after_stamp=90.0, timeout=0)
        message = str(failed.exception)
        self.assertIn("最新帧距采集 2.000s", message)
        self.assertIn("新鲜度上限 0.6s", message)
        for name in ("圆锥", "圆柱", "腰鼓"):
            self.assertIn(name, message)

    def test_actual_main_has_detection_and_dry_run_does_not_invent_results(self):
        self.assertIn("detect-balls", [step.command for step in load_main(ROOT / "tasks.txt")])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = execute(Step(1, "detect-balls", ()), None, None, DryVision())
        self.assertIsNone(result)
        self.assertIn("vision.observe_balls()", output.getvalue())
        self.assertNotIn("小球位置：", output.getvalue())
