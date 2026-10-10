"""物品校准：形状过滤、中间目标选择、连续运动和任务指令接入。"""
import contextlib
import io
from itertools import permutations
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import main
from base.api import Base
from base.control import MotionCancelled
from base.object_position import calibrate_object_position, validate_config
from config import BALL_POSITION, OBJECT_POSITION
from debug.dry_run import DryBase, DryVision
from tasks.calibrate_object_position import run
from tasks.runner import execute, load_section, Step
from tests.test_ball_position import SimulatedRobot
from tests.test_vision_api import ContinuousCamera
from vision.api import Vision


class SimulatedObjectRobot(SimulatedRobot):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = dict(zip(("red", "green", "blue"), ("cylinder", "cone", "drum")))

    def object_layout(self, sample):
        return {**sample, "candidates": [
            {**row, "value": self.shapes[row["value"]]} for row in sample["candidates"]]}

    def observe_object_layout(self, **kwargs):
        return self.object_layout(self.observe_ball_layout(**kwargs))

    def object_layout_sample(self):
        return self.object_layout(self.ball_layout_sample())

    def wait_object_layout(self, *, after, timeout, stop_event=None):
        self.advance(timeout, stop_event)
        sample = self.object_layout_sample()
        return sample if sample["frame_index"] != after else None

    def run(self, *, log=lambda _: None, **config):
        with patch("base.ball_position.time.monotonic", side_effect=lambda: self.clock), \
                patch("base.ball_position.time.time", side_effect=lambda: self.clock):
            return calibrate_object_position(self, self, config=config,
                                              stop_event=self.stop_event, log=log)


class ObjectPositionTests(unittest.TestCase):
    def test_centers_the_spatial_middle_for_every_shape_order_from_both_sides(self):
        for shapes in permutations(("cylinder", "cone", "drum")):
            for error in (-120, 120):
                with self.subTest(shapes=shapes, error=error):
                    robot = SimulatedObjectRobot(error, deadzone=25, coast_tau=0.4)
                    robot.shapes = dict(zip(("red", "green", "blue"), shapes))
                    result = robot.run(mm_per_px=0.25)
                    self.assertTrue(result["ok"])
                    self.assertEqual(result["shape"], shapes[1])
                    self.assertLessEqual(abs(result["error_px"]), OBJECT_POSITION["tolerance_px"])
                    self.assertLessEqual(abs(result["position_error_mm"]), OBJECT_POSITION["position_tolerance_mm"])
                    moving = [values for _, values in robot.commands if any(values)]
                    self.assertTrue(moving)
                    self.assertTrue(all(value * error > 0 for value in moving[0]))
                    self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_actual_frame_center_already_aligned_does_not_move(self):
        robot = SimulatedObjectRobot(error=0, width=600)
        result = robot.run()
        self.assertEqual(result["center_x"], 300)
        self.assertTrue(all(not any(values) for _, values in robot.commands))

    def test_missing_objects_stop_immediately_and_resume_after_recovery(self):
        robot = SimulatedObjectRobot()
        robot.missing = lambda: 0.2 <= robot.clock - 100 < 0.4
        self.assertTrue(robot.run()["ok"])
        stopped_at = next(stamp for stamp in robot.stops if 0.2 <= stamp - 100 < 0.5)
        self.assertTrue(any(stamp > stopped_at and any(values) for stamp, values in robot.commands))

    def test_changed_middle_shape_missing_frames_and_encoder_loss_stop(self):
        cases = (("shape", RuntimeError, "中间物品身份发生变化"),
                 ("missing", TimeoutError, "持续缺物品"),
                 ("frozen", TimeoutError, "视觉断流"),
                 ("encoder", RuntimeError, "编码器"))
        for case, error, message in cases:
            with self.subTest(case=case):
                robot = SimulatedObjectRobot()
                if case == "shape":
                    robot.color = "red"
                elif case == "missing":
                    robot.missing = lambda: True
                elif case == "frozen":
                    robot.freeze = True
                else:
                    robot.no_totals_after = 0.2
                with self.assertRaisesRegex(error, message):
                    robot.run()
                self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_cancel_and_invalid_geometry_cannot_send_motion(self):
        for case, error in (("cancel", MotionCancelled), ("geometry", ValueError)):
            with self.subTest(case=case):
                robot = SimulatedObjectRobot()
                if case == "cancel":
                    robot.stop_event.set()
                else:
                    robot.latest["size"] = None
                with self.assertRaises(error):
                    robot.run()
                self.assertTrue(all(not any(values) for _, values in robot.commands))

    def test_object_parameters_are_independent_and_validated(self):
        ball_scale = BALL_POSITION["mm_per_px"]
        with patch.dict(OBJECT_POSITION, {"mm_per_px": 0.5, "speed": 60}):
            self.assertEqual(validate_config()["mm_per_px"], 0.5)
            self.assertEqual(BALL_POSITION["mm_per_px"], ball_scale)
            self.assertEqual(validate_config({"speed": 45})["speed"], 45)
        for field in ("mm_per_px", "position_tolerance_mm", "timeout"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "OBJECT_POSITION"):
                validate_config({field: 0})


class ObjectLayoutApiTests(unittest.TestCase):
    def test_new_object_layouts_reuse_background_model_and_ignore_balls(self):
        camera, vision = ContinuousCamera(), Vision()
        threads = []

        def predict(frame):
            threads.append(threading.current_thread().name)
            return {"size": [600, 240], "detections": [
                {"name": name, "box": [x - 20, 10, x + 20, 50]}
                for name, x in (("腰鼓", 450), ("圆柱", 150), ("圆锥", 300), ("红球", 200))]}

        vision._object_model = Mock()
        vision._object_model.predict.side_effect = predict
        try:
            with patch("vision.api.CameraStream", return_value=camera) as factory:
                first = vision.observe_object_layout(timeout=1)
                second = vision.observe_object_layout(timeout=1)
            factory.assert_called_once()
            self.assertEqual(first["size"], [600, 240])
            self.assertEqual([row["value"] for row in first["candidates"]], ["cylinder", "cone", "drum"])
            self.assertEqual([row["center_x"] for row in first["candidates"]], [150, 300, 450])
            self.assertGreater(second["frame_index"], first["frame_index"])
            self.assertGreater(second["capture_stamp"], first["capture_stamp"])
            self.assertEqual(set(threads), {"vision-inference"})
        finally:
            vision.close()

    def test_partial_object_frame_is_available_for_immediate_stop_and_not_repeated(self):
        clock = [100.1]
        vision = Vision(clock=lambda: clock[0])
        vision._enabled["objects"] = True
        vision._objects = ({"size": [600, 240], "detections": [
            {"name": "圆柱", "box": [10, 20, 30, 40]},
            {"name": "圆锥", "box": [50, 20, 70, 40]},
            {"name": "红球", "box": [90, 20, 110, 40]}]}, 9, 100.0)
        try:
            sample = vision.wait_object_layout(after=8, timeout=0)
            self.assertEqual(len(sample["candidates"]), 2)
            self.assertEqual(sample["size"], [600, 240])
            self.assertIsNone(vision.wait_object_layout(after=9, timeout=0))
            clock[0] = 100.7
            self.assertIsNone(vision.object_layout_sample())
            clock[0] = 100.1
            vision._enabled["objects"] = False
            self.assertIsNone(vision.object_layout_sample())
            vision._enabled["objects"] = True
            vision._scanning.set()
            self.assertIsNone(vision.object_layout_sample())
        finally:
            vision.close()


class ObjectPositionTaskTests(unittest.TestCase):
    def test_task_syntax_and_dry_run_without_connecting_devices(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[物品校准]\ncalibrate-object-position # 对齐中间物品\n", encoding="utf-8")
            steps = load_section(path, "物品校准")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertIsNone(execute(steps[0], DryBase(), None, DryVision()))
            self.assertIn("中间物品偏左 → 后退，偏右 → 前进", output.getvalue())
            self.assertIn("编码器目标位置 → 位置 PID → 轮速 PI", output.getvalue())
            path.write_text("[物品校准]\ncalibrate-object-position 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "需要 0 个参数"):
                load_section(path, "物品校准")

    def test_task_and_base_share_devices_parameters_and_cancellation(self):
        base, vision = Base(), Mock()
        base.motor, base._imu_checked = Mock(), True
        with patch("base.api.object_position.calibrate_object_position", return_value={"ok": True}) as calibrate:
            self.assertEqual(run(base, vision, stop_event=base.stop_event, log=None), {"ok": True})
        calibrate.assert_called_once_with(base.motor, vision, config=OBJECT_POSITION,
                                          log=None, stop_event=base.stop_event)
        base.motor.stop.assert_called()

    def test_selected_object_section_starts_vision_and_runs_before_home(self):
        base, arm, vision = Mock(), Mock(), Mock()
        stop = threading.Event()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[主线]\nstraight 1\n[物品校准]\ncalibrate-object-position\nhome\n", encoding="utf-8")
            with patch("tasks.runner.calibrate_object_position.run", return_value={"ok": True}) as calibrate, \
                    patch("tasks.runner.pause"), contextlib.redirect_stdout(io.StringIO()):
                main.run(base, arm, vision, tasks_path=path, section="物品校准", stop_event=stop)
        vision.start.assert_called_once()
        calibrate.assert_called_once_with(base, vision, stop_event=stop)
        arm.home.assert_called_once()
        base.straight.assert_not_called()
        base.stop.assert_called_once()
        vision.close.assert_called_once()
