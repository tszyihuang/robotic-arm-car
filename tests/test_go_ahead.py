"""任务表前进到跑道尽头；共用原图、真实端点检测及停车控制，不连接硬件。"""
import contextlib
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, patch

import numpy as np

import main
from base.api import Base
from base.control import MotionCancelled
from base import vision_straight as vs
from config import BASE, VISION, VISION_CONTROL
from debug.dry_run import DryArm, DryBase, DryVision
from tasks.runner import load_section
from vision.boundary import BoundarySample
from vision.camera import CameraStream
from vision.runway_endpoint import EndpointStop


ROOT = Path(__file__).resolve().parents[1]
DETECTOR_CONFIG = {"window": 1, "stable_frames": 2, "jump_frames": 3,
                   "max_jump_px": 40.0, "stable_tolerance_px": 8.0, "smoothing": 1.0}


def shared_camera():
    camera = CameraStream.__new__(CameraStream)
    camera.condition = threading.Condition()
    camera.stop = threading.Event()
    camera.frame, camera.index, camera.capture_stamp = None, 0, 0.0
    camera.error = ""
    return camera


def publish(camera, y, stamp, *, height=400):
    frame = np.full((height, 32, 3), (143, 140, 141), dtype=np.uint8)
    if y is not None:
        frame[:y] = (143, 159, 155)
    with camera.condition:
        camera.frame = frame
        camera.index += 1
        camera.capture_stamp = stamp
        camera.condition.notify_all()


def lane_heading(now):
    def edge(x):
        return {"a": 0.0, "b": x, "near_y": 719.0, "far_y": 100.0,
                "bottom_x": x, "far_x": x, "confidence": 1.0}
    info = {"size": [1280, 720], "left": edge(360), "right": edge(960)}
    vision = Mock()
    vision.sample.return_value = BoundarySample(info, 1, 1, "ready", "", now, now)
    vision.age.return_value = 0
    # 未知距离时即使配置末段 gap，也仍需全程视觉纠偏。
    return vs.VisionHeading(vs.VisionCfg(rate_src="gap", gap_finish_mm=10000), vision)


class EndpointStopTests(unittest.TestCase):
    def setUp(self):
        self.clock = [100.0]
        clock_patch = patch("vision.runway_endpoint.time.time", side_effect=lambda: self.clock[0])
        clock_patch.start()
        self.addCleanup(clock_patch.stop)
        self.camera = shared_camera()
        self.endpoint = EndpointStop(self.camera, config=DETECTOR_CONFIG, log=lambda _: None)

    def sample(self, y):
        self.clock[0] += 0.02
        publish(self.camera, y, self.clock[0])
        return self.endpoint(self.clock[0])

    def test_stops_at_292_after_tracking_below_threshold(self):
        for y in (270, 270, 291):
            self.assertFalse(self.sample(y))
        self.assertTrue(self.sample(292))
        self.assertEqual(self.endpoint.result["point"], (16, 292))

    def test_skipping_the_exact_row_still_stops(self):
        self.assertFalse(self.sample(280))
        self.assertFalse(self.sample(280))
        self.assertTrue(self.sample(300))

    def test_repeated_frame_cannot_complete_initial_confirmation(self):
        self.assertFalse(self.sample(300))
        for _ in range(4):
            self.assertFalse(self.endpoint(self.clock[0]))
        self.assertEqual(self.endpoint.result["pending_count"], 1)
        self.assertTrue(self.sample(300))

    def test_large_jump_requires_consistent_new_frames(self):
        self.sample(200)
        self.sample(200)
        for _ in range(2):
            self.assertFalse(self.sample(320))
            self.assertEqual(self.endpoint.result["point"], (16, 200))
        self.assertTrue(self.sample(320))

    def test_single_jump_and_missing_edge_do_not_complete_command(self):
        for y in (200, 200, 320, None, 200, 320):
            self.assertFalse(self.sample(y))
        self.assertFalse(self.endpoint.result["valid"])
        self.assertEqual(self.endpoint.result["pending_count"], 1)

    def test_stale_image_above_threshold_cannot_stop_and_timeout_is_an_error(self):
        publish(self.camera, 320, self.clock[0] - VISION["frame_stale"] - 0.1)
        self.assertFalse(self.endpoint(self.clock[0]))
        self.clock[0] += VISION["frame_stale"] + 0.01
        with self.assertRaisesRegex(RuntimeError, "没有新鲜画面"):
            self.endpoint(self.clock[0])

    def test_capture_timeout_after_tracking_is_an_error(self):
        self.sample(270)
        self.sample(270)
        self.clock[0] += VISION["frame_stale"] + 0.01
        with self.assertRaisesRegex(RuntimeError, "没有新鲜画面"):
            self.endpoint(self.clock[0])

    def test_camera_errors_and_cancellation_are_propagated(self):
        self.camera.error = "摄像头断开"
        with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
            self.endpoint(self.clock[0])
        stop = threading.Event()
        stop.set()
        endpoint = EndpointStop(self.camera, stop_event=stop)
        with self.assertRaises(MotionCancelled):
            endpoint(self.clock[0])

    def test_unreachable_stop_row_is_rejected(self):
        publish(self.camera, 200, self.clock[0], height=292)
        with self.assertRaisesRegex(ValueError, "超出画面高度"):
            self.endpoint(self.clock[0])


class GoAheadMotionTests(unittest.TestCase):
    def run_motion(self, frames, *, encoder_missing=False, stable_frames=2, stop_y=292):
        clock = [100.0]
        camera = shared_camera()
        board = Mock()
        distance, tick, stopped = [0], [0], [False]
        images = iter(frames)

        def feedback(*args):
            clock[0] += 0.02
            if tick[0] == 0 or stopped[0]:
                tick[0] += 1
                return [distance[0]] * 4, None
            y = next(images)
            if y == "error":
                camera.error = "摄像头断开"
            else:
                publish(camera, y, clock[0])
            distance[0] += 10
            return (None if encoder_missing else [distance[0]] * 4), None

        def speed(*values):
            if values == (0, 0, 0, 0):
                stopped[0] = True

        board.feedback.side_effect = feedback
        board.spd.side_effect = speed
        config = dict(DETECTOR_CONFIG, stable_frames=stable_frames)
        with patch.dict(BASE, meters_per_count=0.001), \
                patch("base.vision_straight.time.time", side_effect=lambda: clock[0]):
            endpoint = EndpointStop(camera, stop_y=stop_y, config=config, log=lambda _: None)
            heading = lane_heading(clock[0])
            try:
                result = vs.vision_straight(board, heading, stop_condition=endpoint, log=lambda _: None)
            except RuntimeError:
                self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))
                self.assertTrue(any(command.args != (0, 0, 0, 0)
                                    for command in board.spd.call_args_list))
                raise
        return board, endpoint, result

    def test_control_keeps_visual_steering_until_confirmed_endpoint_then_stops(self):
        board, endpoint, (distance, info) = self.run_motion([None, 270, 270, 291, 292])
        self.assertFalse(info["reason"])
        self.assertEqual(distance, 50)
        self.assertEqual(info["heading_mode"], "vision")
        self.assertIsNone(info["finish_gap_ref_mm"])
        self.assertEqual(endpoint.result["point"][1], 292)
        commands = board.spd.call_args_list
        self.assertEqual(len(commands), 5)
        for command in commands[:-1]:
            self.assertNotEqual(command.args[0], command.args[2])
        self.assertEqual(commands[-1].args, (0, 0, 0, 0))

    def test_selected_stop_row_keeps_moving_past_292_and_stops_at_300(self):
        board, endpoint, (distance, info) = self.run_motion([270, 270, 292, 299, 300], stop_y=300)
        self.assertFalse(info["reason"])
        self.assertEqual(distance, 50)
        self.assertEqual(endpoint.result["point"][1], 300)
        self.assertEqual(len(board.spd.call_args_list), 5)
        self.assertTrue(all(command.args != (0, 0, 0, 0)
                            for command in board.spd.call_args_list[:-1]))
        self.assertEqual(board.spd.call_args_list[-1], call(0, 0, 0, 0))

    def test_already_confirmed_boundary_stops_before_any_movement(self):
        board, _, (_, info) = self.run_motion([292], stable_frames=1)
        self.assertFalse(info["reason"])
        self.assertEqual(board.spd.call_args_list, [call(0, 0, 0, 0)])

    def test_endpoint_is_checked_even_when_encoder_feedback_is_missing(self):
        board, endpoint, (_, info) = self.run_motion([292, 292], encoder_missing=True)
        self.assertFalse(info["reason"])
        self.assertTrue(endpoint.result["valid"])
        self.assertEqual(board.spd.call_args_list, [call(0, 0, 0, 0)])

    def test_camera_failure_during_motion_stops_and_fails_command(self):
        with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
            self.run_motion([270, 270, "error"])


class GoAheadApiTests(unittest.TestCase):
    def test_api_uses_shared_camera_and_applies_selected_row_and_speed(self):
        base, vision = Base(), Mock()
        base.config["go_ahead_stop_y"] = 310
        base.config["go_ahead_speed"] = 200
        base.motor = Mock()
        heading = lane_heading(100.0)
        heading.wait_track = Mock(return_value=object())
        settings = []

        def control(board, current_heading, *, stop_condition, **kwargs):
            self.assertIs(board, base.motor)
            self.assertIsNone(kwargs.get("goal_mm"))
            self.assertIs(stop_condition.camera, vision.start_camera.return_value)
            self.assertIs(stop_condition.stop_event, base.stop_event)
            self.assertEqual(current_heading.cfg.lateral_offset_mm, VISION_CONTROL["LATERAL_OFFSET_MM"])
            self.assertEqual(current_heading.cfg.lane_width_mm, VISION_CONTROL["LANE_WIDTH_MM"])
            settings.append((stop_condition.stop_y, current_heading.cfg.speed))
            return 850, {"reason": ""}

        with patch.dict(VISION_CONTROL, LOG_DIR=None), \
                patch.object(base, "_heading", return_value=heading), \
                patch.object(vs, "vision_straight", side_effect=control):
            self.assertTrue(base.go_ahead(vision, 300, 600)["ok"])
            self.assertTrue(base.go_ahead(vision, 300)["ok"])
            result = base.go_ahead(vision)
        self.assertEqual(result["distance_m"], 0.85)
        self.assertEqual(settings, [(300, 600), (300, 200), (310, 200)])
        self.assertEqual(base.config["go_ahead_stop_y"], 310)
        self.assertEqual(base.config["go_ahead_speed"], 200)
        self.assertEqual(base.config["speed"], BASE["speed"])

    def test_invalid_speed_is_rejected_before_connecting(self):
        for speed in (0, -1, 1001, float("nan"), float("inf")):
            with self.subTest(speed=speed):
                base = Base()
                with patch.object(base, "_heading") as prepare, self.assertRaisesRegex(ValueError, "速度"):
                    base.go_ahead(Mock(), speed=speed)
                prepare.assert_not_called()

    def test_invalid_stop_row_is_rejected_before_connecting(self):
        for stop_y in (-1, VISION["height"], VISION["height"] + 1, float("nan"), float("inf")):
            with self.subTest(stop_y=stop_y):
                base = Base()
                with patch.object(base, "_heading") as prepare, self.assertRaisesRegex(ValueError, "停车行"):
                    base.go_ahead(Mock(), stop_y)
                prepare.assert_not_called()

    def test_control_failure_does_not_report_success(self):
        base = Base()
        heading = lane_heading(100.0)
        heading.wait_track = Mock(return_value=object())
        with patch.dict(VISION_CONTROL, LOG_DIR=None), \
                patch.object(base, "_heading", return_value=heading), \
                patch.object(vs, "vision_straight", return_value=(20, {"reason": "编码器数据中断"})), \
                self.assertRaisesRegex(RuntimeError, "编码器数据中断"):
            base.go_ahead(Mock())


class GoAheadTaskTests(unittest.TestCase):
    def run_tasks(self, text, base, vision):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[尽头测试]\n" + text, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main.run(base, DryArm(), vision, tasks_path=path, section="尽头测试")
            return output.getvalue()

    def test_task_passes_shared_vision_stop_row_and_speed_then_continues(self):
        base, vision = Mock(dry_run=True), Mock()
        self.run_tasks("go-ahead\ngo-ahead 300\ngo-ahead 300 600\ngo-ahead 0\nstraight 0.3\n", base, vision)
        self.assertEqual(base.go_ahead.call_args_list,
                         [call(vision), call(vision, 300), call(vision, 300, 600), call(vision, 0)])
        base.straight.assert_called_once_with(0.3)
        vision.start.assert_called_once_with()
        vision.close.assert_called_once_with()

    def test_dry_run_prints_threshold_and_speed_without_waiting(self):
        with patch.dict(BASE, go_ahead_stop_y=310, go_ahead_speed=200):
            output = self.run_tasks("go-ahead\ngo-ahead 300\ngo-ahead 300 600\n", DryBase(), DryVision())
        self.assertIn("base.go_ahead(vision, stop_y=310px, speed=200 mm/s)", output)
        self.assertIn("base.go_ahead(vision, stop_y=300px, speed=200 mm/s)", output)
        self.assertIn("base.go_ahead(vision, stop_y=300px, speed=600 mm/s)", output)
        self.assertIn("y ≥ 310px", output)
        self.assertIn("y ≥ 300px", output)

    def test_invalid_task_parameters_report_line_before_moving(self):
        for command in ("go-ahead 300 200 1", "go-ahead -1", f"go-ahead {VISION['height']}",
                        "go-ahead nan", "go-ahead inf", "go-ahead 错误", "go-ahead 300 0",
                        "go-ahead 300 -1", "go-ahead 300 1001", "go-ahead 300 nan", "go-ahead 300 inf"):
            with self.subTest(command=command):
                base = Mock(dry_run=True)
                with self.assertRaisesRegex(ValueError, "tasks.txt:3"):
                    self.run_tasks("straight 0.3\n" + command + "\n", base, Mock())
                base.straight.assert_not_called()

    def test_task_file_provides_independent_go_ahead_section(self):
        steps = load_section(ROOT / "tasks.txt", "跑道尽头")
        self.assertEqual([(step.command, step.args) for step in steps], [("go-ahead", (300,))])
