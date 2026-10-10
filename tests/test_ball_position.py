"""小球编码器位置 PID：视觉目标、延迟补偿、连续运动与异常停车。"""
import contextlib
import io
from pathlib import Path
import re
import threading
import time
import unittest
from unittest.mock import Mock, patch

from base.api import Base
from base.ball_position import calibrate_ball_position, PositionPID, validate_config
from base.control import MotionCancelled
from config import BASE, BALL_POSITION
from debug.dry_run import DryBase, DryVision
from tasks.calibrate_ball_position import run
from tasks.runner import execute, load_section, Step
from tests.test_vision_api import ContinuousCamera
from vision.api import Vision

ROOT = Path(__file__).resolve().parents[1]


def layout(x, width=640, color="green", index=0, stamp=100.0):
    return {"size": [width, 480], "frame_index": index, "capture_stamp": stamp, "candidates": [
        {"center_x": width - 10, "value": "blue"},
        {"center_x": 10, "value": "red"},
        {"center_x": x, "value": color}]}


class SimulatedRobot:
    """10 Hz 延迟视觉、100 Hz 编码器及有惯性的电机，驱动真实控制函数。"""

    def __init__(self, error=120, width=640, deadzone=0, coast_tau=0.06):
        self.clock = 100.0
        self.error0, self.width = error, width
        self.deadzone, self.coast_tau = deadzone, coast_tau
        self.command = [0.0, 0.0]
        self.velocity = [0.0, 0.0]
        self.position = [0.0, 0.0]
        self.last_counts = [0, 0]
        self.commands = []
        self.stops = []
        self.brakes = []
        self.next_capture = 100.1
        self.index = 0
        self.pending = []
        self.latest = layout(width / 2 + error, width, stamp=self.clock)
        self.freeze = False
        self.missing = lambda: False
        self.color = "green"
        self.no_totals_after = self.no_steps_after = float("inf")
        self.steps_ready = 0.0
        self.observe_count = self.sample_count = 0
        self.stop_event = threading.Event()

    def spd(self, *values):
        self.command = [values[0], values[2]]
        self.commands.append((self.clock, values))

    def stop(self):
        self.stops.append(self.clock)
        self.spd(0, 0, 0, 0)

    def brake(self, speeds):
        self.brakes.append((self.clock, speeds))
        # 反向力矩比零速停车更快；下一次控制拍继续根据实测速制动。
        self.velocity = [value * 0.5 for value in self.velocity]
        self.spd(0, 0, 0, 0)

    def feedback(self, timeout=0):
        counts = [round(value / (BASE["meters_per_count"] * 1000)) for value in self.position]
        increments = [value - previous for value, previous in zip(counts, self.last_counts)]
        self.last_counts = counts
        totals = [counts[0]] * 2 + [counts[1]] * 2
        steps = [increments[0]] * 2 + [increments[1]] * 2
        elapsed = self.clock - 100
        return (None if elapsed >= self.no_totals_after else totals,
                None if elapsed >= self.no_steps_after or elapsed < self.steps_ready else steps)

    def advance(self, seconds, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            raise MotionCancelled("取消")
        for i in range(2):
            effective = self.command[i] if abs(self.command[i]) >= self.deadzone else 0.0
            tau = 0.06 if effective else self.coast_tau
            self.velocity[i] += seconds / (tau + seconds) * (effective - self.velocity[i])
            self.position[i] += self.velocity[i] * seconds
        self.clock += seconds
        if not self.freeze and self.clock + 1e-8 >= self.next_capture:
            self.index += 1
            error = self.error0 - 4 * sum(self.position) / 2
            frame = layout(self.width / 2 + error, self.width, self.color, self.index, self.clock)
            if self.missing():
                frame["candidates"] = frame["candidates"][:2]
            self.pending.append((self.clock + 0.05, frame))
            self.next_capture += 0.1
        while self.pending and self.pending[0][0] <= self.clock + 1e-8:
            _, self.latest = self.pending.pop(0)

    def observe_ball_layout(self, **kwargs):
        self.observe_count += 1
        return self.latest

    def ball_layout_sample(self):
        self.sample_count += 1
        return self.latest

    def wait_ball_layout(self, *, after, timeout, stop_event=None):
        self.advance(timeout, stop_event)
        sample = self.ball_layout_sample()
        return sample if sample["frame_index"] != after else None

    def run(self, *, log=lambda _: None, **config):
        with patch("base.ball_position.time.monotonic", side_effect=lambda: self.clock), \
                patch("base.ball_position.time.time", side_effect=lambda: self.clock):
            return calibrate_ball_position(self, self, config=config, stop_event=self.stop_event, log=log)


class PositionPIDTests(unittest.TestCase):
    def test_integrates_over_control_interval_and_filters_derivative(self):
        pid = PositionPID(validate_config({"kp": 0.6, "ki": 0.02, "kd": 0.08, "speed": 30}))
        self.assertAlmostEqual(pid.step(20, 100), 12)
        self.assertAlmostEqual(pid.step(30, 100.1), 18 + 3.2 + 0.06)
        self.assertAlmostEqual(pid.integral, 0.06)
        self.assertAlmostEqual(pid.derivative, 40)

    def test_saturated_error_does_not_wind_up_and_crossing_resets_integral(self):
        pid = PositionPID(validate_config({"kp": 0.6, "ki": 1, "kd": 0, "speed": 30}))
        for i in range(100):
            self.assertEqual(pid.step(100, 100 + i / 10), 30)
        self.assertEqual(pid.integral, 0)
        pid.step(10, 110)
        pid.step(10, 110.1)
        self.assertGreater(pid.integral, 0)
        self.assertLess(pid.step(-10, 110.2), 0)
        self.assertLessEqual(pid.integral, 0)
        pid.reset()
        self.assertEqual(pid.integral, 0)
        self.assertIsNone(pid.stamp)

    def test_invalid_distance_scale_and_position_tolerance_are_rejected(self):
        for field in ("mm_per_px", "position_tolerance_mm"):
            for value in (0, -1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    validate_config({field: value})


class ContinuousBallPositionTests(unittest.TestCase):
    def test_encoder_progress_reduces_position_error_between_visual_frames(self):
        robot = SimulatedRobot()
        robot.freeze = True
        errors = []
        original = PositionPID.step

        def step(pid, error, stamp):
            errors.append(error)
            return original(pid, error, stamp)

        with patch.object(PositionPID, "step", autospec=True, side_effect=step):
            with self.assertRaisesRegex(TimeoutError, "校准超时"):
                robot.run(mm_per_px=0.25, kp=0.6, timeout=0.3)
        self.assertEqual(robot.index, 0)
        self.assertGreater(len(errors), 20, "位置环在视觉帧间隙仍持续更新")
        self.assertAlmostEqual(errors[0], 30.0)
        self.assertLess(errors[-1], errors[0] - 1.0)

    def test_overshooting_encoder_target_brakes_before_reversing_on_a_fresh_frame(self):
        robot = SimulatedRobot()
        advance = robot.advance
        disturbed = False

        def disturb(seconds, stop_event=None):
            nonlocal disturbed
            advance(seconds, stop_event)
            if not disturbed and robot.clock - 100 >= 0.05:
                robot.position = [35.0, 35.0]  # 目标是 30 mm，模拟越过目标。
                disturbed = True

        robot.advance = disturb
        robot.run(mm_per_px=0.25, kp=0.6)
        self.assertTrue(robot.brakes)
        self.assertTrue(any(values[0] < 0 and values[2] < 0 for _, values in robot.commands))
        reverse_at = next(stamp for stamp, values in robot.commands if values[0] < 0 and values[2] < 0)
        self.assertGreaterEqual(reverse_at - robot.brakes[0][0], BALL_POSITION["settle"])

    def test_deadzone_and_slow_zero_speed_stop_converge_from_both_sides(self):
        for error in (-240, -30, 30, 240):
            with self.subTest(error=error):
                robot = SimulatedRobot(error, deadzone=25, coast_tau=0.4)
                result = robot.run(mm_per_px=0.25)
                self.assertLess(result["elapsed"], 8)
                self.assertLessEqual(abs(result["position_error_mm"]), 2)
                self.assertLess(max(abs(v) for v in robot.velocity), BALL_POSITION["speed_tolerance"])
                self.assertTrue(robot.brakes)

    def test_a_frame_frozen_before_stopping_cannot_confirm_completion(self):
        robot = SimulatedRobot(error=0)
        advance = robot.advance

        def freeze_early(seconds, stop_event=None):
            advance(seconds, stop_event)
            if robot.index >= 3:
                robot.freeze = True

        robot.advance = freeze_early
        with self.assertRaisesRegex(TimeoutError, "视觉断流"):
            robot.run(settle=0.5)

    def test_calibrated_visual_target_uses_capture_position_despite_inference_delay(self):
        for error in (-120, 120):
            with self.subTest(error=error):
                robot = SimulatedRobot(error)
                messages = []
                result = robot.run(mm_per_px=0.25, kp=0.6, log=messages.append)
                targets = [float(match.group(1)) for message in messages
                           if (match := re.search(r"目标位置 ([+-]?[\d.]+)mm", message))]
                self.assertGreater(len(targets), 10)
                # 正确比例下目标始终是初始误差 / 4，不随 50 ms 推理延迟向前漂移。
                self.assertTrue(all(abs(target - error / 4) <= 0.1 for target in targets))
                self.assertLessEqual(abs(result["position_error_mm"]), BALL_POSITION["position_tolerance_mm"])
                self.assertLessEqual(abs(result["error_px"]), BALL_POSITION["tolerance_px"])

    def test_new_detection_is_logged_without_waiting_half_a_second(self):
        robot = SimulatedRobot(error=70)
        messages = []
        with self.assertRaisesRegex(TimeoutError, "校准超时"):
            robot.run(timeout=0.35, log=messages.append)
        updates = [message for message in messages if message.startswith("  位置校准：")]
        self.assertGreaterEqual(len(updates), 3)

    def test_delayed_camera_converges_from_both_sides_without_step_stops(self):
        for error in (-120, 120):
            with self.subTest(error=error):
                robot = SimulatedRobot(error)
                result = robot.run()
                self.assertTrue(result["ok"])
                self.assertLessEqual(abs(result["error_px"]), BALL_POSITION["tolerance_px"])
                self.assertLessEqual(abs(result["position_error_mm"]), BALL_POSITION["position_tolerance_mm"])
                self.assertEqual(robot.observe_count, 1)
                self.assertGreater(robot.sample_count, robot.index)
                self.assertEqual(len(robot.stops), 2, "启动与退出停车，移动期间连续发轮速")
                moving = [values for _, values in robot.commands if any(values)]
                self.assertTrue(moving)
                self.assertTrue(all(value * error >= 0 for value in moving[0]))
                self.assertLessEqual(max(abs(value) for values in moving for value in values), BALL_POSITION["speed"])
                self.assertTrue(any(0 < abs(values[0]) < 10 for values in moving))
                for (previous_t, previous), (stamp, values) in zip(robot.commands[1:-1], robot.commands[2:-1]):
                    allowed = BALL_POSITION["accel"] * (stamp - previous_t) + 1e-6
                    if any(values):  # 到位制动直接归零，运动过程仍按加速度限幅。
                        self.assertLessEqual(max(abs(v - old) for v, old in zip(values, previous)), allowed)
                self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_uses_actual_image_width_and_already_centered_never_moves(self):
        robot = SimulatedRobot(error=0, width=600)
        result = robot.run()
        self.assertEqual(result["center_x"], 300)
        self.assertGreaterEqual(robot.index, 2)
        self.assertTrue(all(not any(values) for _, values in robot.commands))

    def test_duplicate_centered_frame_cannot_complete_or_keep_camera_alive(self):
        robot = SimulatedRobot(error=0)
        robot.freeze = True
        with self.assertRaisesRegex(TimeoutError, "视觉断流"):
            robot.run()
        self.assertGreaterEqual(robot.clock - 100, BALL_POSITION["lost_timeout"])
        self.assertTrue(all(not any(values) for _, values in robot.commands))

    def test_duplicate_frame_with_rewritten_stamp_cannot_renew_validity(self):
        robot = SimulatedRobot()
        robot.freeze = True
        robot.ball_layout_sample = lambda: {**robot.latest, "capture_stamp": robot.clock}
        with self.assertRaisesRegex(TimeoutError, "视觉断流"):
            robot.run()
        self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_frozen_delayed_frame_watchdog_includes_inference_delay(self):
        robot = SimulatedRobot()
        robot.freeze = True
        robot.latest["capture_stamp"] = 99.5
        with self.assertRaisesRegex(TimeoutError, "视觉断流"):
            robot.run()
        self.assertLessEqual(robot.clock - 100, 0.12)
        self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_missing_frame_stops_immediately_and_recovery_resumes_pid(self):
        robot = SimulatedRobot()
        robot.missing = lambda: 0.2 <= robot.clock - 100 < 0.4
        result = robot.run()
        self.assertTrue(result["ok"])
        pause = next(stamp for stamp in robot.stops if 0.2 <= stamp - 100 < 0.5)
        self.assertTrue(any(stamp > pause and any(values) for stamp, values in robot.commands))
        self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_permanent_missing_frames_fail_without_more_motion(self):
        robot = SimulatedRobot()
        robot.missing = lambda: robot.clock - 100 >= 0.2
        with self.assertRaisesRegex(TimeoutError, "持续缺球"):
            robot.run()
        first_stop = next(stamp for stamp in robot.stops if stamp - 100 > 0.1)
        self.assertTrue(all(not any(values) for stamp, values in robot.commands if stamp >= first_stop))

    def test_encoder_totals_and_speed_each_have_a_watchdog(self):
        for field in ("no_totals_after", "no_steps_after"):
            with self.subTest(field=field):
                robot = SimulatedRobot()
                setattr(robot, field, 0.2)
                with self.assertRaisesRegex(RuntimeError, "编码器"):
                    robot.run()
                self.assertLess(robot.clock - 100, 0.6)
                self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_delayed_initial_speed_feedback_does_not_accumulate_launch_command(self):
        robot = SimulatedRobot()
        robot.steps_ready = 0.15
        robot.run()
        stamp, values = next((stamp, values) for stamp, values in robot.commands if any(values))
        self.assertGreaterEqual(stamp - 100, 0.15)
        self.assertLessEqual(max(abs(value) for value in values), 1.01)

    def test_cumulative_distance_limit_stops_even_with_a_valid_far_target(self):
        robot = SimulatedRobot()
        with self.assertRaisesRegex(RuntimeError, "累计移动距离上限"):
            robot.run(max_distance_m=0.005)
        self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_changed_middle_identity_stops(self):
        robot = SimulatedRobot()
        robot.color = "red"
        with self.assertRaisesRegex(RuntimeError, "身份发生变化"):
            robot.run()
        self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_timeout_and_cancel_always_stop(self):
        robot = SimulatedRobot()
        with self.assertRaisesRegex(TimeoutError, "校准超时"):
            robot.run(timeout=0.1)
        self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))
        robot = SimulatedRobot()
        robot.stop_event.set()
        with self.assertRaises(MotionCancelled):
            robot.run()
        self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))

    def test_cancel_during_sampling_prevents_another_motor_command(self):
        robot = SimulatedRobot()
        original = robot.ball_layout_sample
        def sample():
            robot.stop_event.set()
            return original()
        robot.ball_layout_sample = sample
        with self.assertRaises(MotionCancelled):
            robot.run()
        self.assertEqual(robot.commands[-1][1], (0, 0, 0, 0))
        self.assertEqual(len(robot.commands), 3, "启动、首拍运动、取消后停车")

    def test_motion_error_is_preserved_even_if_cleanup_stop_fails(self):
        robot = SimulatedRobot()
        original = OSError("电机断开")
        def feedback(timeout=0):
            if timeout == 0:
                raise original
            return [0] * 4, None
        robot.feedback = feedback
        stop = robot.stop
        def failing_stop():
            if robot.stops:
                raise OSError("停车失败")
            stop()
        robot.stop = failing_stop
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(OSError) as failed:
            robot.run()
        self.assertIs(failed.exception, original)

    def test_invalid_image_geometry_cannot_send_motion(self):
        for field, value in (("size", None), ("size", [0, 480]), ("candidates", [{"center_x": float("inf")}])):
            with self.subTest(field=field):
                robot = SimulatedRobot()
                robot.latest[field] = value
                with self.assertRaises(ValueError):
                    robot.run()
                self.assertTrue(all(not any(values) for _, values in robot.commands))


class BallPositionTaskTests(unittest.TestCase):
    def test_command_runs_before_home_and_dry_run_shows_pid_without_hardware(self):
        commands = [step.command for step in load_section(ROOT / "tasks.txt", "跑图")]
        index = commands.index("calibrate-ball-position")
        self.assertEqual(commands[index - 1], "detect-balls")
        self.assertEqual(commands[index + 1], "home")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            result = execute(Step(1, "calibrate-ball-position", ()), DryBase(), None, DryVision())
        self.assertIsNone(result)
        self.assertIn("偏左 → 后退，偏右 → 前进", output.getvalue())
        self.assertIn("编码器目标位置 → 位置 PID → 轮速 PI", output.getvalue())
        self.assertIn(f"视觉换算 {BALL_POSITION['mm_per_px']:g} mm/px", output.getvalue())
        self.assertIn(f"Kp={BALL_POSITION['kp']:g}(1/s)", output.getvalue())
        self.assertIn(f"Ki={BALL_POSITION['ki']:g}(1/s²)，Kd={BALL_POSITION['kd']:g}", output.getvalue())
        self.assertNotIn("校准完成", output.getvalue())

    def test_task_and_base_pass_shared_devices_and_cancellation_to_controller(self):
        base, vision = Base(), Mock()
        base.motor, base._imu_checked = Mock(), True
        stop = base.stop_event
        with patch("base.api.ball_position.calibrate_ball_position", return_value={"ok": True}) as calibrate:
            self.assertEqual(run(base, vision, stop_event=stop, log=None), {"ok": True})
        calibrate.assert_called_once_with(base.motor, vision, config=BALL_POSITION, log=None, stop_event=stop)
        base.motor.stop.assert_called()

    def test_task_command_passes_shared_cancellation_to_calibration(self):
        stop = threading.Event()
        base, vision = Mock(), Mock()
        with patch("tasks.runner.calibrate_ball_position.run", return_value={"ok": True}) as calibrate, \
                patch("tasks.runner.pause"):
            execute(Step(1, "calibrate-ball-position", ()), base, None, vision, stop_event=stop)
        calibrate.assert_called_once_with(base, vision, stop_event=stop)


class BallLayoutApiTests(unittest.TestCase):
    def test_nonblocking_sample_keeps_missing_balls_and_rejects_stale_or_disabled_model(self):
        clock = [100.1]
        vision = Vision(clock=lambda: clock[0])
        vision.camera = Mock(error="")
        vision._enabled["objects"] = True
        vision._objects = ({"size": [600, 240], "detections": [
            {"name": "红球", "box": [10, 20, 30, 40]},
            {"name": "绿球", "box": [50, 20, 70, 40]}]}, 9, 100.0)
        try:
            sample = vision.ball_layout_sample()
            self.assertEqual(sample["size"], [600, 240])
            self.assertEqual(sample["frame_index"], 9)
            self.assertEqual(sample["capture_stamp"], 100.0)
            self.assertEqual(len(sample["candidates"]), 2)
            vision.camera.next_frame.assert_not_called()
            clock[0] = 100.7
            self.assertIsNone(vision.ball_layout_sample())
            clock[0] = 100.1
            vision._enabled["objects"] = False
            self.assertIsNone(vision.ball_layout_sample())
        finally:
            vision.close()

    def test_nonblocking_sample_propagates_camera_error_and_cancellation(self):
        vision = Vision()
        vision.camera = Mock(error="摄像头断开")
        try:
            with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
                vision.ball_layout_sample()
            vision.camera.error = ""
            vision.stop_event.set()
            with self.assertRaises(MotionCancelled):
                vision.ball_layout_sample()
        finally:
            vision.close()

    def test_reads_fresh_coordinates_and_size_from_existing_background_model(self):
        camera, vision = ContinuousCamera(), Vision()
        predictor = Mock()
        threads = []

        def predict(frame):
            threads.append(threading.current_thread().name)
            return {"size": [600, 240], "detections": [
                {"name": name, "box": [x - 20, 10, x + 20, 50]}
                for name, x in (("蓝球", 450), ("红球", 150), ("绿球", 300))]}

        predictor.predict.side_effect = predict
        vision._object_model = predictor
        try:
            with patch("vision.api.CameraStream", return_value=camera):
                vision.set_models(boundary=False, objects=True)
                first = vision.observe_ball_layout(timeout=1)
                second = vision.observe_ball_layout(timeout=1)
            self.assertEqual(first["size"], [600, 240])
            self.assertEqual([ball["center_x"] for ball in first["candidates"]], [150, 300, 450])
            self.assertGreater(second["frame_index"], first["frame_index"])
            self.assertGreater(second["capture_stamp"], first["capture_stamp"])
            self.assertEqual(set(threads), {"vision-inference"})
        finally:
            vision.close()

    def test_old_cached_result_cannot_supply_calibration_geometry(self):
        vision = Vision()
        vision.camera = Mock(error="")
        stamp = time.time()
        vision._objects = ({"size": [640, 480], "detections": []}, 7, stamp)
        try:
            for after, cutoff in ((7, stamp - 0.1), (6, stamp)):
                with self.subTest(after=after, cutoff=cutoff), self.assertRaises(TimeoutError):
                    vision._wait_candidates("ball", after=after, after_stamp=cutoff, timeout=0, include_size=True)
        finally:
            vision.close()
