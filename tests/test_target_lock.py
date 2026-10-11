"""彩色靶锁定、PID 方向和模拟两轴闭环；不连接硬件。"""
from contextlib import ExitStack, redirect_stderr
import io
import signal
import struct
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from arm.config import ArmConfig, JOINT_IDS
from arm.errors import ProtocolError
from arm.motor import ANGLE_SCALE, SimulatedBus, position_payload
from base.control import MotionCancelled
from config import ARM
from tasks.lock_target import (PID, RecoveryPose, TargetAcquisition, TargetLock, WAIT_JOINTS,
                               main, run, validate_config)
from vision.targets import candidates_from_detections


def target(color, x, y):
    return {"value": color, "name": color, "box": [x - 10, y - 10, x + 10, y + 10],
            "center_x": x}


def image(color="red", x=220, y=80):
    """假画面附带可控的 YOLO 输出，不通过像素颜色生成检测框。"""
    rows = [] if color is None else [{"name": color + "_target", "confidence": 0.9,
                                    "box": [round(x) - 10, round(y) - 10,
                                            round(x) + 10, round(y) + 10]}]
    return SimpleNamespace(shape=(240, 320, 3), detections=rows)


def prediction(frame):
    return {"size": [frame.shape[1], frame.shape[0]], "detections": frame.detections}


def target_sample(frame, index, stamp):
    return {"size": (frame.shape[1], frame.shape[0]),
            "candidates": candidates_from_detections(frame.detections, "target"),
            "frame_index": index, "capture_stamp": stamp}


def scene(color="red", x=220, y=80):
    """三个完整标靶；指定颜色居中，其余两色分别放在左右。"""
    frame = image(color, x, y)
    if color is not None:
        others = [value for value in ("red", "green", "blue") if value != color]
        for other, center_x in zip(others, (40, 290)):
            frame.detections.extend(image(other, center_x, 120).detections)
    return frame


class Clock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def wall(self):
        return 1000.0 + self.now

    def wait(self, seconds, stop_event=None):
        if stop_event is not None and stop_event.is_set():
            raise MotionCancelled("测试取消")
        self.now += max(0.0, seconds)


class TargetLockTests(unittest.TestCase):
    def test_default_selects_middle_target_in_horizontal_order(self):
        lock = TargetLock()
        result = lock.step([target("blue", 100, 300), target("green", 800, 300),
                            target("red", 300, 180)], (1000, 600), 0.05)
        self.assertEqual(result["target"]["value"], "red")

    def test_position_and_color_parameters_select_the_requested_target(self):
        rows = [target("blue", 900, 300), target("green", 500, 300), target("red", 100, 300)]
        for selector, color in (("left", "red"), ("right", "blue"), ("middle", "green"),
                                ("red", "red"), ("green", "green"), ("blue", "blue")):
            with self.subTest(selector=selector):
                self.assertEqual(TargetLock(selector).step(rows, (1000, 600), 0.05)["target"]["value"], color)

    def test_middle_is_the_middle_of_three_targets_even_when_left_is_near_image_center(self):
        rows = [target("red", 480, 300), target("green", 700, 300), target("blue", 900, 300)]
        self.assertEqual(TargetLock("middle").step(rows, (1000, 600), 0.05)["target"]["value"], "green")

    def test_latched_target_survives_reordering_and_missing_frames(self):
        lock = TargetLock()
        lock.step([target("green", 500, 300), target("red", 800, 300)], (1000, 600), 0.05)
        result = lock.step([target("red", 500, 300), target("green", 650, 300)], (1000, 600), 0.05)
        self.assertEqual(result["target"]["value"], "green")
        self.assertIsNone(lock.step([target("red", 500, 300)], (1000, 600), 0.05))
        self.assertEqual(lock.color, "green")
        self.assertTrue(all(pid.previous is None for pid in lock.pids.values()))
        result = lock.step([target("green", 660, 300)], (1000, 600), 0.05)
        self.assertEqual(result["target"]["value"], "green")

    def test_nearby_matching_does_not_jump_to_a_distant_same_color(self):
        lock = TargetLock()
        lock.step([target("red", 500, 300)], (1000, 600), 0.05)
        result = lock.step([target("red", 900, 300), target("red", 520, 310)], (1000, 600), 0.05)
        self.assertEqual(result["target"]["center_x"], 520)
        self.assertIsNone(lock.step([target("red", 900, 300)], (1000, 600), 0.05))

    def test_all_four_quadrants_use_the_requested_motor_directions(self):
        for x, y in ((350, 200), (650, 200), (350, 400), (650, 400)):
            with self.subTest(x=x, y=y):
                result = TargetLock().step([target("red", x, y)], (1000, 600), 0.05)
                self.assertGreater(result["rates"][1] * (x - 500), 0)
                self.assertGreater(result["rates"][4] * (y - 300), 0)

    def test_center_tolerance_resets_only_the_aligned_axis(self):
        lock = TargetLock(config={"tolerance_px": 8})
        lock.step([target("red", 600, 400)], (1000, 600), 0.05)
        result = lock.step([target("red", 504, 390)], (1000, 600), 0.05)
        self.assertEqual(result["rates"][1], 0)
        self.assertIsNone(lock.pids[1].previous)
        self.assertGreater(result["rates"][4], 0)

    def test_calibrated_aim_controls_directions_and_tolerance(self):
        lock = TargetLock(config={"tolerance_px": 8})
        result = lock.step([target("red", 550, 250)], (1000, 600), 0.05, aim=(0.6, 0.3))
        self.assertAlmostEqual(result["errors"][1], -50)
        self.assertAlmostEqual(result["errors"][4], 70)
        self.assertLess(result["rates"][1], 0)
        self.assertGreater(result["rates"][4], 0)
        result = lock.step([target("red", 604, 183)], (1000, 600), 0.05, aim=(0.6, 0.3))
        self.assertEqual(result["rates"], {1: 0, 4: 0})

    def test_aim_change_resets_pid_history_and_keeps_the_selected_target(self):
        lock = TargetLock(config={"kp": 1, "ki": 1, "kd": 10, "derivative_tau": 0})
        lock.step([target("red", 600, 360)], (1000, 600), 0.1)
        result = lock.step([target("green", 550, 330), target("red", 600, 360)],
                           (1000, 600), 0.1, aim=(0.55, 0.55))
        self.assertEqual(result["target"]["value"], "red")
        for pid in lock.pids.values():
            self.assertEqual(pid.derivative, 0)
            self.assertAlmostEqual(pid.integral, 0.01)
        for rate in result["rates"].values():
            self.assertAlmostEqual(rate, 0.11)

    def test_initial_selection_uses_image_center_even_with_a_calibrated_aim(self):
        result = TargetLock().step([target("red", 500, 300), target("blue", 900, 300)],
                                   (1000, 600), 0.05, aim=(0.9, 0.5))
        self.assertEqual(result["target"]["value"], "red")

    def test_integral_and_derivative_terms_affect_the_control_output(self):
        pid = PID(validate_config({"kp": 1, "ki": 1, "kd": 0}))
        self.assertAlmostEqual(pid.step(0.5, 0.1), 0.55)
        self.assertAlmostEqual(pid.step(0.5, 0.1), 0.60)
        pid = PID(validate_config({"kp": 1, "ki": 0, "kd": 0.2, "derivative_tau": 0}))
        self.assertAlmostEqual(pid.step(0.1, 0.1), 0.1)
        self.assertAlmostEqual(pid.step(0.3, 0.1), 0.7)

    def test_saturation_and_sign_change_do_not_wind_up_or_reverse_direction(self):
        pid = PID(validate_config({"kp": 10, "ki": 2, "kd": 0.5, "max_rate_deg_s": 2}))
        for _ in range(200):
            self.assertEqual(pid.step(1, 0.05), 2)
        self.assertEqual(pid.integral, 0)
        self.assertLessEqual(pid.step(-0.1, 0.05), 0)
        # 快速接近中心时，微分制动可以归零，但不能向相反方向移动。
        self.assertLessEqual(pid.step(-0.01, 0.05), 0)

    def test_invalid_configuration_and_time_steps_are_rejected(self):
        for cfg in ({"kp": float("nan")}, {"hz": 0}, {"ki": -1}, {"max_step_deg": 0}, {"max_step_deg": 0.01},
                    {"match_distance_ratio": 1}, {"unknown": 1},
                    {"motor2_angle_deg": float("nan")}, {"motor2_angle_deg": 1e12},
                    {"stable_frames": 0}, {"stable_frames": 2.5}, {"stable_frames": True},
                    {"stable_distance_ratio": 0}, {"stable_distance_ratio": 1}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_config(cfg)
        with self.assertRaises(ValueError):
            TargetLock("invalid")
        with self.assertRaises(ValueError):
            TargetLock().step([], (320, 240), 0)


class TargetLockRunHarness(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("tasks.lock_target.time.monotonic", self.clock.monotonic))
        self.stack.enter_context(patch("tasks.lock_target.time.time", self.clock.wall))
        self.stack.enter_context(patch("tasks.lock_target.wait_cancelable", self.clock.wait))
        self.predictor = Mock()
        self.predictor.predict.side_effect = prediction
        self.loader = self.stack.enter_context(patch("tasks.lock_target.load_predictor", return_value=self.predictor))
        self.bus = SimulatedBus(ArmConfig())
        for motor in self.bus.motors.values():
            motor.move = Mock(wraps=motor.move)

    def scripted_camera(self, samples):
        samples = iter(samples)
        camera = Mock()

        def next_frame(**kwargs):
            try:
                frame, index, age = next(samples)
            except StopIteration:
                raise KeyboardInterrupt
            if isinstance(frame, Exception):
                raise frame
            return frame, index, self.clock.wall() - age

        camera.next_frame.side_effect = next_frame
        return camera


class DriverIntegralRunTests(TargetLockRunHarness):
    def setUp(self):
        super().setUp()
        for motor in self.bus.motors.values():
            motor.set_speed_ki = Mock(wraps=motor.set_speed_ki)

    def assert_idle_integrals(self):
        for motor in self.bus.motors.values():
            self.assertAlmostEqual(motor.speed_ki, 0.005)
        for addr in (2, 3):
            self.bus.motors[addr].set_speed_ki.assert_not_called()

    def test_high_integral_is_active_before_aiming_and_restored_after_completion(self):
        camera = self.scripted_camera([(scene(x=160, y=120), i, 0) for i in range(1, 7)])
        next_frame = camera.next_frame.side_effect

        def check_integrals(**kwargs):
            for addr in (1, 4):
                self.assertAlmostEqual(self.bus.motors[addr].speed_ki, 0.05)
            for addr in (2, 3):
                self.assertAlmostEqual(self.bus.motors[addr].speed_ki, 0.005)
            return next_frame(**kwargs)

        camera.next_frame.side_effect = check_integrals
        result = run(camera, self.bus, config={"hz": 2, "stable_frames": 1},
                     complete_after=2, log=lambda _: None)
        self.assertTrue(result["ok"])
        self.assert_idle_integrals()
        for addr in (1, 4):
            self.assertEqual([call.args[0] for call in self.bus.motors[addr].set_speed_ki.call_args_list],
                             [0.05, 0.005])

    def test_duration_and_alignment_timeout_restore_integrals(self):
        for complete_after in (0, 2):
            with self.subTest(complete_after=complete_after):
                camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 10)])
                if complete_after:
                    with self.assertRaisesRegex(TimeoutError, "锁靶超时"):
                        run(camera, self.bus, duration=0.03, complete_after=complete_after,
                            log=lambda _: None)
                else:
                    run(camera, self.bus, duration=0.03, log=lambda _: None)
                self.assert_idle_integrals()

    def test_cancellation_and_runtime_errors_restore_integrals_and_keep_original_error(self):
        for error in (KeyboardInterrupt(), MotionCancelled("取消"), RuntimeError("识别失败")):
            with self.subTest(error=error):
                camera = Mock()
                camera.next_frame.side_effect = error
                with self.assertRaises(type(error)) as failed:
                    run(camera, self.bus, log=lambda _: None)
                self.assertIs(failed.exception, error)
                self.assert_idle_integrals()

    def test_partial_startup_failure_restores_both_attempted_axes(self):
        setter = self.bus.motors[4].set_speed_ki._mock_wraps

        def fail_after_write(value):
            setter(value)
            if value == 0.05:
                raise ProtocolError("参数已生效，但应答超时")

        self.bus.motors[4].set_speed_ki.side_effect = fail_after_write
        camera = Mock()
        with self.assertRaisesRegex(ProtocolError, "应答超时"):
            run(camera, self.bus, log=lambda _: None)
        camera.next_frame.assert_not_called()
        self.assert_idle_integrals()

    def test_position_hold_failure_still_restores_integrals(self):
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 10)])
        with patch("tasks.lock_target.hold", side_effect=ProtocolError("保持失败")):
            with self.assertRaisesRegex(ProtocolError, "保持失败"):
                run(camera, self.bus, duration=0.03, log=lambda _: None)
        self.assert_idle_integrals()

    def test_one_restore_failure_retries_and_does_not_skip_the_other_axis(self):
        setter = self.bus.motors[1].set_speed_ki._mock_wraps

        def fail_restore(value):
            if value == 0.005:
                raise ProtocolError("ID1 恢复失败")
            setter(value)

        self.bus.motors[1].set_speed_ki.side_effect = fail_restore
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 10)])
        with redirect_stderr(io.StringIO()) as errors:
            with self.assertRaisesRegex(ProtocolError, "ID1 恢复失败"):
                run(camera, self.bus, duration=0.03, log=lambda _: None)
        self.assertEqual([call.args[0] for call in self.bus.motors[1].set_speed_ki.call_args_list],
                         [0.05, 0.005, 0.005, 0.005])
        self.assertAlmostEqual(self.bus.motors[4].speed_ki, 0.005)
        self.assertIn("ID1 速度环 Ki 恢复", errors.getvalue())

    def test_ctrl_c_during_restore_retries_then_propagates_after_both_axes_are_restored(self):
        setter = self.bus.motors[1].set_speed_ki._mock_wraps
        interrupted = False

        def interrupt_restore_once(value):
            nonlocal interrupted
            if value == 0.005 and not interrupted:
                interrupted = True
                raise KeyboardInterrupt
            setter(value)

        self.bus.motors[1].set_speed_ki.side_effect = interrupt_restore_once
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 10)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, duration=0.03, log=lambda _: None)
        self.assert_idle_integrals()
        self.assertEqual(self.bus.motors[1].set_speed_ki.call_count, 3)


class AcquisitionRunTests(TargetLockRunHarness):
    def test_shared_yolo_results_do_not_load_or_run_a_second_predictor(self):
        frame, camera = scene(), Mock()
        # 返回时间须跟随假时钟，保证三帧采集时间递增。
        samples = iter([None, 1, 2, 3])

        def next_sample(**kwargs):
            try:
                index = next(samples)
            except StopIteration:
                raise KeyboardInterrupt
            return None if index is None else target_sample(frame, index, self.clock.wall())

        source = Mock(side_effect=next_sample)
        updates = Mock()
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, candidate_source=source, on_update=updates, log=lambda _: None)
        camera.next_frame.assert_not_called()
        self.loader.assert_not_called()
        self.predictor.predict.assert_not_called()
        self.assertEqual(updates.call_args.args[1]["target"]["name"], "red_target")

    def test_yolo_no_detections_keeps_waiting_even_when_input_fixture_has_targets(self):
        self.predictor.predict.side_effect = lambda frame: {"size": [320, 240], "detections": []}
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 6)])
        updates = Mock()
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, on_update=updates, log=lambda _: None)
        self.assertEqual(self.predictor.predict.call_count, 5)
        self.assertTrue(all(call.args[1] is None for call in updates.call_args_list))
        self.bus.motors[2].move.assert_not_called()

    def test_ball_classes_do_not_count_as_target_labels(self):
        frame = scene()
        for detection in frame.detections:
            detection["name"] = {"red_target": "红球", "green_target": "绿球", "blue_target": "蓝球"}[
                detection["name"]]
        camera = self.scripted_camera([(frame, i, 0) for i in range(1, 6)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, log=lambda _: None)
        self.bus.motors[2].move.assert_not_called()

    def test_yolo_inference_error_propagates_and_never_starts_aiming(self):
        self.predictor.predict.side_effect = RuntimeError("YOLO 推理失败")
        camera = self.scripted_camera([(scene(), 1, 0)])
        with self.assertRaisesRegex(RuntimeError, "YOLO 推理失败"):
            run(camera, self.bus, log=lambda _: None)
        self.bus.motors[2].move.assert_not_called()
        self.bus.motors[3].move.assert_not_called()

    def test_three_stable_frames_are_required_before_any_aiming_command(self):
        index = 0
        camera, updates = Mock(), Mock()

        def next_frame(**kwargs):
            nonlocal index
            index += 1
            if index <= 3:
                for addr in JOINT_IDS:
                    self.bus.motors[addr].move.assert_not_called()
                return scene(x=160, y=120), index, self.clock.wall()
            raise KeyboardInterrupt

        camera.next_frame.side_effect = next_frame
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, on_update=updates, log=lambda _: None)
        self.assertEqual([call.args[1] is not None for call in updates.call_args_list], [False, False, True])
        self.bus.motors[2].move.assert_called_once()
        self.bus.motors[3].move.assert_not_called()

    def test_cancel_on_confirmation_frame_does_not_start_aiming(self):
        stop, camera = threading.Event(), Mock()
        index = 0

        def next_frame(**kwargs):
            nonlocal index
            index += 1
            if index == 3:
                stop.set()
            return scene(), index, self.clock.wall()

        camera.next_frame.side_effect = next_frame
        with self.assertRaises(MotionCancelled):
            run(camera, self.bus, stop_event=stop, log=lambda _: None)
        self.bus.motors[2].move.assert_not_called()
        self.bus.motors[3].move.assert_not_called()
        for addr in (1, 4):
            self.bus.motors[addr].move.assert_called_once_with(0.0, speed_rpm=0.01)

    def test_incomplete_duplicate_color_and_extra_candidates_never_start_aiming(self):
        duplicate = scene()
        duplicate.detections.extend(image("red", 80, 160).detections)
        same_color = image()
        same_color.detections.extend(image("red", 40, 120).detections + image("red", 290, 120).detections)
        overlapping = image(x=160, y=60)
        overlapping.detections.extend(image("green", x=160, y=120).detections
                                      + image("blue", x=160, y=180).detections)
        for frame in (image(), image(None), duplicate, same_color, overlapping):
            with self.subTest(count=len(frame.detections)):
                camera, updates = self.scripted_camera([(frame, i, 0) for i in range(1, 6)]), Mock()
                with self.assertRaises(KeyboardInterrupt):
                    run(camera, self.bus, on_update=updates, log=lambda _: None)
                self.assertTrue(all(call.args[1] is None for call in updates.call_args_list))
                self.bus.motors[2].move.assert_not_called()

    def test_increasing_frame_indices_with_repeated_capture_time_cannot_start_aiming(self):
        camera, updates = Mock(), Mock()
        camera.next_frame.side_effect = [(scene(), i, self.clock.wall()) for i in range(1, 6)] \
            + [KeyboardInterrupt()]
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, on_update=updates, log=lambda _: None)
        self.assertTrue(all(call.args[1] is None for call in updates.call_args_list))
        self.bus.motors[2].move.assert_not_called()

    def test_invalid_frames_break_the_acquisition_streak(self):
        frame = scene()
        interruptions = ((scene(None), 3, 0), (frame, 2, 0), (frame, 3, 0.8),
                         (frame, 3, -1), (TimeoutError(), 3, 0))
        for interruption in interruptions:
            with self.subTest(interruption=interruption[1:]):
                camera = self.scripted_camera([(frame, 1, 0), (frame, 2, 0), interruption,
                                               (frame, 4, 0), (frame, 5, 0), (frame, 6, 0)])
                updates = Mock()
                with self.assertRaises(KeyboardInterrupt):
                    run(camera, self.bus, on_update=updates, log=lambda _: None)
                self.assertEqual([call.args[1] is not None for call in updates.call_args_list],
                                 [False] * 5 + [True])

    def test_reordering_and_center_drift_restart_acquisition(self):
        # 前两帧次序不同，第三帧次序相同但中心漂移过大；从第三帧重新累计。
        camera = self.scripted_camera([(scene(), 1, 0), (scene("green"), 2, 0),
                                      (scene("green", x=200), 3, 0),
                                      (scene("green", x=201), 4, 0),
                                      (scene("green", x=200), 5, 0)])
        updates = Mock()
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, on_update=updates, log=lambda _: None)
        self.assertEqual([call.args[1] is not None for call in updates.call_args_list], [False] * 4 + [True])
        self.assertEqual(updates.call_args.args[1]["target"]["value"], "green")

    def test_stability_uses_first_frame_to_reject_slow_accumulated_drift(self):
        acquisition = TargetAcquisition(validate_config())
        for offset in (0, 20, 40):
            rows = [target(color, x + offset, 300)
                    for color, x in (("red", 100), ("green", 500), ("blue", 900))]
            self.assertIsNone(acquisition.step(rows, (1000, 600), 1000 + offset / 1000))
        self.assertEqual(acquisition.count, 1)

    def test_complete_timer_only_starts_after_three_target_acquisition(self):
        camera = self.scripted_camera([(scene(x=160, y=120), i, 0) for i in range(1, 8)])
        result = run(camera, self.bus, config={"hz": 2}, complete_after=2, log=lambda _: None)
        self.assertEqual(result["stable_seconds"], 2)
        self.assertEqual(self.clock.now, 3)
        self.assertEqual(camera.next_frame.call_count, 7)

    def test_loss_returns_all_joints_and_reacquires_same_color_at_a_new_position(self):
        zero = {1: 100, 2: 200, 3: 300, 4: 400}
        moved = image(x=55, y=210)
        moved.detections.extend(image("green", 180, 110).detections + image("blue", 270, 110).detections)
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 4)]
                                      + [(image(None), 4, 0)] + [(moved, i, 0) for i in range(5, 8)])
        updates = Mock()
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, encoder_zero=zero, on_update=updates, log=lambda _: None)
        targets = {addr: zero[addr] + ARM["joint_signs"][addr] *
                   (WAIT_JOINTS[addr] - ARM["joint_offsets_deg"][addr]) for addr in JOINT_IDS}
        for addr, angle in targets.items():
            self.assertIn(angle, [call.args[0] for call in self.bus.motors[addr].move.call_args_list])
        self.bus.motors[3].move.assert_called_once_with(targets[3], speed_rpm=ARM["speed_rpm"])
        results = [call.args[1] for call in updates.call_args_list if call.args[1] is not None]
        self.assertEqual([result["target"]["value"] for result in results], ["red", "red"])
        self.assertLess(results[-1]["errors"][1], 0)
        self.assertGreater(results[-1]["errors"][4], 0)
        self.assertEqual(self.bus.motors[2].move.call_count, 3)  # 两次锁定及一次复位。

    def test_tracking_only_requires_the_selected_target_after_acquisition(self):
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 4)] + [(image(), 4, 0)])
        updates = Mock()
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, on_update=updates, log=lambda _: None)
        self.assertIsNotNone(updates.call_args.args[1])
        self.bus.motors[2].move.assert_called_once()
        self.bus.motors[3].move.assert_not_called()

    def test_slow_recovery_keeps_consuming_frames_without_repeating_return_commands(self):
        original = self.bus.motors[1].move

        def deferred_return(angle, *args, **kwargs):
            if angle != 0:
                return original(angle, *args, **kwargs)

        self.bus.motors[1].move = Mock(side_effect=deferred_return)
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 4)]
                                      + [(image(None), 4, 0)] + [(scene(), i, 0) for i in range(5, 8)])
        updates = Mock()
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, on_update=updates, log=lambda _: None)
        self.assertEqual(camera.next_frame.call_count, 8)
        results = [call.args[1] for call in updates.call_args_list if call.args[1] is not None]
        self.assertEqual([result["target"]["value"] for result in results], ["red", "red"])
        self.assertEqual(sum(call.args[0] == 0 for call in self.bus.motors[1].move.call_args_list), 1)
        self.bus.motors[3].move.assert_called_once()

    def test_frame_before_reset_command_cannot_count_toward_reacquisition(self):
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 4)]
                                      + [(image(None), 4, 0), (scene(), 5, 0.2)]
                                      + [(scene(), i, 0) for i in range(6, 9)])
        updates = Mock()
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, on_update=updates, log=lambda _: None)
        results = [call.args[1] is not None for call in updates.call_args_list]
        self.assertEqual(results, [False, False, True, False, False, False, False, True])

    def test_stuck_recovery_does_not_time_out_or_stop_task(self):
        original = self.bus.motors[1].move

        def stuck_return(angle, *args, **kwargs):
            if angle != 0:
                return original(angle, *args, **kwargs)

        self.bus.motors[1].move = Mock(side_effect=stuck_return)
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 4)]
                                      + [(image(None), i, 0) for i in range(4, 111)])
        run(camera, self.bus, duration=1, log=lambda _: None)
        self.assertGreater(camera.next_frame.call_count, 4)
        self.assertNotEqual(self.bus.motors[1].angle, 0)
        self.bus.motors[3].move.assert_called_once_with(0.0, speed_rpm=ARM["speed_rpm"])

    def test_recovery_uses_configured_encoder_zero_instead_of_current_feedback(self):
        zero = {1: 100, 2: 200, 3: 300, 4: 400}
        pose = RecoveryPose(self.bus, ArmConfig(encoder_zero_deg=zero))
        self.assertEqual(pose.targets, {1: 10, 2: 200, 3: 300, 4: 380})


class TargetLockRunTests(TargetLockRunHarness):
    def setUp(self):
        super().setUp()
        # 这些用例集中验证 PID 和到位计时；多帧门槛在 AcquisitionRunTests 单独验证。
        self.stack.enter_context(patch.dict("tasks.lock_target.TARGET_LOCK", {"stable_frames": 1}))

    def test_completion_includes_eight_pixel_boundary_and_requires_full_two_seconds(self):
        camera = self.scripted_camera([(scene(x=168, y=128), i, 0) for i in range(1, 6)])
        result = run(camera, self.bus, config={"hz": 2}, complete_after=2,
                     complete_tolerance_px=8, log=lambda _: None)
        self.assertEqual(result, {"ok": True, "color": "red", "errors": {1: 8, 4: 8},
                                  "stable_seconds": 2})
        self.assertEqual(self.clock.now, 2)
        self.assertEqual(camera.next_frame.call_count, 5)
        self.assertTrue(self.bus.motors[2].enabled)
        self.assertFalse(self.bus.closed)

    def test_alignment_timer_resets_on_error_missing_duplicate_stale_future_and_timeout(self):
        interruptions = ((scene(x=169, y=120), 3, 0), (scene(x=160, y=129), 3, 0),
                         (scene(None), 3, 0), (scene(x=168, y=128), 2, 0),
                         (scene(x=168, y=128), 3, 0.8), (scene(x=168, y=128), 3, -1),
                         (TimeoutError(), 3, 0))
        for interruption in interruptions:
            with self.subTest(interruption=interruption[1:]):
                self.clock.now = 0
                aligned = scene(x=168, y=128)
                samples = [(aligned, 1, 0), (aligned, 2, 0), interruption]
                samples += [(aligned, i, 0) for i in range(4, 9)]
                camera = self.scripted_camera(samples)
                result = run(camera, self.bus, config={"hz": 2}, complete_after=2,
                             complete_tolerance_px=8, log=lambda _: None)
                self.assertEqual(result["stable_seconds"], 2)
                self.assertEqual(self.clock.now, 3.5)
                self.assertEqual(camera.next_frame.call_count, 8)

    def test_long_gap_between_fresh_frames_restarts_alignment_timer(self):
        camera = Mock()
        index = 0
        last_capture = None

        def next_frame(**kwargs):
            nonlocal index, last_capture
            index += 1
            if index == 3:
                self.clock.now += 1
            elif last_capture is not None:
                self.clock.now = max(self.clock.now, last_capture + 0.5)
            last_capture = self.clock.now
            return scene(x=160, y=120), index, self.clock.wall()

        camera.next_frame.side_effect = next_frame
        result = run(camera, self.bus, config={"hz": 2}, complete_after=2, log=lambda _: None)
        self.assertEqual(result["stable_seconds"], 2)
        self.assertEqual(self.clock.now, 4)
        self.assertEqual(index, 7)

    def test_cached_frame_before_task_start_cannot_start_alignment_timer(self):
        camera = Mock()
        index = 0

        def next_frame(**kwargs):
            nonlocal index
            index += 1
            stamp = self.clock.wall() - (0.1 if index == 1 else 0)
            return scene(x=160, y=120), index, stamp

        camera.next_frame.side_effect = next_frame
        result = run(camera, self.bus, config={"hz": 20}, complete_after=2, log=lambda _: None)
        self.assertGreaterEqual(self.clock.now, 2.05)
        self.assertEqual(result["stable_seconds"], 2)

    def test_aim_change_within_tolerance_restarts_alignment_timer(self):
        camera, aim = Mock(), Mock()
        aim.point.return_value = (0.5, 0.5)
        index = 0

        def next_frame(**kwargs):
            nonlocal index
            index += 1
            if index == 3:
                aim.point.return_value = (0.51, 0.5)
            return scene(x=160, y=120), index, self.clock.wall()

        camera.next_frame.side_effect = next_frame
        result = run(camera, self.bus, config={"hz": 2}, aim=aim, complete_after=2, log=lambda _: None)
        self.assertEqual(result["stable_seconds"], 2)
        self.assertEqual(self.clock.now, 3)
        self.assertEqual(index, 7)

    def test_one_axis_outside_tolerance_cannot_complete_before_timeout(self):
        for x, y in ((169, 120), (160, 129)):
            with self.subTest(x=x, y=y):
                self.clock.now = 0
                camera = self.scripted_camera([(scene(x=x, y=y), i, 0) for i in range(1, 6)])
                with self.assertRaisesRegex(TimeoutError, "锁靶超时"):
                    run(camera, self.bus, config={"hz": 2}, duration=2.5,
                        complete_after=2, log=lambda _: None)
                self.assertEqual(camera.next_frame.call_count, 5)

    def test_yolo_boxes_and_simulated_motors_converge_to_center(self):
        camera = Mock()
        count = 0

        def next_frame(**kwargs):
            nonlocal count
            count += 1
            if count > 1:
                self.assertEqual(self.bus.motors[2].angle, 12.0)
                self.assertTrue(self.bus.motors[2].enabled)
            x = 160 + (2.0 - self.bus.motors[1].angle) * 50
            y = 120 + (-1.5 - self.bus.motors[4].angle) * 40
            return scene(x=x, y=y), count, self.clock.wall()

        camera.next_frame.side_effect = next_frame
        run(camera, self.bus, duration=3, config={"tolerance_px": 2}, log=lambda message: None)
        self.assertAlmostEqual(self.bus.motors[1].angle, 2.0, delta=0.07)
        self.assertAlmostEqual(self.bus.motors[4].angle, -1.5, delta=0.09)
        self.bus.motors[2].move.assert_called_once_with(12.0, speed_rpm=ARM["speed_rpm"])
        self.bus.motors[3].move.assert_not_called()
        self.assertFalse(self.bus.motors[3].enabled)

    def test_no_target_does_not_start_motor2_aiming(self):
        self.bus.motors[2].angle = 35.0
        camera = self.scripted_camera([(scene(None), 1, 0), (TimeoutError(), 2, 0)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, log=lambda message: None)
        self.bus.motors[2].move.assert_not_called()
        self.assertEqual(self.bus.motors[2].angle, 35.0)
        self.assertFalse(self.bus.motors[2].enabled)

    def test_live_aim_changes_reach_motor_commands_on_the_next_frame(self):
        aim = Mock()
        aim.point.side_effect = [(0.5, 0.5), (0.8, 0.2)]
        updates = Mock()
        camera = self.scripted_camera([(scene(), 1, 0), (scene(), 2, 0)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, aim=aim, on_update=updates, log=lambda message: None)
        for addr, sign in ((1, 1), (4, -1)):
            calls = self.bus.motors[addr].move.call_args_list
            self.assertGreater(calls[0].args[0] * sign, 0)
            self.assertLess((calls[1].args[0] - calls[0].args[0]) * sign, 0)
        self.assertEqual(updates.call_count, 2)
        self.assertLess(updates.call_args.args[1]["errors"][1], 0)
        self.assertGreater(updates.call_args.args[1]["errors"][4], 0)
        self.bus.motors[2].move.assert_called_once_with(12.0, speed_rpm=ARM["speed_rpm"])

    def test_motor2_command_failure_prevents_starting_target_tracking(self):
        self.bus.motors[2].move = Mock(side_effect=OSError("ID2 串口错误"))
        camera = self.scripted_camera([(scene(), 1, 0)])
        with self.assertRaisesRegex(OSError, "ID2 串口错误"):
            run(camera, self.bus, log=lambda message: None)
        camera.next_frame.assert_called_once()
        for addr in (1, 4):
            self.bus.motors[addr].move.assert_called_once_with(0.0, speed_rpm=0.01)

    def test_missing_target_restores_observation_pose_and_does_not_switch_color(self):
        # 复位后只看到绿靶，不能继续瞄准；原红靶在右侧重新出现后仍跟踪红色。
        camera = self.scripted_camera([(scene(), 1, 0), (scene(None), 2, 0),
                                      (image("green"), 3, 0), (scene(), 4, 0)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, log=lambda message: None)
        for addr, sign in ((1, 1), (4, -1)):
            angles = [call.args[0] for call in self.bus.motors[addr].move.call_args_list]
            self.assertEqual(len(angles), 4)  # 移动、复位、恢复移动、退出保持。
            self.assertGreater(angles[0] * sign, 0)
            self.assertEqual(angles[1], 0)
            self.assertGreater((angles[2] - angles[1]) * sign, 0)
            self.assertEqual(angles[2], angles[3])

    def test_duplicate_stale_future_and_timed_out_frames_never_drive_motion(self):
        camera = self.scripted_camera([(scene(), 1, 0), (scene(), 1, 0), (scene(), 2, 0.8),
                                      (scene(), 3, -1), (TimeoutError(), 4, 0)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, log=lambda message: None)
        for addr in (1, 4):
            angles = [call.args[0] for call in self.bus.motors[addr].move.call_args_list]
            self.assertEqual(len(angles), 3)
            self.assertGreater(abs(angles[0]), 0)
            self.assertEqual(angles[1:], [0, 0])
        self.assertEqual(camera.next_frame.call_args_list[1].kwargs["after"], 1)

    def test_delayed_feedback_makes_the_frame_expire_before_motion(self):
        original = self.bus.motors[1].read_status

        reads = 0

        def delayed():
            nonlocal reads
            reads += 1
            if reads == 2:
                self.clock.now += 0.7
            return original()

        self.bus.motors[1].read_status = delayed
        camera = self.scripted_camera([(scene(), 1, 0)])
        run(camera, self.bus, duration=0.1, log=lambda message: None)
        for addr in (1, 4):
            self.assertEqual(self.bus.motors[addr].move.call_count, 2)
            self.assertTrue(all(call.args[0] == 0 for call in self.bus.motors[addr].move.call_args_list))

    def test_expired_detection_does_not_latch_the_initial_target(self):
        camera = self.scripted_camera([(scene(), 1, 0), (scene("green"), 2, 0)])
        first = True

        def slow_detection(frame):
            nonlocal first
            if first:
                self.clock.now += 0.7
                first = False
            return prediction(frame)

        log = Mock()
        with patch.object(self.predictor, "predict", side_effect=slow_detection), \
                self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, log=log)
        self.assertTrue(any("锁定 green" in call.args[0] for call in log.call_args_list))
        self.assertFalse(any("锁定 red" in call.args[0] for call in log.call_args_list))

    def test_small_corrections_survive_real_motor_protocol_quantization(self):
        original_move = self.bus.motors[1].move

        def firmware_move(angle_deg, speed_rpm=10, accel_rpm_s=None):
            _, payload = position_payload(angle_deg, speed_rpm, accel_rpm_s)
            encoder = struct.unpack_from("<i", payload, 1)[0]
            return original_move(encoder * ANGLE_SCALE, speed_rpm, accel_rpm_s)

        self.bus.motors[1].move = Mock(side_effect=firmware_move)
        frame = SimpleNamespace(shape=(720, 1280, 3), detections=[
            {"name": color + "_target", "box": [x - 20, 340, x + 20, 380]}
            for color, x in (("red", 649), ("green", 160), ("blue", 1120))])
        camera = self.scripted_camera([(frame, i, 0) for i in range(1, 4)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, config={"ki": 0, "kd": 0}, log=lambda message: None)
        self.assertGreater(self.bus.motors[1].angle, 0)
        angles = [call.args[0] for call in self.bus.motors[1].move.call_args_list]
        self.assertTrue(all(later > earlier for earlier, later in zip([0] + angles[:2], angles[:3])))

    def test_slow_motor_targets_use_feedback_and_respect_the_step_limit(self):
        for addr in (1, 4):
            self.bus.motors[addr].move = Mock()  # 反馈不变，模拟目标尚未到达。
        camera = self.scripted_camera([(scene(), i, 0) for i in range(1, 5)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, config={"max_step_deg": 0.1, "ki": 0, "kd": 0}, log=lambda message: None)
        for addr, sign in ((1, 1), (4, -1)):
            angles = [call.args[0] for call in self.bus.motors[addr].move.call_args_list]
            self.assertGreater(angles[0] * sign, 0)
            self.assertLessEqual(abs(angles[0]), 0.1)
            self.assertEqual(angles[:-1], [angles[0]] * 4)
            self.assertEqual(angles[-1], 0.0)

    def test_command_failure_still_attempts_to_hold_both_axes(self):
        original = self.bus.motors[1].move
        failed = False

        def fail_once(*args, **kwargs):
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("串口错误")
            return original(*args, **kwargs)

        self.bus.motors[1].move = Mock(side_effect=fail_once)
        camera = self.scripted_camera([(scene(), 1, 0)])
        with self.assertRaisesRegex(OSError, "串口错误"):
            run(camera, self.bus, log=lambda message: None)
        self.assertEqual(self.bus.motors[1].move.call_count, 2)
        self.bus.motors[4].move.assert_called_once_with(0.0, speed_rpm=0.01)

    def test_cancellation_holds_the_current_motor_angles(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(MotionCancelled):
            run(Mock(), self.bus, stop_event=stop, log=lambda message: None)
        for addr in (1, 4):
            self.bus.motors[addr].move.assert_called_once_with(0.0, speed_rpm=0.01)
        self.bus.motors[2].move.assert_not_called()

    def test_cli_closes_both_resources_after_interruption(self):
        camera, bus, web = Mock(), Mock(), Mock()
        with patch("tasks.lock_target.CameraStream", return_value=camera), \
                patch("tasks.lock_target.MotorBus", return_value=bus), \
                patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                patch("tasks.lock_target.local_addresses", return_value=[]), \
                patch("tasks.lock_target.run", side_effect=KeyboardInterrupt) as task, \
                patch("builtins.print"):
            self.assertEqual(main([]), 0)
        self.assertEqual(task.call_args.args[2], "middle")
        bus.close.assert_called_once_with(disable_motors=True)
        camera.close.assert_called_once()
        web.start.assert_called_once_with(camera)
        web.close.assert_called_once()
        self.assertIs(task.call_args.kwargs["on_update"], web.update)
        self.assertTrue(task.call_args.kwargs["stop_event"].is_set())

    def test_cli_loads_project_yolo_with_requested_device_and_confidence(self):
        camera, bus, web = Mock(), Mock(), Mock()
        with patch("tasks.lock_target.CameraStream", return_value=camera), \
                patch("tasks.lock_target.MotorBus", return_value=bus), \
                patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                patch("tasks.lock_target.local_addresses", return_value=[]), \
                patch("tasks.lock_target.run") as task, patch("builtins.print"):
            self.assertEqual(main(["red", "--device", "cpu", "--conf", "0.4", "--duration", "1"]), 0)
        self.loader.assert_called_once_with(device="cpu", threshold=0.4)
        self.assertIs(task.call_args.kwargs["predictor"], self.predictor)

    def test_cli_ctrl_c_disables_all_four_motors_after_holding(self):
        # 模拟启动前 ID3 已经使能，退出应释放四轴，包括保持 12° 的 ID2。
        self.bus.motors[3].enabled = True
        camera = self.scripted_camera([(scene(), 1, 0)])
        web = Mock()
        with patch("tasks.lock_target.CameraStream", return_value=camera), \
                patch("tasks.lock_target.MotorBus", return_value=self.bus), \
                patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                patch("tasks.lock_target.local_addresses", return_value=[]), \
                patch("builtins.print"):
            self.assertEqual(main([]), 0)
        self.assertTrue(self.bus.closed)
        self.assertTrue(all(not motor.enabled for motor in self.bus.motors.values()))
        self.assertEqual(self.bus.motors[2].angle, 12.0)
        camera.close.assert_called_once()
        web.close.assert_called_once()

    def test_cli_normal_completion_preserves_motor_holding(self):
        camera, bus, web = Mock(), Mock(), Mock()
        with patch("tasks.lock_target.CameraStream", return_value=camera), \
                patch("tasks.lock_target.MotorBus", return_value=bus), \
                patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                patch("tasks.lock_target.local_addresses", return_value=[]), \
                patch("tasks.lock_target.run"), patch("builtins.print"):
            self.assertEqual(main(["--duration", "1"]), 0)
        bus.close.assert_called_once_with(disable_motors=False)

    def test_second_ctrl_c_during_disable_does_not_interrupt_cleanup(self):
        camera, bus, web = Mock(), Mock(), Mock()
        handlers, order = {}, []

        def install_signal(sig, handler):
            handlers[sig] = handler
            return signal.SIG_DFL

        def interrupted_run(*args, **kwargs):
            handlers[signal.SIGINT](signal.SIGINT, None)

        def close_bus(**kwargs):
            self.assertTrue(kwargs["disable_motors"])
            handlers[signal.SIGINT](signal.SIGINT, None)
            order.append("bus")

        bus.close.side_effect = close_bus
        web.close.side_effect = lambda: order.append("web")
        camera.close.side_effect = lambda: order.append("camera")
        with patch("tasks.lock_target.signal.signal", side_effect=install_signal), \
                patch("tasks.lock_target.CameraStream", return_value=camera), \
                patch("tasks.lock_target.MotorBus", return_value=bus), \
                patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                patch("tasks.lock_target.local_addresses", return_value=[]), \
                patch("tasks.lock_target.run", side_effect=interrupted_run), patch("builtins.print"):
            self.assertEqual(main([]), 0)
        self.assertEqual(order, ["bus", "web", "camera"])

    def test_web_bind_failure_does_not_open_hardware(self):
        with patch("tasks.lock_target.TargetLockWeb", side_effect=OSError("端口已占用")), \
                patch("tasks.lock_target.CameraStream") as camera, \
                patch("tasks.lock_target.MotorBus") as bus, patch("builtins.print"):
            self.assertEqual(main([]), 1)
        camera.assert_not_called()
        bus.assert_not_called()

    def test_camera_failure_closes_the_bound_web_port(self):
        web = Mock()
        with patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                patch("tasks.lock_target.CameraStream", side_effect=RuntimeError("摄像头异常")), \
                patch("tasks.lock_target.MotorBus") as bus, patch("builtins.print"):
            self.assertEqual(main([]), 1)
        web.close.assert_called_once()
        web.start.assert_not_called()
        bus.assert_not_called()
