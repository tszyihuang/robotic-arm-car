"""彩色靶锁定、PID 方向和模拟两轴闭环；不连接硬件。"""
from contextlib import ExitStack
import signal
import struct
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from arm.config import ArmConfig
from arm.motor import ANGLE_SCALE, SimulatedBus, position_payload
from base.control import MotionCancelled
from config import ARM
from tasks.lock_target import PID, TargetLock, main, run, validate_config
from vision.targets import colored_targets


def target(color, x, y):
    return {"value": color, "name": color, "box": [x - 10, y - 10, x + 10, y + 10],
            "center_x": x}


def image(color="red", x=220, y=80):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    if color is not None:
        bgr = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}[color]
        frame[round(y) - 10:round(y) + 10, round(x) - 10:round(x) + 10] = bgr
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
    def test_default_selects_closest_target_to_image_center(self):
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
        lock = TargetLock()
        lock.step([target("red", 600, 400)], (1000, 600), 0.05)
        result = lock.step([target("red", 504, 390)], (1000, 600), 0.05)
        self.assertEqual(result["rates"][1], 0)
        self.assertIsNone(lock.pids[1].previous)
        self.assertGreater(result["rates"][4], 0)

    def test_calibrated_aim_controls_directions_and_tolerance(self):
        lock = TargetLock()
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
                    {"min_area_ratio": 1}, {"match_distance_ratio": 1}, {"unknown": 1},
                    {"motor2_angle_deg": float("nan")}, {"motor2_angle_deg": 1e12}):
            with self.subTest(cfg=cfg), self.assertRaises(ValueError):
                validate_config(cfg)
        with self.assertRaises(ValueError):
            TargetLock("invalid")
        with self.assertRaises(ValueError):
            TargetLock().step([], (320, 240), 0)


class TargetLockRunTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("tasks.lock_target.time.monotonic", self.clock.monotonic))
        self.stack.enter_context(patch("tasks.lock_target.time.time", self.clock.wall))
        self.stack.enter_context(patch("tasks.lock_target.wait_cancelable", self.clock.wait))
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

    def test_real_color_detection_and_simulated_motors_converge_to_center(self):
        camera = Mock()
        count = 0

        def next_frame(**kwargs):
            nonlocal count
            count += 1
            self.assertEqual(self.bus.motors[2].angle, 12.0)
            self.assertTrue(self.bus.motors[2].enabled)
            x = 160 + (2.0 - self.bus.motors[1].angle) * 50
            y = 120 + (-1.5 - self.bus.motors[4].angle) * 40
            return image(x=x, y=y), count, self.clock.wall()

        camera.next_frame.side_effect = next_frame
        run(camera, self.bus, duration=3, config={"tolerance_px": 2}, log=lambda message: None)
        self.assertAlmostEqual(self.bus.motors[1].angle, 2.0, delta=0.07)
        self.assertAlmostEqual(self.bus.motors[4].angle, -1.5, delta=0.09)
        self.bus.motors[2].move.assert_called_once_with(12.0, speed_rpm=ARM["speed_rpm"])
        self.bus.motors[3].move.assert_not_called()
        self.assertFalse(self.bus.motors[3].enabled)

    def test_motor2_holds_12_degrees_with_no_target_and_after_exit(self):
        self.bus.motors[2].angle = 35.0
        camera = self.scripted_camera([(image(None), 1, 0), (TimeoutError(), 2, 0)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, log=lambda message: None)
        self.bus.motors[2].move.assert_called_once_with(12.0, speed_rpm=ARM["speed_rpm"])
        self.assertEqual(self.bus.motors[2].angle, 12.0)
        self.assertTrue(self.bus.motors[2].enabled)

    def test_live_aim_changes_reach_motor_commands_on_the_next_frame(self):
        aim = Mock()
        aim.point.side_effect = [(0.5, 0.5), (0.8, 0.2)]
        updates = Mock()
        camera = self.scripted_camera([(image(), 1, 0), (image(), 2, 0)])
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
        camera = Mock()
        with self.assertRaisesRegex(OSError, "ID2 串口错误"):
            run(camera, self.bus, log=lambda message: None)
        camera.next_frame.assert_not_called()
        for addr in (1, 4):
            self.bus.motors[addr].move.assert_called_once_with(0.0, speed_rpm=0.01)

    def test_missing_target_holds_both_axes_and_does_not_switch_color(self):
        camera = self.scripted_camera([(image(), 1, 0), (image(None), 2, 0),
                                      (image("green"), 3, 0), (image(), 4, 0)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, log=lambda message: None)
        for addr, sign in ((1, 1), (4, -1)):
            angles = [call.args[0] for call in self.bus.motors[addr].move.call_args_list]
            self.assertEqual(len(angles), 4)  # 移动、缺靶保持、恢复移动、退出保持。
            self.assertGreater(angles[0] * sign, 0)
            self.assertEqual(angles[0], angles[1])
            self.assertGreater((angles[2] - angles[1]) * sign, 0)
            self.assertEqual(angles[2], angles[3])

    def test_duplicate_stale_future_and_timed_out_frames_never_drive_motion(self):
        camera = self.scripted_camera([(image(), 1, 0), (image(), 1, 0), (image(), 2, 0.8),
                                      (image(), 3, -1), (TimeoutError(), 4, 0)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, log=lambda message: None)
        for addr in (1, 4):
            angles = [call.args[0] for call in self.bus.motors[addr].move.call_args_list]
            self.assertEqual(len(angles), 3)
            self.assertTrue(all(angle == angles[0] for angle in angles))
        self.assertEqual(camera.next_frame.call_args_list[1].kwargs["after"], 1)

    def test_delayed_feedback_makes_the_frame_expire_before_motion(self):
        original = self.bus.motors[1].read_status

        def delayed():
            self.clock.now += 0.7
            return original()

        self.bus.motors[1].read_status = delayed
        camera = self.scripted_camera([(image(), 1, 0)])
        run(camera, self.bus, duration=0.1, log=lambda message: None)
        for addr in (1, 4):
            self.bus.motors[addr].move.assert_called_once_with(0.0, speed_rpm=0.01)

    def test_expired_detection_does_not_latch_the_initial_target(self):
        camera = self.scripted_camera([(image(), 1, 0), (image("green"), 2, 0)])
        first = True

        def slow_detection(frame, min_area_ratio):
            nonlocal first
            if first:
                self.clock.now += 0.7
                first = False
            return colored_targets(frame, min_area_ratio)

        log = Mock()
        with patch("tasks.lock_target.colored_targets", side_effect=slow_detection), \
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
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[340:380, 629:669] = (0, 0, 255)  # 横向只偏 9px，略超默认容差。
        camera = self.scripted_camera([(frame, i, 0) for i in range(1, 4)])
        with self.assertRaises(KeyboardInterrupt):
            run(camera, self.bus, config={"ki": 0, "kd": 0}, log=lambda message: None)
        self.assertGreater(self.bus.motors[1].angle, 0)
        angles = [call.args[0] for call in self.bus.motors[1].move.call_args_list]
        self.assertTrue(all(later > earlier for earlier, later in zip([0] + angles[:2], angles[:3])))

    def test_slow_motor_targets_use_feedback_and_respect_the_step_limit(self):
        for addr in (1, 4):
            self.bus.motors[addr].move = Mock()  # 反馈不变，模拟目标尚未到达。
        camera = self.scripted_camera([(image(), i, 0) for i in range(1, 5)])
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
        camera = self.scripted_camera([(image(), 1, 0)])
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

    def test_cli_ctrl_c_disables_all_four_motors_after_holding(self):
        # 模拟启动前 ID3 已经使能，退出应释放四轴，包括保持 12° 的 ID2。
        self.bus.motors[3].enabled = True
        camera = self.scripted_camera([(image(), 1, 0)])
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
