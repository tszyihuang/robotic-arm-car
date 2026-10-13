"""用延迟到位反馈验证任务串行等待、夹持停稳、超时及取消。"""
import contextlib
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import main
from arm.api import Arm
from arm.config import ArmConfig
from arm.errors import MotionTimeout
from arm.servo import FeetechSTSServo, GOAL_POSITION, MODE, TORQUE_ENABLE, ServoFault
from base.control import MotionCancelled, check_cancel
from config import ARM, SERVO
from tasks.runner import Step, execute


class Clock:
    def __init__(self):
        self.now = 100.0

    def wait(self, seconds, stop_event=None):
        check_cancel(stop_event)
        # 正数等待至少推进一个时钟刻度，避免浮点尾差让超时测试停在截止时间之前。
        self.now = round(self.now + (max(seconds, 1e-9) if seconds > 0 else 0), 9)
        check_cancel(stop_event)


class BlockingArmTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('arm.controller.time.monotonic', side_effect=lambda: self.clock.now))
        self.stack.enter_context(patch('arm.controller.wait_cancelable', self.clock.wait))
        self.stack.enter_context(patch('arm.servo.wait_cancelable', self.clock.wait))

    def session(self, *, delay=.3, blocked=False):
        arm = Arm(simulate=True)
        self.addCleanup(arm.close)
        with contextlib.redirect_stdout(io.StringIO()):
            arm.calibrate()
        sent = []
        for addr, motor in arm._bus.motors.items():
            state = {'initial': motor.angle, 'target': motor.angle, 'start': None}
            original_read = motor.read_status

            def read(motor=motor, state=state, original_read=original_read):
                fraction = 1 if state['start'] is None else min(1, (self.clock.now - state['start']) / delay)
                if blocked:
                    fraction = 0
                motor.angle = state['initial'] + (state['target'] - state['initial']) * fraction
                status = original_read()
                status['speed_rpm'] = 10 if fraction < 1 and state['target'] != state['initial'] else 0
                return status

            def move(angle, speed=10, accel=None, addr=addr, motor=motor, state=state):
                sent.append((self.clock.now, addr, angle))
                state.update(initial=motor.angle, target=angle, start=self.clock.now)
                motor.enabled = True

            motor.read_status = Mock(side_effect=read)
            motor.move = Mock(side_effect=move)
        return arm, sent

    def test_grab_steps_and_both_home_commands_wait_before_next_instruction(self):
        arm, sent = self.session()
        arm.dry_run = True  # 只跳过原有任务间隔，保留实际控制器的到位等待。
        zero = dict(arm._arm.encoder_zero_deg)
        steps = [Step(1, 'gripper-open', ()),
                 Step(2, 'arm-move', (-128, 78, 118, -50)),
                 Step(3, 'arm-move', (-128, 110, 134, -90)),
                 Step(4, 'gripper-close', ()),
                 Step(5, 'arm-move', (-128, 34, 116, 16)),
                 Step(6, 'arm-home', ()), Step(7, 'home', ())]
        previous_end = self.clock.now
        for step in steps:
            is_gripper = step.command.startswith('gripper-')
            if not is_gripper:
                expected = (dict(enumerate(step.args, 1)) if step.command == 'arm-move'
                            else arm.config.joint_offsets_deg)
                already_there = arm._arm.get_joints() == expected
            result = execute(step, None, arm, None)
            self.assertTrue(result['ok'])
            if is_gripper:
                self.assertGreaterEqual(self.clock.now - previous_end + 1e-9, SERVO['settle_seconds'])
            else:
                self.assertGreaterEqual(self.clock.now - previous_end + 1e-9, .2 if already_there else .5)
                self.assertEqual(arm._arm.get_joints(), expected)
            previous_end = self.clock.now
        self.assertEqual(len(sent), 20)
        self.assertEqual(arm._arm.encoder_zero_deg, zero)

    def test_arrived_arm_continues_despite_id3_speed_feedback(self):
        arm, _ = self.session()
        arm.config.arrival_tolerance_deg = 2
        arm.config.move_timeout = 1
        arm.dry_run = True
        feedback = {1: .18, 2: -.29, 3: -2.82, 4: 0}
        for addr, motor in arm._bus.motors.items():
            original_read = motor.read_status.side_effect

            def read(addr=addr, original_read=original_read):
                status = original_read()
                status['speed_rpm'] = feedback[addr]
                if addr == 3:
                    status['multi_turn_deg'] -= .04
                return status

            motor.read_status.side_effect = read

        execute(Step(1, 'arm-move', (6, 0, 160, 1)), None, arm, None)
        self.assertGreaterEqual(self.clock.now, 100.5)
        self.assertLess(self.clock.now, 101)
        for addr, angle in enumerate((6, 0, 160, 1), 1):
            self.assertAlmostEqual(arm._arm.get_joints()[addr], angle, delta=.05)
        servo = arm._ensure_servo()
        with patch.object(servo, 'move_angle', wraps=servo.move_angle) as close:
            execute(Step(2, 'gripper-close', ()), None, arm, None)
        close.assert_called_once()
        self.assertTrue(all(motor.enabled for motor in arm._bus.motors.values()))

    def test_id2_position_error_uses_configured_arrival_tolerance(self):
        for tolerance, error, finishes in ((None, -3.03, True),
                                           (2, -3.03, False),
                                           (None, -5.03, False)):
            with self.subTest(tolerance=tolerance, error=error):
                arm, _ = self.session()
                arm.config.move_timeout = 1
                arm.dry_run = True
                if tolerance is not None:
                    arm.config.arrival_tolerance_deg = tolerance
                feedback = {1: .46, 2: -.06, 3: .14, 4: -.24}
                for addr, motor in arm._bus.motors.items():
                    original_read = motor.read_status.side_effect

                    def read(addr=addr, original_read=original_read, error=error):
                        status = original_read()
                        status['speed_rpm'] = feedback[addr]
                        if addr == 2:
                            status['multi_turn_deg'] += error
                        return status

                    motor.read_status.side_effect = read

                started = self.clock.now
                step = Step(1, 'arm-move', (-90, 0, 102, 91))
                if finishes:
                    execute(step, None, arm, None)
                    self.assertGreaterEqual(self.clock.now - started + 1e-9, .5)
                    self.assertLess(self.clock.now - started, 1)
                    self.assertAlmostEqual(arm._arm.get_joints()[2], -3.03)
                    self.assertTrue(all(motor.enabled for motor in arm._bus.motors.values()))
                else:
                    with self.assertRaisesRegex(MotionTimeout, 'ID2 目标 0°'):
                        execute(step, None, arm, None)
                    self.assertTrue(all(not motor.enabled for motor in arm._bus.motors.values()))

    def test_optional_speed_check_waits_until_speed_is_below_limit(self):
        arm, _ = self.session()
        arm.config.arrival_speed_rpm = .5
        motor = arm._bus.motors[1]
        original_read = motor.read_status.side_effect

        def read():
            status = original_read()
            if self.clock.now < 100.6:
                status['speed_rpm'] = 1
            return status

        motor.read_status.side_effect = read
        arm.home()
        self.assertGreaterEqual(self.clock.now, 100.8)

    def test_optional_speed_check_times_out_on_persistent_speed_feedback(self):
        arm, _ = self.session()
        arm.config.arrival_speed_rpm = .5
        arm.config.move_timeout = .5
        motor = arm._bus.motors[3]
        original_read = motor.read_status.side_effect

        def read():
            status = original_read()
            status['speed_rpm'] = -2.82
            return status

        motor.read_status.side_effect = read
        with self.assertRaisesRegex(MotionTimeout, r'ID3 目标 160°.*速度 -2.82rpm'):
            arm.home()
        self.assertTrue(all(not motor.enabled for motor in arm._bus.motors.values()))

    def test_leaving_position_tolerance_restarts_arrival_confirmation(self):
        arm, _ = self.session()
        motor = arm._bus.motors[1]
        original_read = motor.read_status.side_effect

        def read():
            status = original_read()
            if 100.1 <= self.clock.now < 100.2:
                status['multi_turn_deg'] += 6
            return status

        motor.read_status.side_effect = read
        arm.home()
        self.assertGreaterEqual(self.clock.now, 100.4)

    def test_timeout_prevents_next_step_and_reports_joint_feedback(self):
        arm, _ = self.session(blocked=True)
        arm.config.move_timeout = .3
        arm.dry_run = True
        servo = arm._ensure_servo()
        with patch.object(servo, 'move_angle', wraps=servo.move_angle) as close:
            with self.assertRaisesRegex(MotionTimeout, 'ID1 目标 -128°，实际 0.00°'):
                for step in [Step(1, 'arm-move', (-128, 78, 118, -50)), Step(2, 'gripper-close', ())]:
                    execute(step, None, arm, None)
        close.assert_not_called()
        self.assertTrue(all(not motor.enabled for motor in arm._bus.motors.values()))

    def test_cancellation_during_arrival_wait_stops_all_joints(self):
        arm, _ = self.session()
        stop = threading.Event()

        def cancel(seconds, stop_event=None):
            stop.set()
            self.clock.wait(seconds, stop_event)

        with patch('arm.controller.wait_cancelable', side_effect=cancel):
            with self.assertRaises(MotionCancelled):
                arm.move_joints(-128, 78, 118, -50, stop_event=stop)
        self.assertTrue(all(not motor.enabled for motor in arm._bus.motors.values()))

    def test_explicit_nonblocking_call_returns_before_motion_completes(self):
        arm, _ = self.session()
        arm.move_joints(-128, 78, 118, -50, wait=False)
        self.assertEqual(self.clock.now, 100)
        self.assertNotEqual(arm._arm.get_joints()[1], -128)

    def test_segmented_move_waits_for_first_group_before_sending_second(self):
        arm, sent = self.session()
        arm._arm.move_joints({1: -128, 2: 78, 3: 118, 4: -50}, order='plane-first')
        self.assertEqual([row[1] for row in sent], [2, 3, 4, 1])
        self.assertGreaterEqual(sent[-1][0] - sent[0][0] + 1e-9, .5)

    def test_invalid_wait_configuration_is_rejected(self):
        for key in ('arrival_tolerance_deg', 'move_timeout', 'arrival_speed_rpm'):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ArmConfig(**{key: 0})
        with self.assertRaises(ValueError):
            ArmConfig(arrival_settle_seconds=-1)

    def test_speed_check_is_disabled_by_default_and_accepts_explicit_none(self):
        self.assertIsNone(ArmConfig().arrival_speed_rpm)
        self.assertIsNone(ArmConfig(arrival_speed_rpm=None).arrival_speed_rpm)
        self.assertEqual(ArmConfig(arrival_speed_rpm=.5).arrival_speed_rpm, .5)


class ToolCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('arm.servo.time.monotonic', side_effect=lambda: self.clock.now))
        self.stack.enter_context(patch('arm.servo.wait_cancelable', self.clock.wait))
        self.stack.enter_context(patch.dict(SERVO['fixed_angles_deg'], {1: 164.0}))
        self.arm = Arm(simulate=True)
        self.arm.dry_run = True
        self.addCleanup(self.arm.close)
        self.arm.close_gripper()
        self.servo = self.arm._servo
        self.gripper_before = self.servo.status()
        self.gripper_target = self.servo._target
        self.started = self.clock.now
        self.status = self.servo.status

    def assert_tool_released_and_gripper_preserved(self):
        self.assertFalse(self.servo.status(servo_id=1)['torque_enabled'])
        self.assertEqual(self.servo.status(), self.gripper_before)
        self.assertEqual(self.servo._target, self.gripper_target)
        self.assertFalse(self.arm.calibrated)

    def test_configured_degrees_match_angle_feedback_and_calibration_waits_for_tool(self):
        def feedback(*, servo_id=None):
            state = self.status(servo_id=servo_id)
            if servo_id == 1 and self.clock.now - self.started < .3:
                state.update(position=1920, moving=True, speed_raw=100)
            return state

        with patch.object(self.servo, 'status', side_effect=feedback), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            execute(Step(1, 'arm-calibrate', ()), None, self.arm, None)
        self.assertGreaterEqual(self.clock.now + 1e-9, self.started + .5)
        self.assertEqual(int.from_bytes(self.servo.read_register(GOAL_POSITION, 2, servo_id=1), 'little'), 1866)
        self.assertAlmostEqual(self.arm.read_servo_angle(1), 164, delta=.05)
        self.assertEqual(self.servo.status(), self.gripper_before)
        self.assertEqual(self.servo._target, self.gripper_target)
        self.assertTrue(self.arm.calibrated)
        self.assertLess(output.getvalue().index('舵机 ID1 已到位'),
                        output.getvalue().index('机械臂已校准'))

    def test_blocked_tool_times_out_without_reporting_calibration_success(self):
        def feedback(*, servo_id=None):
            state = self.status(servo_id=servo_id)
            if servo_id == 1:
                state.update(position=1920, moving=False, speed_raw=0)
            return state

        with patch.dict(SERVO, move_timeout=.4), \
                patch.object(self.servo, 'status', side_effect=feedback), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(MotionTimeout, 'ID1.*目标 1866，实际 1920'):
                execute(Step(1, 'arm-calibrate', ()), None, self.arm, None)
        self.assertNotIn('已校准', output.getvalue())
        self.assertNotIn('已到位', output.getvalue())
        self.assert_tool_released_and_gripper_preserved()

    def test_tool_overload_releases_id1_and_preserves_gripper(self):
        def feedback(*, servo_id=None):
            if servo_id == 1:
                raise ServoFault('舵机 ID1 故障 0x20：过载')
            return self.status(servo_id=servo_id)

        with patch.object(self.servo, 'status', side_effect=feedback):
            with self.assertRaisesRegex(ServoFault, 'ID1.*过载'):
                execute(Step(1, 'arm-calibrate', ()), None, self.arm, None)
        self.assert_tool_released_and_gripper_preserved()

    def test_existing_tool_overload_during_mode_query_also_releases_id1(self):
        self.servo._write_register(TORQUE_ENABLE, b'\x01', servo_id=1)
        read = self.servo.read_register

        def overloaded_mode(address, size=2, *, servo_id=None):
            if address == MODE and servo_id == 1:
                raise ServoFault('舵机 ID1 故障 0x20：过载')
            return read(address, size, servo_id=servo_id)

        with patch.object(self.servo, 'read_register', side_effect=overloaded_mode):
            with self.assertRaisesRegex(ServoFault, 'ID1.*过载'):
                execute(Step(1, 'arm-calibrate', ()), None, self.arm, None)
        self.assert_tool_released_and_gripper_preserved()

    def test_cancellation_during_tool_arrival_wait_releases_id1(self):
        stop = threading.Event()

        def cancel(seconds, stop_event=None):
            stop.set()
            self.clock.wait(seconds, stop_event)

        with patch('arm.servo.wait_cancelable', side_effect=cancel):
            with self.assertRaises(MotionCancelled):
                self.arm.calibrate(hold_tool=True, stop_event=stop)
        self.assert_tool_released_and_gripper_preserved()

    def test_invalid_fixed_angles_cannot_enable_tool(self):
        for angle in (-1, 361, float('nan'), float('inf'), True):
            with self.subTest(angle=angle), patch.dict(SERVO['fixed_angles_deg'], {1: angle}), \
                    patch.object(self.servo, '_write_register', wraps=self.servo._write_register) as write:
                with self.assertRaises(ValueError):
                    self.servo.hold_fixed_position(1, wait=True)
                write.assert_not_called()


class BlockingGripperTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('arm.servo.time.monotonic', side_effect=lambda: self.clock.now))
        self.stack.enter_context(patch('arm.servo.wait_cancelable', self.clock.wait))
        self.servo = FeetechSTSServo(servo_id=2, simulate=True)
        self.addCleanup(self.servo.close)

    def state(self, position, *, moving=False, speed=0):
        return dict(position=position, moving=moving, speed_raw=speed, torque_enabled=True)

    def test_light_waits_for_target_instead_of_accepting_contact(self):
        arm = Arm(simulate=True)
        arm.dry_run = True
        arm._servo = self.servo
        self.addCleanup(arm.close)
        target = round(ARM['light_angle_deg'] * 4095 / 360)

        def status():
            elapsed = self.clock.now - 100
            return self.state(3000 if elapsed < .5 else target)

        with patch.object(self.servo, 'status', side_effect=status):
            execute(Step(1, 'light', ()), None, arm, None)
        self.assertGreaterEqual(self.clock.now + 1e-9, 100.7)
        self.assertEqual(self.servo._target, target)
        self.assertIsNone(arm._bus)

    def test_light_timeout_after_one_second_preserves_torque_and_runs_next_tasks(self):
        for position, moving, speed in ((4025, False, 0), (3000, True, 100)):
            with self.subTest(position=position, moving=moving):
                arm = Arm(simulate=True)
                arm.dry_run = True
                self.addCleanup(arm.close)
                servo = arm._ensure_servo()
                status = servo.status
                target = round(arm.config.light_angle_deg * 4095 / 360)
                started = self.clock.now
                base, vision = Mock(dry_run=True), Mock()

                def feedback():
                    state = status()
                    if servo._target == target:
                        state.update(position=position, moving=moving, speed_raw=speed)
                    return state

                def next_task(distance):
                    self.assertAlmostEqual(self.clock.now - started, 1.0)
                    self.assertEqual(servo._target, target)
                    self.assertTrue(status()['torque_enabled'])

                base.straight.side_effect = next_task
                with tempfile.TemporaryDirectory() as folder:
                    path = Path(folder) / 'tasks.txt'
                    path.write_text('[光]\nlight\nstraight 0.1\ngripper-close\n', encoding='utf-8')
                    with patch.object(servo, 'status', side_effect=feedback), \
                            patch.object(arm, '_stop_after_error', wraps=arm._stop_after_error) as stop, \
                            contextlib.redirect_stdout(io.StringIO()) as output:
                        main.run(base, arm, vision, tasks_path=path, section='光')
                base.straight.assert_called_once_with(.1)
                stop.assert_not_called()
                self.assertIn('1s 内未完成点灯动作，继续下一条指令', output.getvalue())
                self.assertIn('[光 3/3] gripper-close', output.getvalue())
                self.assertEqual(servo._target, round(arm.config.gripper_close_angle_deg * 4095 / 360))

    def test_light_cancellation_still_releases_id2(self):
        stop = threading.Event()
        arm = Arm(simulate=True, stop_event=stop)
        arm._servo = self.servo
        self.addCleanup(arm.close)

        def cancel(seconds, stop_event=None):
            stop.set()
            self.clock.wait(seconds, stop_event)

        with patch.object(self.servo, 'status', return_value=self.state(3000)), \
                patch('arm.servo.wait_cancelable', side_effect=cancel):
            with self.assertRaises(MotionCancelled):
                arm.light()
        self.assertFalse(self.servo.status()['torque_enabled'])

    def test_light_servo_fault_still_raises_and_releases_id2(self):
        arm = Arm(simulate=True)
        arm._servo = self.servo
        self.addCleanup(arm.close)
        with patch.object(self.servo, 'status', side_effect=ServoFault('舵机 ID2 过载')):
            with self.assertRaisesRegex(ServoFault, 'ID2 过载'):
                arm.light()
        self.assertFalse(self.servo.status()['torque_enabled'])

    def test_open_waits_through_initial_idle_until_arrival_and_settling(self):
        def status():
            elapsed = self.clock.now - 100
            if elapsed < .1:
                return self.state(2048)
            if elapsed < .4:
                return self.state(2500, moving=True, speed=100)
            return self.state(3000)

        with patch.object(self.servo, 'status', side_effect=status):
            self.servo.move_to(3000, wait=True)
        self.assertGreaterEqual(self.clock.now + 1e-9, 100.6)
        self.assertTrue(self.servo.status()['torque_enabled'])

    def test_close_can_finish_at_contact_without_reaching_empty_gripper_target(self):
        def status():
            elapsed = self.clock.now - 100
            return (self.state(2200, moving=True, speed=100) if elapsed < .2 else self.state(2500))

        with patch.object(self.servo, 'status', side_effect=status):
            self.servo.move_to(3000, wait=True, allow_contact=True)
        self.assertGreaterEqual(self.clock.now + 1e-9, 100.4)
        self.assertTrue(self.servo.status()['torque_enabled'])

    def test_unmoving_gripper_is_not_mistaken_for_successful_contact(self):
        with patch.object(self.servo, 'status', return_value=self.state(2048)), \
                patch.dict(SERVO, move_timeout=.4):
            with self.assertRaisesRegex(MotionTimeout, '目标 3000，实际 2048'):
                self.servo.move_to(3000, wait=True, allow_contact=True)
        self.assertFalse(self.servo.status()['torque_enabled'])

    def test_unstable_contact_resets_settling_timer(self):
        def status():
            elapsed = self.clock.now - 100
            if elapsed < .1:
                return self.state(2200, moving=True, speed=100)
            if elapsed < .2:
                return self.state(2500)
            if elapsed < .4:
                return self.state(2550, moving=True, speed=100)
            return self.state(2600)

        with patch.object(self.servo, 'status', side_effect=status):
            self.servo.move_to(3000, wait=True, allow_contact=True)
        self.assertGreaterEqual(self.clock.now + 1e-9, 100.6)

    def test_moving_flag_without_encoder_progress_is_not_contact_completion(self):
        def status():
            return self.state(2048, moving=self.clock.now < 100.1)

        with patch.object(self.servo, 'status', side_effect=status), \
                patch.dict(SERVO, move_timeout=.4):
            with self.assertRaises(MotionTimeout):
                self.servo.move_to(3000, wait=True, allow_contact=True)

    def test_motion_away_from_goal_is_not_contact_completion(self):
        with patch.object(self.servo, 'status', return_value=self.state(1800)), \
                patch.dict(SERVO, move_timeout=.4):
            with self.assertRaises(MotionTimeout):
                self.servo.move_to(3000, wait=True, allow_contact=True)

    def test_cancel_during_gripper_wait_disables_torque(self):
        stop = threading.Event()

        def cancel(seconds, stop_event=None):
            stop.set()
            self.clock.wait(seconds, stop_event)

        with patch.object(self.servo, 'status', return_value=self.state(2200, moving=True, speed=100)), \
                patch('arm.servo.wait_cancelable', side_effect=cancel):
            with self.assertRaises(MotionCancelled):
                self.servo.move_to(3000, wait=True, stop_event=stop)
        self.assertFalse(self.servo.status()['torque_enabled'])
