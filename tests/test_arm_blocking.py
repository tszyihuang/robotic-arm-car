"""用延迟到位反馈验证任务串行等待、夹持停稳、超时及取消。"""
import contextlib
import io
import threading
import unittest
from unittest.mock import Mock, patch

from arm.api import Arm
from arm.config import ArmConfig
from arm.errors import MotionTimeout
from arm.servo import FeetechSTSServo
from base.control import MotionCancelled, check_cancel
from config import SERVO
from tasks.runner import Step, execute


class Clock:
    def __init__(self):
        self.now = 100.0

    def wait(self, seconds, stop_event=None):
        check_cancel(stop_event)
        self.now = round(self.now + seconds, 9)
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

    def test_target_position_without_stopped_speed_does_not_finish(self):
        arm, _ = self.session()
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
