"""取消机械臂软件角度限位和舵机目标锁定；只使用模拟设备。"""
import contextlib
import io
import unittest

from arm.api import Arm
from arm.config import ArmConfig
from arm.servo import FeetechSTSServo, SERVO_STEP_PER_DEG
from debug.dry_run import DryArm


class UnrestrictedArmTests(unittest.TestCase):
    def test_all_joints_accept_targets_beyond_previous_lower_and_upper_limits(self):
        arm = Arm(simulate=True)
        self.addCleanup(arm.close)
        with contextlib.redirect_stdout(io.StringIO()):
            arm.calibrate()
            for target in ((-720, -90, -30, -240), (720, 240, 210, 240)):
                with self.subTest(target=target):
                    arm.move_joints(*target)
                    self.assertEqual(arm._arm.get_joints(), dict(enumerate(target, 1)))
                    DryArm().move_joints(*target)
            arm.home()
        self.assertEqual(arm._arm.get_joints(), arm.config.joint_offsets_deg)

    def test_offsets_outside_old_limits_are_valid(self):
        offsets = {1: 720, 2: -90, 3: 210, 4: -240}
        self.assertEqual(ArmConfig(joint_offsets_deg=offsets).joint_offsets_deg, offsets)

    def test_protocol_overflow_is_rejected_before_any_motor_moves(self):
        arm = Arm(simulate=True)
        self.addCleanup(arm.close)
        with contextlib.redirect_stdout(io.StringIO()):
            arm.calibrate()
        initial = arm._arm.get_joints()
        with self.assertRaisesRegex(ValueError, '协议 int32'):
            arm.move_joints(0, 0, 0, 1e12)
        self.assertEqual(arm._arm.get_joints(), initial)
        self.assertTrue(all(not motor.enabled for motor in arm._bus.motors.values()))


class UnrestrictedServoTests(unittest.TestCase):
    def make_servo(self, address):
        servo = FeetechSTSServo(servo_id=address, simulate=True)
        self.addCleanup(servo.close)
        return servo

    def test_tool_calibration_target_does_not_lock_later_movements(self):
        servo = self.make_servo(1)
        servo.hold_fixed_position(1)
        self.assertEqual(servo.read_position(), 340)
        servo.move_angle(120)
        self.assertEqual(servo.read_position(), 1365)
        servo.hold()
        self.assertEqual(servo.read_position(), 1365)
        servo.disable_torque()
        servo.enable_torque()
        self.assertEqual(servo.read_position(), 1365)
        self.assertTrue(servo.status()['torque_enabled'])
        servo.move_to(3000)
        self.assertEqual(servo.read_position(), 3000)

    def test_relative_rotation_is_allowed_for_both_ids_without_angle_guard(self):
        for address in (1, 2):
            with self.subTest(address=address):
                servo = self.make_servo(address)
                servo.move_to(2048)
                expected = 2048
                for delta in (30, -30, 450, -450):
                    expected = (expected - int(delta * SERVO_STEP_PER_DEG)) % 4096
                    servo.move_relative_deg(delta)
                    self.assertEqual(servo.read_position(), expected)

    def test_single_turn_protocol_ranges_and_finite_values_are_preserved(self):
        servo = self.make_servo(1)
        for target in (-1, 4096):
            with self.subTest(target=target), self.assertRaises(ValueError):
                servo.move_to(target)
        for angle in (-1, 361, float('nan')):
            with self.subTest(angle=angle), self.assertRaises(ValueError):
                servo.move_angle(angle)
        self.assertFalse(servo.status()['torque_enabled'])
