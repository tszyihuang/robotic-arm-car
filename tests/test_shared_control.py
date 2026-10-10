"""共用测速和机械臂控制的行为回归；只使用模拟设备。"""
import contextlib
import io
import threading
import unittest
from unittest.mock import Mock, patch

from arm.api import Arm
from arm.config import ArmConfig
from arm.controller import Arm as JointController
from arm.motor import SimulatedBus
from arm.servo import MODE, TORQUE_ENABLE, SERVO_FIXED_POSITIONS
from base.control import MotionCancelled
from base.feedback import WheelOdometry
from config import BASE, STRAIGHT
from tasks.runner import Step, execute


class OdometryTests(unittest.TestCase):
    def test_nonzero_origin_and_reversed_encoder_signs(self):
        with patch.dict(STRAIGHT, FORWARD_SIGN=(-1, 1, -1, 1)):
            wheels = WheelOdometry([-1000, 1000, -2000, 2000])
            left, right = wheels.travel([-1100, 1120, -2080, 2090])
        scale = BASE["meters_per_count"] * 1000
        self.assertAlmostEqual(left, 110 * scale)
        self.assertAlmostEqual(right, 85 * scale)

    def test_speed_window_preserves_missing_increment_and_discards_old_samples(self):
        wheels = WheelOdometry([0] * 4)
        self.assertEqual(wheels.speed(None), (0, 0))
        self.assertFalse(wheels.has_speed)
        for counts in (10, 20, 30, 40):
            speed = wheels.speed([counts, counts, counts * 2, counts * 2])
        scale = BASE["meters_per_count"] * 1000 / STRAIGHT["TEP_WINDOW"]
        self.assertAlmostEqual(speed[0], 30 * scale)
        self.assertAlmostEqual(speed[1], 60 * scale)
        self.assertEqual(wheels.speed(None), speed)
        self.assertTrue(wheels.has_speed)

    def test_origin_read_checks_cancellation_after_feedback(self):
        stop = threading.Event()
        board = Mock()

        def feedback(timeout):
            stop.set()
            return [100] * 4, None

        board.feedback.side_effect = feedback
        with self.assertRaises(MotionCancelled):
            WheelOdometry.read_origin(board, stop)


class ArmControlTests(unittest.TestCase):
    def test_task_calibration_holds_tool_and_reuses_gripper_connection(self):
        for gripper_first in (False, True):
            with self.subTest(gripper_first=gripper_first), contextlib.redirect_stdout(io.StringIO()):
                session = Arm(simulate=True)
                self.addCleanup(session.close)
                session.dry_run = True
                if gripper_first:
                    session.close_gripper()
                servo = session._ensure_servo()
                gripper_before = servo.status()
                gripper_target = servo._target
                for _ in range(2):
                    execute(Step(2, 'arm-calibrate', ()), None, session, None)
                    self.assertTrue(session.calibrated)
                    self.assertIs(session._servo, servo)
                    self.assertIs(session._arm._gripper, servo)
                    tool = servo.status(servo_id=1)
                    self.assertEqual(tool['position'], SERVO_FIXED_POSITIONS[1])
                    self.assertTrue(tool['torque_enabled'])
                    self.assertEqual(servo.status(), gripper_before)
                    self.assertEqual(servo._target, gripper_target)
                    self.assertTrue(all(not motor.enabled for motor in session._bus.motors.values()))
                execute(Step(3, 'arm-disable', ()), None, session, None)
                self.assertTrue(servo.status(servo_id=1)['torque_enabled'])
                session.close()
                self.assertEqual(servo._ser.memories[1][TORQUE_ENABLE], 1)

    def test_default_calibration_and_angle_reads_do_not_enable_tool(self):
        session = Arm(simulate=True)
        self.addCleanup(session.close)
        before = session.angles()['servos']['1']['angle_deg']
        with contextlib.redirect_stdout(io.StringIO()):
            session.calibrate()
        self.assertEqual(session.angles()['servos']['1']['angle_deg'], before)
        self.assertFalse(session._servo.status(servo_id=1)['torque_enabled'])

    def test_invalid_tool_mode_fails_task_calibration_without_moving_servos(self):
        session = Arm(simulate=True)
        session.dry_run = True
        self.addCleanup(session.close)
        session.close_gripper()
        servo = session._servo
        before = servo.status()
        servo._write_register(MODE, b'\x01', servo_id=1)
        with patch.object(servo, '_write_register', wraps=servo._write_register) as write:
            with self.assertRaisesRegex(ValueError, 'ID1.*mode=0'):
                execute(Step(2, 'arm-calibrate', ()), None, session, None)
        write.assert_not_called()
        self.assertFalse(session.calibrated)
        self.assertEqual(servo.status(), before)

    def test_tool_write_failure_releases_only_tool_and_fails_calibration(self):
        session = Arm(simulate=True)
        session.dry_run = True
        self.addCleanup(session.close)
        session.close_gripper()
        servo = session._servo
        gripper_before = servo.status()
        write = servo._write_register

        def fail_tool_enable(address, data, *, servo_id=None):
            if address == TORQUE_ENABLE and data == b'\x01' and servo_id == 1:
                raise OSError('ID1 使能失败')
            return write(address, data, servo_id=servo_id)

        with patch.object(servo, '_write_register', side_effect=fail_tool_enable):
            with self.assertRaisesRegex(OSError, 'ID1 使能失败'):
                execute(Step(2, 'arm-calibrate', ()), None, session, None)
        self.assertFalse(session.calibrated)
        self.assertFalse(servo.status(servo_id=1)['torque_enabled'])
        self.assertEqual(servo.status(), gripper_before)

    def test_joint_and_gripper_movement_preserves_each_order(self):
        target = {1: 90, 2: 82, 3: 142, 4: -56}
        for order, expected in (("together", [1, 2, 3, 4]),
                                ("plane-first", [2, 3, 4, 1]),
                                ("base-first", [1, 2, 3, 4])):
            with self.subTest(order=order):
                arm = JointController(simulate=True)
                self.addCleanup(arm.close)
                servo = arm._connect_gripper()
                servo.move_angle = Mock(wraps=servo.move_angle)
                servo.wait_for_arrival = Mock()
                sent = []
                for addr, motor in arm._bus.motors.items():
                    move = motor.move

                    def record(*args, addr=addr, move=move):
                        sent.append(addr)
                        return move(*args)

                    motor.move = record
                arm.move_joints(target, order=order, gripper="open")
                self.assertEqual(sent, expected)
                self.assertEqual(arm.get_joints(), target)
                servo.move_angle.assert_called_once_with(291.0, wait=False)
                servo.wait_for_arrival.assert_not_called()

    def test_invalid_gripper_mode_cannot_start_joints(self):
        arm = JointController(simulate=True)
        self.addCleanup(arm.close)
        arm._gripper = Mock()
        arm._gripper.read_register.return_value = b"\x01"
        with self.assertRaisesRegex(ValueError, "mode=0"):
            arm.move_joints({1: 90, 2: 82, 3: 142, 4: -56}, gripper="close")
        self.assertTrue(all(not motor.enabled for motor in arm._bus.motors.values()))
        arm._gripper.move_angle.assert_not_called()

    def test_shared_gripper_is_closed_once_before_or_after_calibration(self):
        for gripper_first in (True, False):
            with self.subTest(gripper_first=gripper_first), contextlib.redirect_stdout(io.StringIO()):
                session = Arm(simulate=True)
                if gripper_first:
                    session.open_gripper()
                session.calibrate()
                session.close_gripper()
                servo = session._servo
                servo.close = Mock(wraps=servo.close)
                session.close()
                session.close()
                servo.close.assert_called_once()

    def test_uncalibrated_disable_reuses_read_only_bus(self):
        session = Arm(simulate=True)
        self.addCleanup(session.close)
        bus = session._ensure_bus()
        bus.disable_all = Mock(wraps=bus.disable_all)
        with contextlib.redirect_stdout(io.StringIO()):
            session.disable()
        self.assertIs(session._bus, bus)
        bus.disable_all.assert_called_once()
        self.assertFalse(bus.closed)

    def test_cancelled_initial_calibration_closes_bus_and_can_retry(self):
        stop = threading.Event()
        session = Arm(simulate=True, stop_event=stop)
        self.addCleanup(session.close)
        bus = SimulatedBus(ArmConfig())
        read = bus.motors[2].read_status

        def cancel_during_read():
            stop.set()
            return read()

        bus.motors[2].read_status = cancel_during_read
        bus.motors[3].read_status = Mock(wraps=bus.motors[3].read_status)
        session._bus = bus
        with self.assertRaises(MotionCancelled):
            session.calibrate()
        self.assertTrue(bus.closed)
        self.assertIsNone(session._bus)
        self.assertFalse(session.calibrated)
        bus.motors[3].read_status.assert_not_called()
        stop.clear()
        with contextlib.redirect_stdout(io.StringIO()):
            session.calibrate()
        self.assertTrue(session.calibrated)
        self.assertIsNot(session._bus, bus)
