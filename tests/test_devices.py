"""直接采集反馈、设备复用及异常停车；全部使用内存反馈或假串口。"""
import contextlib
import io
import struct
import threading
import unittest
from unittest.mock import Mock, patch

from arm.api import Arm
from base.api import Base
from base.motor import Motor, SerialBoard
from base import straight_pid, arc_turn, calibrate_position
from sensor.imu import ImuLink, FUNC_RAW, FUNC_QUAT
from vision.api import Vision
from base.control import MotionCancelled
from tasks import pause


class DeviceConstructionTests(unittest.TestCase):
    def test_arm_step_gap_observes_shared_cancellation_event(self):
        stop = threading.Event()
        arm = Arm(stop_event=stop)
        stop.set()
        with self.assertRaises(MotionCancelled):
            pause(arm)
        self.assertIsNone(arm._bus)

    def test_devices_do_not_connect_during_construction(self):
        with patch("serial.Serial", side_effect=AssertionError("不应开串口")), \
                patch("vision.camera.CameraStream", side_effect=AssertionError("不应开摄像头")):
            base, arm, vision = Base(), Arm(), Vision()
            self.assertIsNone(base.motor)
            self.assertIsNone(arm._bus)
            self.assertIsNone(vision.camera)
            base.close()
            arm.close()
            vision.close()

    def test_base_reuses_motor_and_imu_and_failure_stops(self):
        base = Base()
        base.motor, base.imu, base._imu_checked = Mock(), Mock(), True
        base.imu.has_data.return_value = True
        with patch.object(straight_pid, "straight", return_value=(300, {"reason": ""})), \
                patch.object(arc_turn, "turn_with_radius", return_value={"ok": True}) as turn:
            base.straight(0.3)
            base.turn(87, 0.38)
            base.straight(0.3)
        self.assertIs(turn.call_args.kwargs["board"], base.motor)
        self.assertIs(turn.call_args.kwargs["imu"], base.imu)
        base.imu.close.assert_not_called()
        base.motor.close.assert_not_called()
        base.close()

    def test_partial_motor_startup_failure_closes_serial(self):
        serial_board = Mock()
        serial_board.upload.side_effect = OSError("上传指令失败")
        with patch("base.motor.SerialBoard", return_value=serial_board), self.assertRaises(OSError):
            Motor("/fake")
        serial_board.close.assert_called_once()

    def test_arm_invalid_angles_do_not_start_movement(self):
        arm = Arm(simulate=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                arm.calibrate()
            before = arm._arm.get_joints()
            with self.assertRaises(ValueError):
                arm.move_joints(0, 0, 170, 24)
            self.assertEqual(arm._arm.get_joints(), before)
            self.assertTrue(all(not m.enabled for m in arm._bus.motors.values()))
        finally:
            arm.close()


class EncoderTests(unittest.TestCase):
    def setUp(self):
        # 不运行串口构造与采集线程，只向实际反馈方法注入采集结果。
        self.motor = Motor.__new__(Motor)
        self.motor.condition = threading.Condition()
        self.motor.stop_event = threading.Event()
        self.motor.error = None
        self.motor.sequence = self.motor._last_sequence = 0
        self.motor._totals = self.motor._increments = None
        self.motor.capture_stamp = 0.0
        self.motor.counts = None
        self.motor.stamp = 0.0

    def test_encoder_increment_consumed_once_and_duplicate_does_not_refresh(self):
        with patch("base.motor.time.monotonic", return_value=100.1):
            self.motor._store_feedback([20] * 4, [1] * 4, 100.0)
            self.assertEqual(self.motor.feedback(0), ([20] * 4, [1] * 4))
            self.motor._store_feedback([20] * 4, [1] * 4, 100.0)
            self.assertEqual(self.motor.feedback(0), (None, None))
        with patch("base.motor.time.monotonic", return_value=101.0):
            self.assertEqual(self.motor.feedback(0), (None, None))

    def test_stale_queued_feedback_is_not_new(self):
        self.motor._store_feedback([20] * 4, [1] * 4, 100.0)
        with patch("base.motor.time.monotonic", return_value=101.0):
            self.assertEqual(self.motor.feedback(0), (None, None))

    def test_poll_preserves_capture_time_and_failure_is_visible(self):
        self.motor._store_feedback([20] * 4, None, 100.0)
        with patch("base.motor.time.monotonic", return_value=100.1):
            self.motor.poll()
            self.assertEqual(self.motor.stamp, 100.0)
        self.motor.error = OSError("串口读取失败")
        with self.assertRaisesRegex(RuntimeError, "串口读取失败"):
            self.motor.feedback(0)

    def test_protocol_discards_old_increment_and_keeps_partial_frames(self):
        board = SerialBoard.__new__(SerialBoard)
        board._feedback_buf, board.battery = b"", None
        board.ser = Mock()
        board.ser.read.side_effect = [b"$MAll:1,2,3,4#$MTEP:5,6,7,8#$MAll:9,", b"",
                                     b"10,11,12#", b""]
        self.assertEqual(board.feedback(0.01), ([1, 2, 3, 4], [5, 6, 7, 8]))
        self.assertEqual(board.feedback(0.01), ([9, 10, 11, 12], None))


class ImuTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.clock = patch("sensor.imu.time.time", side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        with patch("serial.Serial"):
            self.imu = ImuLink("/fake")

    def raw(self, acc):
        self.imu._handle(FUNC_RAW, struct.pack("<9h", *acc, 0, 0, 0, 0, 0, 0))

    def quat(self):
        self.imu._handle(FUNC_QUAT, struct.pack("<4f", 1, 0, 0, 0))

    def test_quaternion_cannot_renew_acceleration_watchdog(self):
        self.raw((0, 0, 0))
        self.quat()
        self.now = 101.0
        self.quat()
        self.assertTrue(self.imu.has_data())
        self.assertGreater(self.imu.accel_state().age, 0.3)
        with self.assertRaises(RuntimeError):
            self.imu.arm_impact()

    def test_single_raw_impact_latches_after_calm_frames(self):
        self.raw((0, 0, 0))
        self.imu.arm_impact()
        self.now += 0.01
        self.raw((151, 0, 0))
        self.now += 0.01
        self.raw((151, 0, 0))
        self.assertTrue(self.imu.accel_state().hit)
        self.assertEqual(self.imu.accel_state().peak, 151)
        self.assertEqual(self.imu.accel_state().frames, 3)

    def test_yaw_expires_without_new_quaternion(self):
        self.quat()
        self.now += 0.4
        self.raw((0, 0, 0))
        self.assertFalse(self.imu.has_data())


class CleanupTests(unittest.TestCase):
    def test_controller_error_and_cancel_keep_original_when_stop_fails(self):
        for original in (KeyboardInterrupt(), OSError("反馈断开"), MotionCancelled("取消")):
            board = Mock()
            board.feedback.side_effect = original
            board.stop.side_effect = OSError("停车指令失败")
            with contextlib.redirect_stderr(io.StringIO()) as output, self.assertRaises(type(original)) as failed:
                straight_pid.straight(board, 300, 150)
            self.assertIs(failed.exception, original)
            self.assertIn("停车指令失败", output.getvalue())

    def test_arc_stops_and_releases_without_closing_shared_devices(self):
        board, imu = Mock(), Mock()
        board.poll.side_effect = OSError("读数断开")
        with self.assertRaises(OSError):
            arc_turn.turn_with_radius(0.38, 87, board=board, imu=imu, verbose=False)
        board.stop.assert_called_once()
        board.release.assert_called_once()
        board.close.assert_not_called()
        imu.close.assert_not_called()

    def test_position_cancel_attempts_release_and_disarm_even_if_stop_fails(self):
        board, imu = Mock(), Mock()
        stop = threading.Event()
        stop.set()
        board.stop.side_effect = OSError("停车失败")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(MotionCancelled):
            calibrate_position.calibrate_position(board, imu, stop_event=stop)
        board.release.assert_called_once()
        imu.disarm_impact.assert_called_once()
