"""直接采集反馈、设备复用及异常停车；全部使用内存反馈或假串口。"""
import contextlib
import io
import struct
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, call, patch

import serial

from arm.api import Arm
from base.api import Base
from base.motor import Motor, SerialBoard
from base import straight_pid, arc_turn, calibrate_position
from sensor.imu import ImuLink, FUNC_RAW, FUNC_QUAT
from vision.api import Vision
from base.control import MotionCancelled
from config import TURN, VISION_CONTROL
from tasks import pause


class DeviceConstructionTests(unittest.TestCase):
    def test_base_close_can_keep_zero_speed_without_releasing_pwm(self):
        for release_motors in (False, True):
            with self.subTest(release_motors=release_motors):
                base = Base()
                motor = Motor.__new__(Motor)
                motor.shutdown = threading.Event()
                motor.stop_event = threading.Event()
                motor.lock = threading.RLock()
                motor.thread = None
                motor._released = False
                motor.board = Mock()
                base.motor = motor
                base.close(release_motors=release_motors)
                self.assertTrue(motor.shutdown.is_set())
                motor.board.spd.assert_called_once_with(0, 0, 0, 0)
                motor.board.ser.close.assert_called_once()
                if release_motors:
                    motor.board._send.assert_called_once_with('pwm:0,0,0,0')
                else:
                    motor.board._send.assert_not_called()

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

    def test_movement_distance_error_does_not_block_following_commands(self):
        base = Base()
        base.motor, base._imu_checked = Mock(), True
        heading, vision = Mock(), Mock()
        with patch.object(straight_pid, "straight", return_value=(20, {"reason": ""})), \
                patch.object(base, "_heading", return_value=heading), \
                patch("base.vision_straight.vision_straight", return_value=(900, {"reason": ""})), \
                patch.dict(VISION_CONTROL, LOG_DIR=None):
            forward = base.straight(0.3)
            backward = base.straight(-0.3)
            visual = base.vision_straight(0.3, vision)
        self.assertTrue(all(result["ok"] for result in (forward, backward, visual)))
        self.assertEqual(forward["distance_m"], 0.02)
        self.assertEqual(backward["distance_m"], -0.02)
        self.assertEqual(visual["distance_m"], 0.9)

    def test_movement_feedback_faults_still_stop_task(self):
        base = Base()
        base.motor, base._imu_checked = Mock(), True
        with patch.object(straight_pid, "straight", return_value=(20, {"reason": "编码器数据中断"})):
            with self.assertRaisesRegex(RuntimeError, "编码器数据中断"):
                base.straight(0.3)
        with patch.object(arc_turn, "turn_with_radius", return_value={"ok": False, "fault": "IMU 数据中断"}):
            with self.assertRaisesRegex(RuntimeError, "IMU 数据中断"):
                base.turn(87, 0.38)
        with patch.object(base, "_heading", return_value=Mock()), \
                patch("base.vision_straight.vision_straight", return_value=(20, {"reason": "编码器数据中断"})), \
                patch.dict(VISION_CONTROL, LOG_DIR=None):
            with self.assertRaisesRegex(RuntimeError, "编码器数据中断"):
                base.vision_straight(0.3, Mock())

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
                arm.move_joints(0, 0, float('nan'), 24)
            self.assertEqual(arm._arm.get_joints(), before)
            self.assertTrue(all(not m.enabled for m in arm._bus.motors.values()))
        finally:
            arm.close()


class SerialWriteTests(unittest.TestCase):
    def test_timed_out_speed_command_preserves_error_and_is_not_replayed(self):
        board = SerialBoard.__new__(SerialBoard)
        board.ser = Mock(port="/fake", write_timeout=1.0)
        original = serial.SerialTimeoutException("Write timeout")
        board.ser.write.side_effect = original
        with self.assertRaises(serial.SerialTimeoutException) as failed:
            board.spd(100, 100, 200, 200)
        self.assertIs(failed.exception.__cause__, original)
        self.assertIn("/fake", str(failed.exception))
        self.assertIn("write_timeout=1.0s", str(failed.exception))
        self.assertIn("spd:100,100,200,200", str(failed.exception))
        board.ser.write.assert_called_once_with(b"$spd:100,100,200,200#\r\n")
        board.ser.flush.assert_not_called()

    def test_partial_speed_command_is_rejected_without_replay(self):
        board = SerialBoard.__new__(SerialBoard)
        board.ser = Mock()
        board.ser.write.side_effect = lambda data: len(data) - 1
        with self.assertRaisesRegex(RuntimeError, "电机指令未完整发送"):
            board.spd(100, 100, 200, 200)
        board.ser.write.assert_called_once()
        board.ser.flush.assert_not_called()


class MotorOutputTests(unittest.TestCase):
    def setUp(self):
        self.motor = Motor.__new__(Motor)
        self.motor.shutdown = threading.Event()
        self.motor.stop_event = threading.Event()
        self.motor.lock = threading.Lock()
        self.motor.thread = None
        self.motor._released = False
        self.motor.default_timeout = 0.08
        self.motor.board = Mock()

    def test_release_survives_task_stop_and_connection_close(self):
        base = Base()
        base.motor = self.motor
        self.motor.release()
        base.stop()
        base.close(release_motors=False)
        self.motor.board.spd.assert_not_called()
        self.assertTrue(self.motor._released)
        self.assertEqual(self.motor.board._send.call_args_list,
                         [call("pwm:0,0,0,0")] * 3)
        self.motor.board.ser.close.assert_called_once()

    def test_next_movement_restores_drive_and_zero_speed_hold(self):
        self.motor.release()
        self.motor.speed((0.1, 0.1, 0.2, 0.2))
        self.assertFalse(self.motor._released)
        self.motor.stop()
        self.motor.close(release_motors=False)
        self.assertEqual(self.motor.board.spd.call_args_list,
                         [call(100.0, 100.0, 200.0, 200.0), call(0, 0, 0, 0), call(0, 0, 0, 0)])
        self.motor.board._send.assert_called_once_with("pwm:0,0,0,0")

    def test_position_impact_releases_wheels_through_task_cleanup(self):
        base = Base()
        base.motor, base.imu, base._imu_checked = self.motor, Mock(), True
        self.motor.feedback = Mock(side_effect=[([0] * 4, None), ([-1] * 4, [-1] * 4)])
        fresh = SimpleNamespace(age=0.0, hit=False, peak=0.0, acc=(0, 0, 0), delta=0.0)
        impact = SimpleNamespace(age=0.0, hit=True, peak=200.0, acc=(200, 0, 0), delta=200.0)
        base.imu.accel_state.side_effect = [fresh, fresh, fresh, impact]
        with patch("base.calibrate_position.wait_cancelable"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = base.calibrate_position()
        self.assertTrue(result["ok"])
        self.assertIn("失能底盘四轮", output.getvalue())
        self.assertTrue(self.motor._released)
        self.assertTrue(any(any(value < 0 for value in command.args)
                            for command in self.motor.board.spd.call_args_list))
        base.stop()
        base.close(release_motors=False)
        commands = self.motor.board.mock_calls
        released_at = commands.index(call._send("pwm:0,0,0,0"))
        self.assertFalse(any(command[0] == "spd" for command in commands[released_at:]))


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
        self.motor._increment_stamp = self.motor._last_increment_stamp = 0.0
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

    def test_protocol_preserves_speed_only_frame_and_partial_totals(self):
        board = SerialBoard.__new__(SerialBoard)
        board._feedback_buf, board.battery = b"", None
        board.ser = Mock()
        board.ser.read.side_effect = [b"$MTEP:5,6,7,8#$MAll:9,", b"",
                                     b"10,11,12#", b""]
        self.assertEqual(board.feedback(0.01), (None, [5, 6, 7, 8]))
        self.assertEqual(board.feedback(0.01), ([9, 10, 11, 12], None))

    def test_split_speed_frame_survives_totals_updates_and_is_consumed_once(self):
        with patch("base.motor.time.monotonic", return_value=100.1):
            self.motor._store_feedback([20] * 4, None, 100.0)
            self.assertEqual(self.motor.feedback(0), ([20] * 4, None))
            self.motor._store_feedback(None, [1] * 4, 100.01)
            self.motor._store_feedback([21] * 4, None, 100.02)
            self.motor._store_feedback([22] * 4, None, 100.03)
            self.assertEqual(self.motor.feedback(0), ([22] * 4, [1] * 4))
            self.motor._store_feedback([23] * 4, None, 100.04)
            self.assertEqual(self.motor.feedback(0), ([23] * 4, None))

    def test_new_totals_cannot_refresh_stale_speed(self):
        self.motor._store_feedback(None, [1] * 4, 100.0)
        self.motor._store_feedback([20] * 4, None, 100.4)
        with patch("base.motor.time.monotonic", return_value=100.4):
            self.assertEqual(self.motor.feedback(0), ([20] * 4, None))

    def test_reader_stores_speed_only_packets(self):
        self.motor.shutdown = threading.Event()
        self.motor.lock = threading.Lock()
        self.motor.board = Mock()

        def feedback(timeout):
            self.motor.shutdown.set()
            return None, [1] * 4

        self.motor.board.feedback.side_effect = feedback
        with patch("base.motor.time.monotonic", return_value=100.0):
            self.motor._read()
        self.motor._store_feedback([20] * 4, None, 100.01)
        with patch("base.motor.time.monotonic", return_value=100.02):
            self.assertEqual(self.motor.feedback(0), ([20] * 4, [1] * 4))

    def test_brake_uses_config_and_limits_pwm_in_both_directions(self):
        self.motor.pwm = Mock()
        with patch.dict(TURN, BRAKE_KP=2000.0):
            duties = self.motor.brake((1.0, -0.03, 0.01, -1.0))
        self.assertEqual(duties, (-1000, 60, 0, 1000))
        self.motor.pwm.assert_called_once_with(duties)


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
