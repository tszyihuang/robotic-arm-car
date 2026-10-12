"""TTL 舵机写入时限、调度延迟与故障日志；不连接实机。"""
import unittest
from unittest.mock import Mock, patch

import serial

from arm.config import ArmConfig
from arm.errors import ProtocolError
from arm.servo import FeetechSTSServo, GOAL_POSITION


class ServoCommunicationTests(unittest.TestCase):
    def test_serial_write_timeout_is_independent_of_reply_timeout(self):
        with patch('arm.servo.serial.Serial') as open_port:
            servo = FeetechSTSServo('/fake', baudrate=115200, timeout=.1, write_timeout=1.0)
            self.addCleanup(servo.close)
        open_port.assert_called_once_with('/fake', baudrate=115200,
                                         timeout=.01, write_timeout=1.0, exclusive=True)
        self.assertEqual(servo.timeout, .1)

    def test_buffered_reply_survives_thread_scheduling_delay_between_header_and_payload(self):
        servo = FeetechSTSServo(servo_id=2, simulate=True)
        self.addCleanup(servo.close)
        clock = [10.0]
        read = servo._ser.read

        def delayed_read(size):
            data = read(size)
            clock[0] += .2
            return data

        with patch.object(servo._ser, 'read', side_effect=delayed_read), \
                patch('arm.servo.time.monotonic', side_effect=lambda: clock[0]):
            self.assertEqual(servo.read_position(servo_id=1), 2048)

    def test_write_timeout_reports_port_baudrate_query_id_limit_and_frame_size(self):
        servo = FeetechSTSServo('/fake', baudrate=115200, servo_id=2, simulate=True)
        self.addCleanup(servo.close)
        with patch.object(servo._ser, 'write', side_effect=serial.SerialTimeoutException('Write timeout')) as write:
            with self.assertRaises(serial.SerialTimeoutException) as failed:
                servo.read_position(servo_id=1)
        write.assert_called_once()
        for detail in ('/fake', 'baudrate=115200', 'ID1', '0x02', 'write_timeout=1.0s', '耗时', '帧长 8'):
            self.assertIn(detail, str(failed.exception))

    def test_missing_reply_reports_requested_id_and_stops_within_reply_timeout(self):
        connection = Mock(in_waiting=0)
        connection.write.side_effect = len
        clock = [10.0]

        def empty_read(size):
            clock[0] += .01
            return b''

        connection.read.side_effect = empty_read
        servo = FeetechSTSServo('/fake', baudrate=115200, servo_id=2, serial_port=connection)
        self.addCleanup(servo.close)
        with patch('arm.servo.time.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(ProtocolError, 'ID1.*timeout=0.1s'):
                servo.read_position(servo_id=1)
        self.assertLessEqual(clock[0], 10.12)

    def test_timeout_does_not_replay_a_position_register_write(self):
        servo = FeetechSTSServo(servo_id=2, simulate=True)
        self.addCleanup(servo.close)
        with patch.object(servo._ser, 'write', side_effect=serial.SerialTimeoutException('Write timeout')) as write:
            with self.assertRaises(serial.SerialTimeoutException):
                servo._write_register(GOAL_POSITION, bytes(6))
        write.assert_called_once()

    def test_invalid_write_timeouts_are_rejected_before_opening_serial(self):
        for value in (0, -1, float('nan'), float('inf'), True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    ArmConfig(gripper_write_timeout=value)
                with patch('arm.servo.serial.Serial') as open_port, self.assertRaises(ValueError):
                    FeetechSTSServo('/fake', baudrate=115200, write_timeout=value)
                open_port.assert_not_called()
