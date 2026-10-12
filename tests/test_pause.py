"""暂停的回车与舵机转动触发，使用内存角度、单调时钟和真实输入管道。"""
import contextlib
import io
import os
import threading
import unittest
from unittest.mock import Mock, patch

from serial import SerialTimeoutException

from arm.errors import ProtocolError
from base.control import MotionCancelled
from tasks.wait_continue import ServoMotion, run


class ServoMotionTests(unittest.TestCase):
    def test_five_degrees_in_either_direction_within_half_second(self):
        for angle, elapsed in ((105, .5), (95, .4)):
            with self.subTest(angle=angle, elapsed=elapsed):
                motion = ServoMotion()
                self.assertFalse(motion.update(100, 0))
                self.assertTrue(motion.update(angle, elapsed))

    def test_slow_motion_jitter_and_samples_older_than_half_second_do_not_trigger(self):
        cases = [
            [(0, 100), (.5001, 105)],
            [(i / 10, i * .8) for i in range(11)],
            [(i / 10, 2 if i % 2 else -2) for i in range(11)],
            [(0, 100), (2, 120)],
        ]
        for samples in cases:
            with self.subTest(samples=samples):
                motion = ServoMotion()
                self.assertFalse(any(motion.update(angle, stamp) for stamp, angle in samples))

    def test_crossing_zero_counts_small_rotation_in_both_directions(self):
        for angles in ((359, 1, 4), (1, 359, 356)):
            with self.subTest(angles=angles):
                motion = ServoMotion()
                self.assertFalse(motion.update(angles[0], 0))
                self.assertFalse(motion.update(angles[1], .1))
                self.assertTrue(motion.update(angles[2], .2))

    def test_delayed_reply_does_not_make_old_movement_look_recent(self):
        motion = ServoMotion()
        self.assertFalse(motion.update(100, 10.4, sampled_at=10.0))
        self.assertFalse(motion.update(105, 10.6, sampled_at=10.55))


class PauseChannelsTests(unittest.TestCase):
    def setUp(self):
        read_fd, self.write_fd = os.pipe()
        self.stdin = os.fdopen(read_fd, 'rb', buffering=0)
        self.addCleanup(self.stdin.close)
        self.addCleanup(self.close_input)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('tasks.wait_continue.sys.stdin', self.stdin))
        self.output = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def close_input(self):
        if self.write_fd is not None:
            os.close(self.write_fd)
            self.write_fd = None

    def test_enter_remains_available_for_two_consecutive_pauses(self):
        arm = Mock()
        os.write(self.write_fd, b'\n\n')
        run(arm)
        run(arm)
        arm.read_servo_angle.assert_not_called()

    def test_partial_console_input_does_not_block_servo_trigger_or_steal_next_enter(self):
        arm = Mock()
        arm.read_servo_angle.side_effect = [359.0, 4.0]
        os.write(self.write_fd, b'x')
        with patch('tasks.wait_continue.time.monotonic', side_effect=[100.0, 100.01, 100.4, 100.41]):
            run(arm)
        self.assertIn('继续执行', self.output.getvalue())
        os.write(self.write_fd, b'\n')
        run(arm)
        self.assertEqual(arm.read_servo_angle.call_count, 2)

    def test_query_timeout_retries_and_recovery_uses_a_fresh_angle_baseline(self):
        for error in (SerialTimeoutException('Write timeout'), ProtocolError('应答超时')):
            with self.subTest(error=type(error).__name__):
                arm = Mock()
                clock = [10.0]
                samples = iter([100.0, error, 120.0, 125.0])

                def poll_enter(fd):
                    clock[0] = round(clock[0] + .05, 9)
                    return False

                def read_angle(*args, **kwargs):
                    clock[0] = round(clock[0] + .01, 9)
                    value = next(samples)
                    if isinstance(value, Exception):
                        raise value
                    return value

                arm.read_servo_angle.side_effect = read_angle
                with patch('tasks.wait_continue._enter_pressed', side_effect=poll_enter), \
                        patch('tasks.wait_continue.time.monotonic', side_effect=lambda: clock[0]):
                    run(arm)
                self.assertEqual(arm.read_servo_angle.call_count, 4)
        self.assertEqual(self.output.getvalue().count('通信已恢复'), 2)

    def test_persistent_timeout_keeps_enter_available_and_logs_once(self):
        arm = Mock()
        arm.read_servo_angle.side_effect = SerialTimeoutException('Write timeout')
        clock = [10.0]

        def poll_enter(fd):
            clock[0] = round(clock[0] + .3, 9)
            return arm.read_servo_angle.call_count == 3

        with patch('tasks.wait_continue._enter_pressed', side_effect=poll_enter), \
                patch('tasks.wait_continue.time.monotonic', side_effect=lambda: clock[0]):
            run(arm)
        self.assertEqual(arm.read_servo_angle.call_count, 3)
        self.assertEqual(self.output.getvalue().count('查询暂时失败'), 1)
        self.assertNotIn('转动已达', self.output.getvalue())

    def test_read_failure_keeps_console_channel_without_resuming_automatically(self):
        arm = Mock()

        def fail_read(*args, **kwargs):
            os.write(self.write_fd, b'\n')
            raise OSError('舵机断开')

        arm.read_servo_angle.side_effect = fail_read
        run(arm)
        self.assertIn('检测不可用', self.output.getvalue())
        self.assertIn('舵机断开', self.output.getvalue())
        self.assertNotIn('转动已达', self.output.getvalue())
        arm.read_servo_angle.assert_called_once()

    def test_cancellation_during_read_does_not_become_a_servo_trigger(self):
        arm = Mock()
        stop = threading.Event()

        def cancel_read(*args, **kwargs):
            stop.set()
            return 105.0

        arm.read_servo_angle.side_effect = cancel_read
        with self.assertRaises(MotionCancelled):
            run(arm, stop_event=stop)
        self.assertNotIn('转动已达', self.output.getvalue())

    def test_closed_input_stops_without_reading_servo(self):
        arm = Mock()
        self.close_input()
        with self.assertRaises(EOFError):
            run(arm)
        arm.read_servo_angle.assert_not_called()
