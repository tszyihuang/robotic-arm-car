"""RS485 分片、调度延迟、残留帧及只读查询重试；不连接实机。"""

import os
import struct
import unittest
from unittest.mock import patch

from arm.errors import ProtocolError
from arm.motor import MotorBus, crc16


def response(request, payload=bytes(22), *, sequence=None, address=None, command=None):
    frame = bytes((0xAC, request[1] if sequence is None else sequence,
                   request[2] if address is None else address,
                   request[3] if command is None else command, len(payload))) + payload
    return frame + struct.pack("<H", crc16(frame))


class FakeSerial:
    port = None
    timeout = 0.02

    def __init__(self, replies, *, pause_after_first_read=0.0):
        self.replies = list(replies)
        self.pending = []
        self.writes = []
        self.now = 0.0
        self.pause_after_first_read = pause_after_first_read

    @property
    def in_waiting(self):
        return len(self.pending[0]) if self.pending else 0

    def reset_input_buffer(self):
        self.pending.clear()

    def write(self, request):
        self.writes.append(request)
        reply = self.replies.pop(0)(request) if self.replies else b""
        self.pending.extend(reply if isinstance(reply, list) else [reply])
        return len(request)

    def flush(self):
        pass

    def read(self, size):
        if not self.pending or not self.pending[0]:
            if self.pending:
                self.pending.pop(0)
            self.now += self.timeout
            return b""
        chunk = self.pending[0][:size]
        self.pending[0] = self.pending[0][size:]
        if not self.pending[0]:
            self.pending.pop(0)
        self.now += 0.001 + self.pause_after_first_read
        self.pause_after_first_read = 0.0
        return chunk

    def close(self):
        pass


class MotorBusTests(unittest.TestCase):
    def make_bus(self, replies, **kwargs):
        serial_port = FakeSerial(replies, **kwargs)
        self.clock = patch("arm.motor.time.monotonic", side_effect=lambda: serial_port.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        bus = MotorBus("/fake", serial_port=serial_port)
        self.addCleanup(lambda: bus.close(disable_motors=False))
        return bus, serial_port

    def test_buffered_reply_is_read_after_thread_scheduling_delay(self):
        bus, serial_port = self.make_bus([response], pause_after_first_read=0.15)
        self.assertEqual(bus.exchange(1, 0x0B), bytes(22))
        self.assertEqual(len(serial_port.writes), 1)

    def test_header_and_payload_can_arrive_in_separate_fragments(self):
        def fragmented(request):
            frame = response(request)
            return [frame[:2], b"", frame[2:5], b"", frame[5:12], frame[12:]]

        bus, _ = self.make_bus([fragmented])
        self.assertEqual(bus.exchange(2, 0x0B), bytes(22))

    def test_garbage_and_unrelated_frames_are_skipped(self):
        def noisy(request):
            return (b"\x00\xFF\xAC" + response(request, sequence=0) +
                    response(request, address=2) + response(request, command=0x25) +
                    response(request))

        bus, serial_port = self.make_bus([noisy])
        self.assertEqual(bus.exchange(1, 0x0B), bytes(22))
        self.assertEqual(len(serial_port.writes), 1)

    def test_bad_length_and_crc_resynchronize_without_resending_motion(self):
        def damaged(request):
            valid = response(request)
            bad_crc = valid[:-1] + bytes((valid[-1] ^ 0xFF,))
            return valid[:4] + b"\xFF" + bad_crc + valid

        bus, serial_port = self.make_bus([damaged])
        self.assertEqual(bus.exchange(1, 0x25, bytes(9)), bytes(22))
        self.assertEqual(len(serial_port.writes), 1)

    def test_status_retry_uses_new_sequence_and_ignores_late_reply(self):
        def retry(request):
            return response(request, sequence=1) + response(request)

        bus, serial_port = self.make_bus([lambda request: response(request)[:5], retry])
        self.assertEqual(bus.motors[1].read_status()["fault_code"], 0)
        self.assertEqual([frame[1] for frame in serial_port.writes], [1, 2])
        self.assertTrue(all(frame[3] == 0x0B for frame in serial_port.writes))

    def test_status_timeout_is_bounded_and_keeps_received_bytes_in_error(self):
        bus, serial_port = self.make_bus([lambda request: response(request)[:5]] * 3)
        with self.assertRaisesRegex(ProtocolError, r"ID1.*0x0B.*AC.*16"):
            bus.exchange(1, 0x0B)
        self.assertEqual(len(serial_port.writes), 3)
        self.assertLessEqual(serial_port.now, 0.36)

    def test_position_and_disable_commands_are_not_automatically_retried(self):
        for command in (0x25, 0x26, 0x2F):
            with self.subTest(command=command):
                bus, serial_port = self.make_bus([lambda request: b"", response])
                with self.assertRaises(ProtocolError):
                    bus.exchange(1, command)
                self.assertEqual(len(serial_port.writes), 1)

    @unittest.skipUnless(os.name == "posix", "串口独占使用 POSIX 锁")
    def test_second_bus_cannot_open_the_same_serial_device(self):
        master, slave = os.openpty()
        self.addCleanup(os.close, master)
        self.addCleanup(os.close, slave)
        port = os.ttyname(slave)
        bus = MotorBus(port)
        self.addCleanup(lambda: bus.close(disable_motors=False))
        with self.assertRaises(OSError):
            MotorBus(port)


if __name__ == "__main__":
    unittest.main()
