"""Single motor serial owner and controller view of sensor-node encoders."""
import threading
import time

from ..common.control import check_cancel
from ..common.messages import set_stamp
from .motor import Board, find_motor_port


class MotorOwner:
    def __init__(self, port, publish, stop_event):
        from car_interfaces.msg import Encoders
        self._message_type = Encoders
        self.publish = publish
        self.lock = threading.Lock()
        self.shutdown = threading.Event()
        self.error = None
        self.sequence = 0
        self.stop_event = stop_event
        self.thread = None
        port = port or find_motor_port(stop_event=stop_event)
        check_cancel(stop_event)
        if not port:
            raise RuntimeError('找不到电机驱动板，请设置 motor_port')
        self.board = Board(port)
        try:
            check_cancel(stop_event)
            self.board.stop()
            self.board.upload()
            self.board._send('read_vol')
            self.thread = threading.Thread(target=self._read, name='motor-encoders', daemon=True)
            self.thread.start()
        except Exception:
            self.board.close()
            raise

    def _read(self):
        try:
            while not self.shutdown.is_set():
                with self.lock:
                    totals, increments = self.board.feedback(0.001)
                if totals is not None:
                    self.sequence += 1
                    message = self._message_type(sequence=self.sequence, totals=totals,
                        increments=increments or [0] * 4, has_increments=increments is not None)
                    set_stamp(message.header, time.time(), 'base_link')
                    self.publish(message)
                self.shutdown.wait(0.001)
        except Exception as exc:
            self.error = exc

    def send(self, command):
        with self.lock:
            if self.stop_event.is_set() and command.startswith('pwm:'):
                command = 'pwm:0,0,0,0'
            self.board._send(command)

    def spd(self, *wheels):
        with self.lock:
            if self.stop_event.is_set():
                wheels = (0, 0, 0, 0)
            self.board.spd(*wheels)

    def close(self):
        self.shutdown.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        with self.lock:
            try:
                self.board._send('pwm:0,0,0,0')
            finally:
                self.board.ser.close()


class ControllerBoard:
    def __init__(self, owner, source, default_timeout=0.08):
        self.owner, self.source = owner, source
        self.default_timeout = default_timeout
        self._last_sequence = None
        self.counts = None
        self.stamp = 0.0
        self.battery = None

    def feedback(self, timeout=None):
        timeout = self.default_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        with self.source.condition:
            while True:
                if self.owner.error is not None:
                    raise RuntimeError(f'电机反馈读取失败：{self.owner.error}')
                sequence, stamp, totals, increments = self.source.encoder_snapshot()
                if (sequence != self._last_sequence and totals is not None
                        and self.source.clock() - stamp < 0.3):
                    self._last_sequence = sequence
                    return totals, increments
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, None
                self.source.condition.wait(min(remaining, 0.01))

    def poll(self):
        self.battery = self.owner.board.battery
        totals, _ = self.feedback(0.0)
        if totals is not None:
            self.counts = tuple(totals)
            capture_age = max(0.0, self.source.clock() - self.source.encoder_stamp)
            self.stamp = time.monotonic() - capture_age

    def spd(self, *wheels):
        self.owner.spd(*wheels)

    def speed(self, wheels):
        self.spd(*(value * 1000 for value in wheels))

    def pwm(self, wheels):
        self.owner.send('pwm:%d,%d,%d,%d' % tuple(wheels))

    def brake(self, speeds):
        duties = tuple(round(max(-1000, min(1000, -3000 * value)))
                       if abs(value) >= 0.02 else 0 for value in speeds)
        self.pwm(duties)
        return duties

    def stop(self):
        self.spd(0, 0, 0, 0)

    def release(self):
        self.pwm((0, 0, 0, 0))

    def close(self):
        try:
            self.stop()
        finally:
            self.release()
