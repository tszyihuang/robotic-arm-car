"""驱动板串口：速度指令、累计编码器和轮速反馈，单位 mm/s。"""
import re
import time

import serial
from config import BASE


BAUD = BASE["baudrate"]

class SerialBoard:
    """四路电机驱动板串口驱动（协议见 驱动资料/）。"""

    _RE_MALL = re.compile(rb"\$MAll:(-?\d+),(-?\d+),(-?\d+),(-?\d+)#")
    _RE_MTEP = re.compile(rb"\$MTEP:(-?\d+),(-?\d+),(-?\d+),(-?\d+)#")
    _RE_BATTERY = re.compile(rb"\$[Bb]attery:([^#]+)#")

    def __init__(self, port, baud=BAUD):
        # timeout=0 → 非阻塞读，控制环靠"读干净为止"取最新值，延迟最低
        self.ser = serial.Serial(port, baud, timeout=0, write_timeout=0.2)
        self._feedback_buf = b""
        self.battery = None
        try:
            time.sleep(0.2)
            self.ser.reset_input_buffer()
        except BaseException:
            cleanup(('电机串口连接', self.ser.close))
            raise

    def _send(self, cmd):
        data = ("$" + cmd + "#\r\n").encode("ascii")
        if self.ser.write(data) != len(data):
            raise RuntimeError("电机指令未完整发送")
        self.ser.flush()

    def upload(self):
        """打开 累计脉冲 $MAll(算里程/航向) + 10ms 增量 $MTEP(算轮速) 上报。"""
        self._send("upload:1,1,0")

    def spd(self, m1, m2, m3, m4):
        """四轮速度闭环指令 (mm/s, -1000~1000)，无回复。"""
        wheels = tuple(round(max(-1000, min(1000, value))) for value in (m1, m2, m3, m4))
        self._send("spd:%d,%d,%d,%d" % wheels)

    def feedback(self, timeout=0.08):
        """
        读最新一条 $MAll 和同一批数据里的 $MTEP，返回 (totals, tep)。
        CH340 按 USB 帧成批送数据，单次 read 可能只拿到半行，所以循环到解析出完整一行为止。
        """
        buf, end = self._feedback_buf, time.monotonic() + timeout
        while time.monotonic() < end:
            d = self.ser.read(65536)
            if d:
                buf += d
                if self._RE_MALL.search(buf):
                    for _ in range(64):          # 队列里还有就一并读掉，保证拿到最新一条
                        d2 = self.ser.read(65536)
                        if not d2:
                            break
                        buf += d2
                    break
            else:
                time.sleep(0.0005)
        complete, sep, tail = buf.rpartition(b"#")
        self._feedback_buf = tail[-256:] if sep else buf[-256:]
        buf = complete + sep
        voltage = self._RE_BATTERY.findall(buf)
        if voltage:
            try:
                self.battery = float(voltage[-1].strip().rstrip(b'Vv'))
            except ValueError:
                pass
        tm = self._RE_MALL.findall(buf)
        if not tm:
            return None, None
        tp = self._RE_MTEP.findall(buf)
        return ([int(x) for x in tm[-1]],
                [int(x) for x in tp[-1]] if tp else None)

    def stop(self):
        """$spd 归零：板载速度环把轮子锁住，防止收尾后溜车。"""
        self.spd(0, 0, 0, 0)

    def close(self):
        cleanup(('电机停车', self.stop), ('电机串口', self.ser.close))


from control import MotionCancelled, check_cancel, cleanup
import threading
import glob

def find_motor_port(baud=BAUD, stop_event=None):
    """找出电机驱动板：谁能回 $read_flash 里的 Motor_Version 谁是。"""
    check_cancel(stop_event)
    for port in sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*")):
        check_cancel(stop_event)
        try:
            with serial.Serial(port, baud, timeout=0.1) as ser:
                ser.reset_input_buffer()
                ser.write(b"$read_flash#\r\n")
                ser.flush()
                end, buf = time.time() + 1.0, b""
                while time.time() < end and b"Motor_Version" not in buf:
                    check_cancel(stop_event)
                    buf += ser.read(4096)
                    check_cancel(stop_event)
                if b"Motor_Version" in buf:
                    return port
        except MotionCancelled:
            raise
        except Exception:
            continue
    return None


class Motor:
    """独占驱动板串口，后台采集；控制循环每帧只消费一次增量。"""

    def __init__(self, port="", *, stop_event=None):
        self.stop_event = stop_event or threading.Event()
        self.shutdown = threading.Event()
        self.lock = threading.Lock()
        self.condition = threading.Condition()
        self.error = None
        self.sequence = 0
        self._last_sequence = 0
        self._totals = self._increments = None
        self.capture_stamp = 0.0
        self.counts = None
        self.stamp = 0.0
        self.thread = None
        self.default_timeout = BASE['feedback_timeout']
        check_cancel(self.stop_event)
        port = port or find_motor_port(stop_event=self.stop_event)
        if not port:
            raise RuntimeError("找不到电机驱动板，请设置 config.py 的 BASE['motor_port']")
        self.board = SerialBoard(port)
        try:
            check_cancel(self.stop_event)
            self.board.stop()
            self.board.upload()
            self.board._send("read_vol")
            self.thread = threading.Thread(target=self._read, name="motor-encoders", daemon=True)
            self.thread.start()
        except BaseException:
            cleanup(("驱动板连接", self.board.close))
            raise

    @property
    def port(self):
        return self.board.ser.port

    @property
    def battery(self):
        return self.board.battery

    def _store_feedback(self, totals, increments, stamp):
        with self.condition:
            if stamp <= self.capture_stamp:
                return
            self.sequence += 1
            self._totals = list(totals)
            self._increments = list(increments) if increments is not None else None
            self.capture_stamp = stamp
            self.condition.notify_all()

    def _read(self):
        try:
            while not self.shutdown.is_set():
                with self.lock:
                    totals, increments = self.board.feedback(0.001)
                if totals is not None:
                    self._store_feedback(totals, increments, time.monotonic())
                self.shutdown.wait(0.001)
        except Exception as exc:
            with self.condition:
                self.error = exc
                self.condition.notify_all()

    def feedback(self, timeout=None):
        timeout = self.default_timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        with self.condition:
            while True:
                check_cancel(self.stop_event)
                if self.error is not None:
                    raise RuntimeError(f"电机反馈读取失败：{self.error}")
                if (self.sequence != self._last_sequence and self._totals is not None
                        and time.monotonic() - self.capture_stamp < BASE["feedback_stale"]):
                    self._last_sequence = self.sequence
                    self._consumed_stamp = self.capture_stamp
                    return list(self._totals), (list(self._increments) if self._increments is not None else None)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, None
                self.condition.wait(min(remaining, 0.01))

    def poll(self):
        totals, _ = self.feedback(0.0)
        if totals is not None:
            self.counts = tuple(totals)
            self.stamp = self._consumed_stamp

    def spd(self, *wheels):
        with self.lock:
            if self.stop_event.is_set():
                wheels = (0, 0, 0, 0)
            self.board.spd(*wheels)

    def speed(self, wheels):
        self.spd(*(value * 1000 for value in wheels))

    def pwm(self, wheels):
        with self.lock:
            if self.stop_event.is_set():
                wheels = (0, 0, 0, 0)
            self.board._send("pwm:%d,%d,%d,%d" % tuple(wheels))

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
        self.shutdown.set()
        actions = []
        if self.thread is not None:
            actions.append(("编码器采集线程", lambda: self.thread.join(timeout=2.0)))
        actions.extend((("底盘停车", self.stop), ("底盘释放", self.release),
                        ("电机串口", self.board.ser.close)))
        cleanup(*actions)
