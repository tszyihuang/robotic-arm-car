"""驱动板串口：速度指令、累计编码器和轮速反馈，单位 mm/s。"""
import re
import time

import serial

BAUD = 115200

class Board:
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
            self.ser.close()
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
        try:
            self.stop()
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass


from ..common.control import MotionCancelled, check_cancel
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
