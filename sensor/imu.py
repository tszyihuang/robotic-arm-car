"""共享 IMU 读取器与串口识别：连续角、融合角速度、原始陀螺与加速度帧。"""
import glob
import os
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass
from config import SENSOR, POSITION

import serial
from base.control import MotionCancelled, check_cancel, clamp, cleanup

H1, H2 = 0x7E, 0x23
FUNC_QUAT, FUNC_RAW = 0x16, 0x04
CMD_RATE, FIXED_PARAM = 0x60, 0x5F
GYRO_LSB = 2000.0 / 32767.0
BAUD = SENSOR["baudrate"]
IMU_HZ = SENSOR["hz"]
IMU_STALE = SENSOR["stale"]
MAX_YAW_RATE = SENSOR["MAX_YAW_RATE"]
MIN_YAW_STEP = SENSOR["MIN_YAW_STEP"]
MAX_YAW_STEP = SENSOR["MAX_YAW_STEP"]
GLITCH_HOLD = SENSOR["GLITCH_HOLD"]

def build_frame(func, *params):
    """按协议打包一帧：[7E][23][LEN][FUNC][DATA...][CHK]，LEN=整帧字节数。"""
    body = bytes([H1, H2, 0, func]) + bytes(params)
    head = bytes([H1, H2, len(body) + 1])
    return head + body[3:] + bytes([sum(head + body[3:]) & 0xFF])


def wrap_pm180(a):
    return (a + 180.0) % 360.0 - 180.0


def yaw_from_quat(q):
    """四元数 -> 偏航角（度）。"""
    w, x, y, z = q
    return math.degrees(math.atan2(2.0 * (w * z + x * y),
                                   1.0 - 2.0 * (y * y + z * z)))


def has_valid_frame(buf):
    """buf 里是否存在一个校验正确的 7E23 帧（用来认 IMU 串口）。"""
    i, n = 0, len(buf)
    while i + 4 <= n:
        if buf[i] != H1 or buf[i + 1] != H2:
            i += 1
            continue
        length = buf[i + 2]
        if not (7 <= length <= 64) or i + length > n:
            i += 1
            continue
        if sum(buf[i:i + length - 1]) & 0xFF == buf[i + length - 1]:
            return True
        i += 1
    return False


class ImpactDetector:
    """对每个原始样本检测突变；一次撞击即锁存，不要求连续多帧超阈值。"""

    def __init__(self, threshold=POSITION["impact_threshold"]):
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("碰撞阈值必须是有限正数")
        self.threshold = threshold
        self.acc = None
        self.last_t = None
        self.delta = self.peak = 0.0
        self.armed = self.hit = False
        self.frames = 0

    def arm(self):
        self.delta = self.peak = 0.0
        self.hit = False
        self.armed = True

    def feed(self, acc, now):
        # 断流后的第一帧重新取基准，断流本身由独立的加速度看门狗处理。
        fresh = self.last_t is not None and 0 <= now - self.last_t < IMU_STALE
        self.delta = math.dist(acc, self.acc) if fresh else 0.0
        self.acc, self.last_t = tuple(acc), now
        self.frames += 1
        self.peak = max(self.peak, self.delta)
        if self.armed and self.delta >= self.threshold:
            self.hit = True


@dataclass(frozen=True)
class AccelState:
    age: float
    acc: tuple | None
    delta: float
    peak: float
    hit: bool
    frames: int



class ImuLink:
    """后台线程持续解帧，对外只暴露连续角 yaw 与角速度 yaw_rate / rate。"""

    YAW_RATE_WINDOW = SENSOR["YAW_RATE_WINDOW"]     # 求导窗口（秒）：越小越灵敏、噪声越大

    def __init__(self, port, baud=BAUD, gyro_axis=SENSOR["gyro_axis"]):
        self.ser = serial.Serial(port, baud, timeout=0.02, write_timeout=0.2)
        self.lock = threading.Lock()
        self.gyro_axis = gyro_axis
        self.impact = ImpactDetector(POSITION["impact_threshold"])
        self.error = None
        self.cont = 0.0            # 连续展开角
        self.yaw_raw = 0.0         # 模块原始 ±180
        self._rate_raw = 0.0       # 陀螺原始角速度（度/秒）
        self._yaw_rate = 0.0       # 融合 yaw 求导得到的角速度（度/秒）
        self._yaw_hist = deque(maxlen=64)   # (t, cont) 求导用的小窗口
        self.yaw_ref = 0.0
        self.last_t = 0.0
        self.n_quat = 0
        self.n_raw = 0
        self.n_bad = 0
        self.n_bad_quat = 0        # 模长/数值不合法、直接丢掉的四元数帧
        self.n_glitch = 0          # 跳变超物理上限、扣住没积分的帧
        self.n_ref_jump = 0        # 连续异常后被接受的"模块换了绝对基准"次数
        self.last_ref_jump = None  # (时间, 跳变角度) 供上层报警
        self._prev = None
        self._prev_t = 0.0
        self._glitch_t0 = None
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._run, daemon=True)

    # --- 对外接口 ---
    def start(self):
        self._th.start()

    def close(self):
        self._stop.set()
        actions = []
        if self._th.ident is not None:
            actions.append(('IMU 采集线程', lambda: self._th.join(timeout=1.0)))
        actions.append(('IMU 串口', self.ser.close))
        cleanup(*actions)

    @property
    def yaw(self):
        """相对零位的角度（度），可超出 ±180°。"""
        with self.lock:
            return self.cont - self.yaw_ref

    @property
    def rate(self):
        """原始陀螺 Z 角速度（度/秒），用于圆弧转弯的快速反馈。"""
        with self.lock:
            return self._rate_raw

    @property
    def yaw_rate(self):
        """融合 yaw 求导得到的角速度（度/秒），符号与 yaw 一致，控制环用它。"""
        with self.lock:
            return self._yaw_rate

    @property
    def rate_raw(self):
        with self.lock:
            return self._rate_raw

    def age(self):
        """最后一帧距今多久（秒），看门狗用。"""
        with self.lock:
            return time.time() - self.last_t if self.last_t else 1e9

    def has_data(self):
        with self.lock:
            return self.last_t > 0 and time.time() - self.last_t < IMU_STALE

    def zero(self):
        """把当前朝向记为 0°，必须在车体静止时调用。"""
        with self.lock:
            self.yaw_ref = self.cont

    @property
    def impact_threshold(self):
        with self.lock:
            return self.impact.threshold

    @impact_threshold.setter
    def impact_threshold(self, value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("碰撞阈值必须是有限正数")
        with self.lock:
            self.impact.threshold = value

    def accel_state(self):
        with self.lock:
            impact = self.impact
            age = max(0.0, time.time() - impact.last_t) if impact.last_t else 1e9
            return AccelState(age, impact.acc, impact.delta, impact.peak, impact.hit, impact.frames)

    def arm_impact(self):
        with self.lock:
            if self.impact.last_t is None or time.time() - self.impact.last_t >= IMU_STALE:
                raise RuntimeError("没有新鲜的 IMU 原始加速度帧，不能倒车")
            self.impact.arm()

    def disarm_impact(self):
        with self.lock:
            self.impact.armed = False

    def send_rate(self, hz):
        frame = build_frame(CMD_RATE, hz, FIXED_PARAM)
        if self.ser.write(frame) != len(frame):
            raise RuntimeError("IMU 指令未完整发送")
        self.ser.flush()

    # --- 线程主体 ---
    def _run(self):
        buf = bytearray()
        while not self._stop.is_set():
            try:
                data = self.ser.read(256)
            except Exception as exc:
                self.error = exc
                break
            if not data:
                continue
            buf += data
            while len(buf) >= 4:
                if buf[0] != H1 or buf[1] != H2:
                    buf.pop(0)
                    continue
                length = buf[2]
                if not (7 <= length <= 64):
                    buf.pop(0)
                    continue
                if len(buf) < length:
                    break
                if sum(buf[:length - 1]) & 0xFF == buf[length - 1]:
                    func = buf[3]
                    payload = bytes(buf[4:length - 1])
                    del buf[:length]
                    self._handle(func, payload)
                else:
                    buf.pop(0)
                    self.n_bad += 1

    def _handle(self, func, payload):
        now = time.time()
        if func == FUNC_QUAT and len(payload) == 16:
            q = struct.unpack('<4f', payload)
            y = yaw_from_quat(q)
            norm2 = sum(v * v for v in q)
            if not (math.isfinite(y) and 0.75 < norm2 < 1.25):
                with self.lock:
                    self.n_bad_quat += 1
                return
            with self.lock:
                self.yaw_raw = y
                self.last_t = now
                self.n_quat += 1
                self._feed_yaw(now, y)
        elif func == FUNC_RAW and len(payload) == 18:
            v = struct.unpack('<9h', payload)      # acc xyz / gyro xyz / mag xyz
            with self.lock:
                self._rate_raw = v[3 + self.gyro_axis] * GYRO_LSB
                self.n_raw += 1
                self.impact.feed(v[:3], now)

    def _feed_yaw(self, now, y):
        """把一帧四元数 yaw 折进连续角，顺手挡住坏帧（必须在持锁时调用）。

        正常帧：cont += 跨 ±180 展开后的增量。
        异常帧（单帧跳变超过物理上限）：先不积分，往后看。
        连续异常超过 GLITCH_HOLD：认了——不是坏帧，是模块自己换了绝对基准
        （重启 / 磁干扰），把这一跳记进 last_ref_jump 让上层报警。
        """
        if self._prev is None:
            self._prev = y
            self._prev_t = now
            self._track_yaw_rate(now)
            return
        dt = now - self._prev_t
        d = wrap_pm180(y - self._prev)
        limit = min(clamp(MAX_YAW_RATE * dt, MIN_YAW_STEP, 180.0), MAX_YAW_STEP)
        if abs(d) > limit:
            self.n_glitch += 1
            if self._glitch_t0 is None:
                self._glitch_t0 = now
            elif now - self._glitch_t0 >= GLITCH_HOLD:
                self.cont += d                      # 接受新基准，保持连续性
                self._prev = y
                self._prev_t = now
                self._glitch_t0 = None
                self.n_ref_jump += 1
                self.last_ref_jump = (now, d)
                self._yaw_hist.clear()              # 求导窗口里全是旧基准，清掉
                self._yaw_rate = 0.0
                self._track_yaw_rate(now)
            return
        self._glitch_t0 = None
        self.cont += d
        self._prev = y
        self._prev_t = now
        self._track_yaw_rate(now)

    @property
    def glitch_count(self):
        """被扣住没有积分的异常帧数（正常应该是 0）。"""
        with self.lock:
            return self.n_glitch

    @property
    def ref_jump_count(self):
        """模块绝对基准跳变的次数（正常应该是 0）。"""
        with self.lock:
            return self.n_ref_jump

    def _track_yaw_rate(self, now):
        """用一小段窗口的首尾斜率求融合 yaw 的角速度（必须在持锁时调用）。

        窗口内首尾相减可以抵消"同一帧被读两次"的样本保持抖动，比相邻两帧
        差分干净得多；窗口本身要短，否则阻尼项会带上明显相位滞后。
        """
        h = self._yaw_hist
        h.append((now, self.cont))
        while len(h) > 2 and now - h[0][0] > self.YAW_RATE_WINDOW:
            h.popleft()
        t0, y0 = h[0]
        dt = now - t0
        if dt > 1e-3:
            self._yaw_rate = (self.cont - y0) / dt


def _candidate_ports():
    return sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))


def find_imu_port(seconds=1.5, skip=(), baud=BAUD, stop_event=None):
    """找出正在输出合法 7E23 帧的串口。IMU 上电后自动上报，不用发指令。"""
    check_cancel(stop_event)
    excluded = {os.path.realpath(port) for port in skip if port}
    for port in _candidate_ports():
        check_cancel(stop_event)
        if os.path.realpath(port) in excluded:
            continue
        try:
            with serial.Serial(port, baud, timeout=0.05) as ser:
                ser.reset_input_buffer()
                buf = bytearray()
                end = time.time() + seconds
                while time.time() < end:
                    check_cancel(stop_event)
                    buf += ser.read(512)
                    check_cancel(stop_event)
                    if has_valid_frame(buf):
                        return port
        except MotionCancelled:
            raise
        except Exception:
            continue
    return None

