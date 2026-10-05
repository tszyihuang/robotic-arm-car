"""原项目 Feetech STS 工具旋转舵机的独立半双工 TTL 驱动。

使用 STS/SMS 的小端寄存器布局，不用于 SCS 或机械臂 GIM4310 RS485 电机。
寄存器核对：https://gitee.com/ftservo/FTServo_Python/blob/main/scservo_sdk/sms_sts.py
"""

import struct
import threading
import time
import warnings

from .config import finite
from .errors import ArmError, MotionTimeout, ProtocolError

SERVO_ID = 1
SERVO_BAUDRATE = 1000000
SERVO_SPEED = 1000  # 舵机原生速度值，不是关节电机的 rpm。
SERVO_TIMEOUT = 0.1
SERVO_STEP_PER_DEG = 4095 / 360
SERVO_FIXED_POSITIONS = {1: 340}  # 工具旋转舵机固定在 29.89°；夹爪 ID2 自由运动。

MODE = 0x21
TORQUE_ENABLE = 0x28
GOAL_POSITION = 0x2A
TORQUE_LIMIT = 0x30
PRESENT_POSITION = 0x38


class ServoFault(ArmError):
    """舵机状态包报告故障。"""


def _integer(value, name, lo, hi):
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ValueError(f"{name} 必须是 [{lo}, {hi}] 内的整数")
    return value


def _checksum(data):
    return (~sum(data)) & 0xFF


def _signed_magnitude(word):
    # STS 的符号在 bit15，其余位是幅值；不是 int16 二进制补码。
    return -(word & 0x7FFF) if word & 0x8000 else word


class FeetechSTSServo:
    """单颗 STS 舵机；打开时不移动、不使能，默认关闭仅释放串口。"""

    ERROR_CODES = {1: "电压错误", 2: "角度传感器错误", 4: "过热",
                   8: "过流", 16: "校验错误", 32: "过载", 64: "指令错误"}

    def __init__(self, port=None, *, baudrate=None, servo_id=SERVO_ID,
                 timeout=SERVO_TIMEOUT, speed=SERVO_SPEED, simulate=False,
                 release_on_close=False, serial_port=None):
        self.servo_id = _integer(servo_id, "舵机 ID", 0, 253)
        if baudrate is not None:
            _integer(baudrate, "波特率", 1, 10000000)
        if port is not None and (not isinstance(port, str) or not port):
            raise ValueError("舵机串口必须是非空路径")
        self.speed = _integer(speed, "舵机速度", 1, 3400)
        self.timeout = finite(timeout, "舵机串口超时")
        if self.timeout <= 0:
            raise ValueError("舵机串口超时必须大于 0")
        if not isinstance(release_on_close, bool):
            raise ValueError("release_on_close 必须为 bool")
        if simulate and serial_port is not None:
            raise ValueError("仿真和外部串口不能同时使用")
        self.release_on_close = release_on_close
        self._lock = threading.RLock()
        self._closed = False
        self._target = None
        self._motion_commanded = False
        self.port = port
        self.baudrate = baudrate or SERVO_BAUDRATE
        if simulate:
            serial_port = _SimulatedServoSerial(self.servo_id)
        elif serial_port is None:
            if port is None or baudrate is None:
                from .servo_ports import detect_servo_device
                binding = detect_servo_device(port=port, baudrate=baudrate,
                                               servo_id=servo_id, timeout=self.timeout)
                self.port, self.baudrate = binding["port"], binding["baudrate"]
            import serial
            serial_port = serial.Serial(self.port, baudrate=self.baudrate,
                                        timeout=min(self.timeout, 0.01),
                                        write_timeout=self.timeout, exclusive=True)
        self._ser = serial_port

    @staticmethod
    def decode_error(code):
        if not code:
            return "正常"
        text = [label for bit, label in FeetechSTSServo.ERROR_CODES.items() if code & bit]
        if code & 0x80:
            text.append("未知状态位 0x80")
        return " + ".join(text)

    def _read_exact(self, size, deadline):
        data = bytearray()
        while len(data) < size and time.monotonic() < deadline:
            chunk = self._ser.read(size - len(data))
            if chunk:
                data.extend(chunk)
        if len(data) != size:
            raise ProtocolError(f"舵机 ID{self.servo_id} 应答超时：需要 {size} 字节，收到 {len(data)}")
        return bytes(data)

    def _exchange(self, instruction, params=b"", expected_size=0, *, servo_id=None):
        address = self.servo_id if servo_id is None else _integer(servo_id, "舵机 ID", 0, 253)
        with self._lock:
            if self._closed:
                raise ArmError("舵机串口已关闭")
            body = bytes((address, len(params) + 2, instruction)) + params
            packet = b"\xff\xff" + body + bytes((_checksum(body),))
            self._ser.reset_input_buffer()
            if self._ser.write(packet) != len(packet):
                raise ProtocolError("舵机指令未完整发送")
            self._ser.flush()
            deadline = time.monotonic() + self.timeout
            header = self._read_exact(4, deadline)
            if header[:2] != b"\xff\xff" or header[2] != address:
                raise ProtocolError("舵机应答帧头或 ID 不匹配")
            length = header[3]
            if not 2 <= length <= 66:
                raise ProtocolError("舵机应答长度异常")
            tail = self._read_exact(length, deadline)
            if _checksum(header[2:] + tail[:-1]) != tail[-1]:
                raise ProtocolError("舵机应答校验和错误")
            if tail[0]:
                raise ServoFault(f"舵机 ID{address} 故障 0x{tail[0]:02X}："
                                 f"{self.decode_error(tail[0])}")
            if length != expected_size + 2:
                raise ProtocolError(f"舵机应答数据应为 {expected_size} 字节，实际 {length - 2}")
            return tail[1:-1]

    def read_register(self, address, size=2, *, servo_id=None):
        address = _integer(address, "寄存器地址", 0, 255)
        size = _integer(size, "读取长度", 1, min(64, 256 - address))
        return self._exchange(0x02, bytes((address, size)), size, servo_id=servo_id)

    def _write_register(self, address, data):
        return self._exchange(0x03, bytes((address,)) + data)

    def read_position(self):
        """返回编码器位置整数；STS 的负值按符号幅值解码。"""
        return _signed_magnitude(struct.unpack("<H", self.read_register(PRESENT_POSITION))[0])

    def status(self, *, servo_id=None):
        with self._lock:
            data = self.read_register(PRESENT_POSITION, 15, servo_id=servo_id)
            position, speed = struct.unpack("<HH", data[:4])
            position = _signed_magnitude(position)
            return {"id": self.servo_id if servo_id is None else servo_id, "position": position,
                    "angle_deg": position / SERVO_STEP_PER_DEG,
                    "speed_raw": _signed_magnitude(speed),
                    "voltage_v": data[6] / 10.0, "temperature_c": data[7],
                    "moving": bool(data[10]),
                    "torque_enabled": bool(self.read_register(TORQUE_ENABLE, 1, servo_id=servo_id)[0]),
                    "mode": self.read_register(MODE, 1, servo_id=servo_id)[0]}

    def enable_torque(self):
        """显式使能；move_to 会先写目标再使能，避免恢复旧位置目标。"""
        if self.servo_id in SERVO_FIXED_POSITIONS:
            self.move_to(SERVO_FIXED_POSITIONS[self.servo_id])
            return
        self._write_register(TORQUE_ENABLE, b"\x01")

    def disable_torque(self):
        with self._lock:
            self._write_register(TORQUE_ENABLE, b"\x00")
            self._motion_commanded = False
            self._target = None

    def set_torque_limit(self, value):
        value = _integer(value, "扭矩限制", 0, 1000)
        self._write_register(TORQUE_LIMIT, struct.pack("<H", value))

    def _stop_after_error(self):
        if self._motion_commanded:
            try:
                self.disable_torque()
            except Exception as exc:
                warnings.warn(f"舵机异常后关闭扭矩失败：{exc}", RuntimeWarning)
            finally:
                self._motion_commanded = False
                self._target = None

    def move_to(self, target_position, target_speed=None, target_time=0, *, wait=False):
        """发送单圈绝对位置 0..4095；可选择等待到位，所有参数先验证。"""
        target = _integer(target_position, "目标位置", 0, 4095)
        speed = _integer(self.speed if target_speed is None else target_speed, "舵机速度", 1, 3400)
        target_time = _integer(target_time, "目标时间", 0, 65535)
        fixed = SERVO_FIXED_POSITIONS.get(self.servo_id)
        if fixed is not None and target != fixed:
            raise ValueError(f"舵机 ID{self.servo_id} 已锁定在位置 {fixed}"
                             f"（{fixed / SERVO_STEP_PER_DEG:.2f}°），不能设置其他位置")
        with self._lock:
            try:
                if self.read_register(MODE, 1)[0] != 0:
                    raise ValueError("只支持单圈位置模式（mode=0）；请先用舵机配置工具确认模式")
                self._motion_commanded = True
                self._write_register(GOAL_POSITION, struct.pack("<HHH", target, target_time, speed))
                self._write_register(TORQUE_ENABLE, b"\x01")
                self._target = target
                if wait:
                    self.wait_for_arrival(target)
            except BaseException:
                self._stop_after_error()
                raise
        return target

    def move_angle(self, angle_deg, target_speed=None, *, wait=False):
        """绝对编码器角度 0..360°，不改变零点或安装方向。"""
        angle = finite(angle_deg, "舵机绝对角度")
        if not 0 <= angle <= 360:
            raise ValueError("舵机绝对角度必须在 [0, 360]° 范围内")
        return self.move_to(round(angle * SERVO_STEP_PER_DEG), target_speed, wait=wait)

    def move_relative_deg(self, delta_deg, target_speed=None, *, wait=False):
        """保留原项目正=逆时针的映射；模 4096，仅给出最终单圈目标。"""
        delta = finite(delta_deg, "舵机相对角度")
        if not -360 < delta < 360:
            raise ValueError("单圈位置模式不支持一次旋转整圈或多圈，相对角度必须在 (-360, 360)° 内")
        if target_speed is not None:
            _integer(target_speed, "舵机速度", 1, 3400)
        fixed = SERVO_FIXED_POSITIONS.get(self.servo_id)
        if fixed is not None:
            if delta != 0:
                raise ValueError(f"舵机 ID{self.servo_id} 已锁定在位置 {fixed}，不能相对旋转")
            return self.move_to(fixed, target_speed, wait=wait)
        with self._lock:
            try:
                current = _integer(self.read_position(), "实测单圈位置", 0, 4095)
                target = (current - int(delta * SERVO_STEP_PER_DEG)) % 4096
                return self.move_to(target, target_speed, wait=wait)
            except BaseException:
                self._stop_after_error()
                raise

    def wait_for_arrival(self, target_position=None, *, tolerance=6, timeout=8.0):
        target = _integer(self._target if target_position is None else target_position,
                          "等待目标", 0, 4095)
        tolerance = _integer(tolerance, "到位容差", 0, 2047)
        timeout = finite(timeout, "舵机到位超时")
        if timeout <= 0:
            raise ValueError("舵机到位超时必须大于 0")
        deadline = time.monotonic() + timeout
        try:
            while True:
                if abs(self.read_position() - target) <= tolerance:
                    return True
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MotionTimeout(f"舵机 {timeout:g} 秒内未到位")
                time.sleep(min(0.02, remaining))
        except BaseException:
            self._stop_after_error()
            raise

    def hold(self):
        with self._lock:
            if self.servo_id in SERVO_FIXED_POSITIONS:
                return self.move_to(SERVO_FIXED_POSITIONS[self.servo_id])
            return self.move_to(self.read_position())

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                if self.release_on_close:
                    self.disable_torque()
            finally:
                self._closed = True
                self._ser.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is not None:
            self._stop_after_error()
        self.close()
        return False


class _SimulatedServoSerial:
    """即时到位的 STS 寄存器仿真，仍走完整协议封包与应答校验。"""

    def __init__(self, servo_id):
        self.servo_id = servo_id
        self.memory = bytearray(256)
        self.memory[GOAL_POSITION:GOAL_POSITION + 2] = struct.pack("<H", 2048)
        self.memory[PRESENT_POSITION:PRESENT_POSITION + 2] = struct.pack("<H", 2048)
        self.memory[62:64] = bytes((120, 25))
        self.memories = {servo_id: self.memory}
        self.buffer = b""

    def reset_input_buffer(self):
        self.buffer = b""

    def write(self, packet):
        servo_id = packet[2]
        memory = self.memories.setdefault(servo_id, bytearray(self.memory))
        instruction = packet[4]
        address = packet[5]
        data = b""
        if instruction == 0x02:
            size = packet[6]
            data = bytes(memory[address:address + size])
        else:
            payload = packet[6:-1]
            memory[address:address + len(payload)] = payload
            if memory[TORQUE_ENABLE]:
                memory[PRESENT_POSITION:PRESENT_POSITION + 2] = memory[
                    GOAL_POSITION:GOAL_POSITION + 2]
        body = bytes((servo_id, len(data) + 2, 0)) + data
        self.buffer = b"\xff\xff" + body + bytes((_checksum(body),))
        return len(packet)

    def flush(self):
        pass

    def read(self, size):
        result, self.buffer = self.buffer[:size], self.buffer[size:]
        return result

    def close(self):
        pass
