"""提取自 GIM4310_driver.py 的 RS485 V3.03b3 位置驱动。

仅保留状态读取、位置/速度、梯形曲线和失能。四台电机共用同一把总线锁，
逐帧收完应答再发送下一帧，避免多台从机的应答互相碰撞。
"""

import math
import struct
import threading
import time
import warnings
from pathlib import Path

from .config import JOINT_IDS, finite
from .errors import MotorFault, ProtocolError
from base.control import cleanup

ANGLE_SCALE = 360.0 / 16384


def crc16(data):
    """CRC16_MODBUS；初值 0xFFFF，多项式 0xA001。"""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xA001 if crc & 1 else 0)
    return crc


def position_payload(angle_deg, speed_rpm, accel_rpm_s=None):
    """发送前完成数值与协议整数范围检查。"""
    angle = finite(angle_deg, "电机角度")
    speed_rpm = finite(speed_rpm, "电机速度")
    if not -0x80000000 * ANGLE_SCALE <= angle <= 0x7FFFFFFF * ANGLE_SCALE:
        raise ValueError("电机角度超出协议 int32 范围")
    if not 0.01 <= speed_rpm <= 0xFFFFFFFF / 100:
        raise ValueError("电机速度必须在 [0.01, 42949672.95] rpm 范围内")
    encoder = int(angle / ANGLE_SCALE)
    # 消除 0.29 * 100 == 28.999999999999996 这类误差，仍向下量化。
    speed = int(math.nextafter(speed_rpm * 100, math.inf))
    if accel_rpm_s is None:
        return 0x25, struct.pack("<BiI", 0, encoder, speed)
    accel_rpm_s = finite(accel_rpm_s, "电机加速度")
    if not 0.01 <= accel_rpm_s <= 0xFFFFFFFF / 100:
        raise ValueError("电机加速度超出协议范围")
    accel = int(math.nextafter(accel_rpm_s * 100, math.inf))
    return 0x26, struct.pack("<BiIII", 0, encoder, speed, accel, accel)


def parse_status(data, address):
    if len(data) != 22:
        raise ProtocolError(f"ID{address}: 状态应为 22 字节，实际 {len(data)}")
    single, multi, speed, current, voltage, bus_current, temp, mode, enabled, fault = struct.unpack(
        "<HiiiHHBBBB", data)
    return dict(single_turn_deg=single * ANGLE_SCALE, multi_turn_deg=multi * ANGLE_SCALE,
                speed_rpm=speed * 0.01, q_current_a=current * 0.001,
                bus_voltage_v=voltage * 0.01, bus_current_a=bus_current * 0.01,
                temperature=temp, run_mode=mode, motor_enabled=enabled, fault_code=fault)


def check_fault(status, address):
    if status["fault_code"]:
        raise MotorFault(f"ID{address}: 电机故障码 0x{status['fault_code']:02X}")


def set_serial_latency(serial_port, latency_ms=1):
    """设置 Linux USB 串口 latency_timer；稳定 by-id 路径先解析为 ttyUSB 名称。"""
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, int) or not 1 <= latency_ms <= 255:
        raise ValueError("串口延迟必须是 [1, 255] 内的整数毫秒")
    port = getattr(serial_port, "port", None)
    if not isinstance(port, str):
        return None
    device = Path(port).resolve().name
    path = Path("/sys/bus/usb-serial/devices") / device / "latency_timer"
    if not path.exists():
        return None  # 其他串口驱动没有这个属性。
    current = None
    try:
        current = int(path.read_text().strip())
        if current != latency_ms:
            path.write_text(str(latency_ms))
            current = int(path.read_text().strip())
        if current != latency_ms:
            raise OSError(f"设置后回读为 {current} ms")
    except (OSError, ValueError) as exc:
        warnings.warn(f"USB 串口延迟未能设为 {latency_ms} ms（当前 {current} ms）：{exc}。"
                      "请使用 sudo 启动，以写入 latency_timer。", RuntimeWarning)
    return current


class Motor:
    def __init__(self, bus, address):
        self.bus = bus
        self.address = address

    def read_status(self):
        status = parse_status(self.bus.exchange(self.address, 0x0B), self.address)
        check_fault(status, self.address)
        return status

    def move(self, angle_deg, speed_rpm=10.0, accel_rpm_s=None):
        cmd, data = position_payload(angle_deg, speed_rpm, accel_rpm_s)
        status = parse_status(self.bus.exchange(self.address, cmd, data), self.address)
        check_fault(status, self.address)
        return status

    def disable(self):
        # 0x2E 是抱闸开关，不能当作通用使能指令。位置命令直接启动位置控制。
        self.bus.exchange(self.address, 0x2F)


class MotorBus:
    def __init__(self, port, baudrate=921600, timeout=0.1, serial_port=None, *, latency_ms=1):
        self.timeout = finite(timeout, "串口超时")
        if self.timeout <= 0:
            raise ValueError("串口超时必须大于 0")
        self._lock = threading.Lock()
        self._seq = 0
        self._closed = False
        if serial_port is None:
            import serial  # 仿真和纯运动学不依赖 pyserial。
            serial_port = serial.Serial(port, baudrate=baudrate, timeout=min(timeout, 0.02),
                                        write_timeout=timeout, bytesize=8, parity="N", stopbits=1)
        self._ser = serial_port
        try:
            self.latency_ms = set_serial_latency(serial_port, latency_ms)
            self.motors = {addr: Motor(self, addr) for addr in JOINT_IDS}
        except BaseException:
            cleanup(('四轴串口连接', serial_port.close))
            raise

    def _read_exact(self, size, deadline):
        result = bytearray()
        while len(result) < size and time.monotonic() < deadline:
            chunk = self._ser.read(size - len(result))
            if chunk:
                result.extend(chunk)
        if len(result) != size:
            raise ProtocolError(f"RS485 应答超时：需要 {size} 字节，收到 {len(result)}")
        return bytes(result)

    def exchange(self, address, cmd, data=b""):
        if address not in JOINT_IDS or not 0 <= cmd <= 255 or len(data) > 248:
            raise ValueError("无效的电机地址、命令码或数据长度")
        with self._lock:
            if self._closed:
                raise ProtocolError("RS485 总线已关闭")
            self._seq = (self._seq + 1) & 0xFF
            frame = bytes((0xAE, self._seq, address, cmd, len(data))) + data
            frame += struct.pack("<H", crc16(frame))
            try:
                self._ser.reset_input_buffer()
                if self._ser.write(frame) != len(frame):
                    raise ProtocolError("串口未完整发送命令")
                self._ser.flush()
                deadline = time.monotonic() + self.timeout
                head = self._read_exact(5, deadline)
                if head[:4] != bytes((0xAC, self._seq, address, cmd)):
                    raise ProtocolError("响应帧头、包序号、设备地址或命令码不匹配")
                if head[4] > 64:
                    raise ProtocolError("响应数据长度异常")
                tail = self._read_exact(head[4] + 2, deadline)
                if crc16(head + tail[:-2]) != struct.unpack("<H", tail[-2:])[0]:
                    raise ProtocolError("CRC16 校验失败")
                return tail[:-2]
            except (OSError, ProtocolError) as exc:
                raise ProtocolError(f"ID{address} 命令 0x{cmd:02X}: {exc}") from exc

    def disable_all(self):
        failures = []
        for addr, motor in self.motors.items():
            for _ in range(3):
                try:
                    motor.disable()
                    break
                except (OSError, ProtocolError) as exc:
                    error = exc
            else:
                failures.append(f"ID{addr}: {error}")
        if failures:
            raise ProtocolError("部分电机失能失败：" + "; ".join(failures))

    def close(self, *, disable_motors=True):
        if self._closed:
            return
        try:
            if disable_motors:
                self.disable_all()
        except ProtocolError as exc:
            warnings.warn(str(exc), RuntimeWarning)
        finally:
            self._closed = True
            self._ser.close()


class SimulatedMotor:
    """软件测试用，目标即时到达；不模拟负载、惯性或串口时序。"""

    def __init__(self, address, zero):
        self.address = address
        self.angle = zero
        self.enabled = False

    def read_status(self):
        return dict(multi_turn_deg=self.angle, fault_code=0, speed_rpm=0.0,
                    motor_enabled=int(self.enabled))

    def move(self, angle_deg, speed_rpm=10.0, accel_rpm_s=None):
        position_payload(angle_deg, speed_rpm, accel_rpm_s)
        self.angle = angle_deg
        self.enabled = True
        return self.read_status()

    def disable(self):
        self.enabled = False


class SimulatedBus:
    def __init__(self, config):
        zeros = config.encoder_zero_deg or dict.fromkeys(JOINT_IDS, 0.0)
        self.motors = {addr: SimulatedMotor(addr, zeros[addr]) for addr in JOINT_IDS}
        self.closed = False

    def disable_all(self):
        for motor in self.motors.values():
            motor.disable()

    def close(self, *, disable_motors=True):
        if self.closed:
            return
        if disable_motors:
            self.disable_all()
        self.closed = True
