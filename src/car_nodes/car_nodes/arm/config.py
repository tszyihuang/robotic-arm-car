"""任务表所需的串口、电机映射、关节限位和夹爪配置。"""

import math
from dataclasses import dataclass, field
from pathlib import Path

JOINT_IDS = (1, 2, 3, 4)
GRIPPER_SERVO_ID = 2
GRIPPER_OPEN_ANGLE = 291.0
GRIPPER_CLOSE_ANGLE = 243.0


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")
    return float(value)


@dataclass
class ArmConfig:
    port: str = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTBVQPDL-if00-port0"
    baudrate: int = 921600  # 沿用 GIM4310_driver.py 的实际值。
    serial_timeout: float = 0.1
    serial_latency_ms: int = 1  # Linux FTDI USB 串口的 latency_timer，单位毫秒。
    joint_limits: dict = field(default_factory=lambda: {
        1: None, 2: (-5.0, 180.0), 3: (0.0, 160.0), 4: (-120.0, 120.0),
    })
    joint_signs: dict = field(default_factory=lambda: {1: 1, 2: 1, 3: -1, 4: 1})
    joint_offsets_deg: dict = field(default_factory=lambda: {1: 0.0, 2: 0.0, 3: 160.0, 4: 24.0})
    encoder_zero_deg: dict = None  # None: 在固定的启动姿态读取软件零点。
    speed_rpm: float = 10.0
    arrival_tolerance_deg: float = 2.0
    move_timeout: float = 30.0
    gripper_port: str = None  # None: 沿用舵机驱动的只读自动识别。
    gripper_baudrate: int = None
    gripper_servo_id: int = GRIPPER_SERVO_ID
    gripper_speed: int = 1000  # STS 原生速度单位。
    gripper_timeout: float = 0.1
    gripper_open_angle_deg: float = GRIPPER_OPEN_ANGLE
    gripper_close_angle_deg: float = GRIPPER_CLOSE_ANGLE
    gripper_release_on_close: bool = False

    def __post_init__(self):
        if not isinstance(self.port, str) or not self.port:
            raise ValueError("port 不能为空")
        if isinstance(self.baudrate, bool) or not isinstance(self.baudrate, int) or self.baudrate <= 0:
            raise ValueError("baudrate 必须是正整数")
        if (isinstance(self.serial_latency_ms, bool) or not isinstance(self.serial_latency_ms, int)
                or not 1 <= self.serial_latency_ms <= 255):
            raise ValueError("serial_latency_ms 必须是 [1, 255] 内的整数毫秒")
        for name in ("serial_timeout", "speed_rpm",
                     "arrival_tolerance_deg", "move_timeout"):
            if finite(getattr(self, name), name) <= 0:
                raise ValueError(f"{name} 必须大于 0")
        if not 0.01 <= self.speed_rpm <= 0xFFFFFFFF / 100:
            raise ValueError("speed_rpm 超出协议可表示范围")
        if self.gripper_port is not None and (not isinstance(self.gripper_port, str) or not self.gripper_port):
            raise ValueError("gripper_port 必须是非空路径或 null")
        if self.gripper_port is not None and Path(self.gripper_port).resolve() == Path(self.port).resolve():
            raise ValueError("电机 RS485 与夹爪 TTL 必须使用不同的串口")
        for name, lo, hi in (("gripper_servo_id", 0, 253), ("gripper_speed", 1, 3400),
                             ("gripper_baudrate", 1, 10000000)):
            value = getattr(self, name)
            if name == "gripper_baudrate" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} 必须是 [{lo}, {hi}] 内的整数")
        if finite(self.gripper_timeout, "gripper_timeout") <= 0:
            raise ValueError("gripper_timeout 必须大于 0")
        for name in ("gripper_open_angle_deg", "gripper_close_angle_deg"):
            if not 0 <= finite(getattr(self, name), name) <= 360:
                raise ValueError(f"{name} 必须在 [0, 360]° 范围内")
        if self.gripper_open_angle_deg == self.gripper_close_angle_deg:
            raise ValueError("夹爪张开和闭合角度不能相同")
        if not isinstance(self.gripper_release_on_close, bool):
            raise ValueError("gripper_release_on_close 必须为 bool")
        for name in ("joint_limits", "joint_signs", "joint_offsets_deg", "encoder_zero_deg"):
            value = getattr(self, name)
            if name == "encoder_zero_deg" and value is None:
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{name} 必须是关节 1-4 的映射")
            if {str(k) for k in value} != {"1", "2", "3", "4"} or len(value) != 4:
                raise ValueError(f"{name} 必须完整包含关节 1-4")
            setattr(self, name, {int(k): v for k, v in value.items()})
        for addr in JOINT_IDS:
            bounds = self.joint_limits[addr]
            # ID1 的 None 表示取消软件角度限位；仍检查方向、偏置与编码器零点。
            if addr != 1 or bounds is not None:
                if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
                    raise ValueError(f"ID{addr} 限位必须包含上下界")
                lo, hi = (finite(v, f"ID{addr} 限位") for v in bounds)
                if lo >= hi:
                    raise ValueError(f"ID{addr} 限位上下界无效")
                self.joint_limits[addr] = (lo, hi)
            if isinstance(self.joint_signs[addr], bool) or self.joint_signs[addr] not in (-1, 1):
                raise ValueError(f"ID{addr} 方向必须是 +1 或 -1")
            finite(self.joint_offsets_deg[addr], f"ID{addr} 偏置")
            if self.encoder_zero_deg is not None:
                finite(self.encoder_zero_deg[addr], f"ID{addr} 编码器零点")
