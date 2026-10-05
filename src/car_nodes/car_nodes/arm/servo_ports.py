"""通过只读 STS 应答识别 USB 串口，并缓存稳定设备路径及实际波特率。"""

import json
import os
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

from .config import ArmConfig
from .errors import ArmError
from .servo import FeetechSTSServo, SERVO_BAUDRATE

BINDING_FILE = Path(__file__).with_name("servo_binding.json")
BAUDRATES = (SERVO_BAUDRATE, 115200, 500000, 250000, 128000, 76800, 57600, 38400)


def _stable_path(device, serial_number=None):
    real_device = os.path.realpath(device)
    # 无唯一序列号的 CH340，其 by-id 链接可能同时被另一块同型号设备占用。
    directories = ("/dev/serial/by-id", "/dev/serial/by-path") if serial_number else (
        "/dev/serial/by-path",)
    for directory in directories:
        for path in sorted(Path(directory).glob("*")):
            if os.path.realpath(path) == real_device:
                return str(path)
    return device


def _busy(device):
    if not shutil.which("fuser"):
        return False
    try:
        result = subprocess.run(["fuser", device], stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, timeout=1)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return True


def _probe(port, baudrate, servo_id, timeout):
    # 显式传串口和波特率，不递归进入自动发现；正常关闭也不写扭矩寄存器。
    with FeetechSTSServo(port, baudrate=baudrate, servo_id=servo_id,
                         timeout=timeout, release_on_close=False) as servo:
        status = servo.status()
        model = int.from_bytes(servo.read_register(3, 2), "little")
        return {"model": model, "status": status}


def _load_binding():
    try:
        data = json.loads(BINDING_FILE.read_text(encoding="utf-8"))
        if (isinstance(data, dict) and data.get("version") == 1
                and isinstance(data.get("port"), str)
                and isinstance(data.get("baudrate"), int)):
            return data
    except (OSError, ValueError):
        pass
    return None


def _save_binding(binding):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=BINDING_FILE.parent,
                                         prefix=".servo_binding-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(binding, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(BINDING_FILE)
    except OSError as exc:
        warnings.warn(f"舵机已识别，但设备绑定未能保存：{exc}", RuntimeWarning)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def detect_servo_device(*, port=None, baudrate=None, servo_id=1, timeout=0.1):
    """返回 port/baudrate 绑定。缓存也需只读确认；显式参数限制扫描范围。"""
    import serial.tools.list_ports

    ports = list(serial.tools.list_ports.comports())
    info = {os.path.realpath(p.device): p for p in ports}
    cached = _load_binding()
    if cached and (port is None or os.path.realpath(port) == os.path.realpath(cached["port"])):
        if (baudrate is None or baudrate == cached["baudrate"]) and Path(cached["port"]).exists():
            if not _busy(cached["port"]):
                try:
                    _probe(cached["port"], cached["baudrate"], servo_id, timeout)
                    return {**cached, "servo_id": servo_id}
                except (ArmError, OSError, ValueError):
                    pass

    arm_port = os.path.realpath(ArmConfig.port)
    if port is not None:
        candidates = [port]
    else:
        candidates = sorted({p.device for p in ports if
                             (p.vid is not None or p.device.startswith(("/dev/ttyUSB", "/dev/ttyACM")))
                             and os.path.realpath(p.device) != arm_port})
    candidates = [p for p in candidates if not _busy(p)]
    if not candidates:
        raise ArmError("未找到可用的舵机 USB 串口；请检查连接或串口是否被占用")

    matches = []
    errors = {}
    speeds = (baudrate,) if baudrate is not None else tuple(dict.fromkeys(BAUDRATES))
    for candidate in candidates:
        for speed in speeds:
            try:
                result = _probe(candidate, speed, servo_id, timeout)
            except (ArmError, OSError, ValueError) as exc:
                errors[candidate] = str(exc)
                continue
            metadata = info.get(os.path.realpath(candidate))
            matches.append({"version": 1,
                            "port": _stable_path(candidate, getattr(metadata, "serial_number", None)),
                            "baudrate": speed, "servo_id": servo_id,
                            "model": result["model"],
                            "vid": getattr(metadata, "vid", None),
                            "pid": getattr(metadata, "pid", None),
                            "serial_number": getattr(metadata, "serial_number", None),
                            "location": getattr(metadata, "location", None)})
            break
    if not matches:
        detail = "; ".join(f"{p}: {error}" for p, error in errors.items())
        raise ArmError(f"未收到舵机 ID{servo_id} 的有效状态回包。请检查 TTL 接线、供电或 ID；{detail}")
    if len(matches) > 1:
        detail = ", ".join(f"{m['port']} ({m['baudrate']} baud)" for m in matches)
        raise ArmError(f"多个串口均检测到舵机 ID{servo_id}，请用 --port 指定：{detail}")
    _save_binding(matches[0])
    return matches[0]
