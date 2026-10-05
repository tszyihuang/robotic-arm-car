"""小车与机械臂共用的任务语法及无硬件参数校验。"""
from __future__ import annotations

import math
import shlex
from dataclasses import dataclass
from pathlib import Path
from ..base import settings
from ..base import arc_turn, vision_align

TURN_SPEED_MM_S = 150.0
SPIN_SPEED_MM_S = 150.0


HELP = """任务清单：每行一条，支持 # 注释。
  straight 距离 [速度] [kp-gap=数值 ki-gap=数值 kd-gap=数值]
  turn 角度 [半径]
  calibrate-position [speed=数值 max-distance=数值 impact-threshold=数值]
                     [timeout=秒 accel=数值 boot-wait=秒 kp-gap=数值 ...]
  align [ref=e/head/mid tol=度 settle=秒 spin=速度 min-u=速度]
        [timeout=秒 bias=度 max-rot=度 rate-src=auto/gyro/gap]
  vision-straight 距离 [速度]
  scan-qrcode [device=摄像头编号或路径 timeout=秒]
              扫码结果返回 plan 节点的 qr_data，再继续后续任务
  gripper-open   张开夹爪（舵机 ID 2，291°）
  gripper-close  闭合夹爪（舵机 ID 2，243°）
  arm-calibrate  启动机械臂，取当前四轴编码器位置为软件零点和初始位置
  arm-move Q1 Q2 Q3 Q4 [G] 按 ID1-4 的关节角（°）移动，四轴等时并等待到位
                          G 为可选夹爪开度 0..1 或 close/open
                          初始零点姿态对应关节角 0 0 160 24
  arm-home / home 回到最近一次 arm-calibrate 记录的四轴初始零点，等待到位
  arm-disable    失能机械臂 ID1-4，无需校准；保持夹爪状态
机械臂移动前须在同一份清单中执行 arm-calibrate；整趟任务复用连接与零点。
距离单位米，速度单位 mm/s。普通直走支持负距离倒退。
"""

BASE_COMMANDS = frozenset(("straight", "turn", "calibrate-position", "align", "vision-straight",
))
ARM_COMMANDS = frozenset(("arm-calibrate", "arm-move", "arm-home", "home", "arm-disable",
                          "gripper-open", "gripper-close"))
VISION_COMMANDS = frozenset(("scan-qrcode",))
VISUAL_BASE_COMMANDS = frozenset(("vision-straight", "align"))
_COMMON = {"log_path", "verbose"}
_VISION = {key for key in settings.DEFAULTS if key.startswith("vision_")}
_POSITION_NUMBERS = {"speed", "max_distance_m", "impact_threshold", "timeout",
                     "accel", "boot_wait", "kp_gap", "ki_gap", "kd_gap"}
_ALIGN_PARAMETERS = {
    "ref": "align_ref", "tol": "align_tol", "settle": "align_settle",
    "spin": "align_spin_speed", "min_u": "align_min_u", "timeout": "align_timeout",
    "bias": "align_bias", "max_rot": "align_max_rot", "rate_src": "align_rate_src",
}


def default_task_file():
    """源码运行使用项目任务表；独立安装使用包内任务表。"""
    source_root = Path(__file__).resolve().parents[4]
    if (source_root / "plan.py").is_file():
        return source_root / "tasks.txt"
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory("car_nodes")) / "tasks.txt"


def task_sections(text):
    """读取 [分段名] 标题，保留各段指令在原任务表中的行号。"""
    sections = {}
    current = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        heading = raw.partition('#')[0].strip()
        if heading.startswith('[') and heading.endswith(']'):
            name = heading[1:-1].strip()
            if not name or '[' in name or ']' in name:
                raise ValueError(f'清单第 {lineno} 行：分段名称无效')
            if name in sections:
                raise ValueError(f'清单第 {lineno} 行：重复的任务分段 [{name}]')
            current = [''] * lineno
            sections[name] = current
        elif current is not None:
            current.append(raw)
    return {name: '\n'.join(lines) for name, lines in sections.items()}


def select_task_section(text, section=None):
    """分段任务表默认选择主线；未分段的旧任务表仍执行整表。"""
    sections = task_sections(text)
    if not sections and section is None:
        return text
    name = section if section is not None else '主线'
    if name not in sections:
        available = '、'.join(f'--{item}' for item in sections) or '无（任务表未分段）'
        raise ValueError(f'找不到任务分段 [{name}]；可选分段：{available}')
    selected = sections[name]
    if not parse_tasks(selected):
        raise ValueError(f'任务分段 [{name}] 为空，请先在该分段中填写指令')
    return selected


def parse_tasks(text):
    """返回 (行号, 命令, 参数, 原句)，正确保留引号内的 #。"""
    steps = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        try:
            parts = shlex.split(raw, comments=True)
        except ValueError as exc:
            raise ValueError(f"清单第 {lineno} 行：{exc}") from exc
        if parts:
            steps.append((lineno, parts[0], parts[1:], shlex.join(parts)))
    return steps


def _num(text, line):
    try:
        value = float(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"`{line}` 的 {text!r} 不是数字") from exc
    if not math.isfinite(value):
        raise ValueError(f"`{line}` 的数字必须有限")
    return value


def _positive(value, name, allow_zero=False, maximum=None):
    if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} 超出有效范围：{value:g}")


def _prepare_step(cmd, argv, line, params):
    """纯参数解析与校验；整份清单在打开硬件前调用。"""
    if cmd == "scan-qrcode":
        kwargs = {}
        for token in argv:
            key, sep, value = token.partition("=")
            if not sep or key not in ("device", "timeout") or key in kwargs:
                raise ValueError(f"`{line}` 的扫码参数 {token!r} 不支持或重复")
            if key == "timeout":
                value = _num(value, line)
                _positive(value, "扫码超时")
            elif value.isdecimal():
                value = int(value)
            elif not value.startswith("/dev/video") or not value[10:].isdecimal():
                raise ValueError("扫码摄像头必须为非负编号或 /dev/videoN 路径")
            kwargs[key] = value
        return cmd, (), kwargs
    if cmd in ARM_COMMANDS:
        if cmd == "arm-move":
            if len(argv) not in (4, 5):
                raise ValueError(f"`{line}` 需要 Q1 Q2 Q3 Q4 [G]（四个关节角，单位 °）")
            angles = [_num(value, line) for value in argv[:4]]
            from ..arm import ArmConfig
            from ..arm.joints import validate_joints

            validate_joints(dict(enumerate(angles, start=1)), ArmConfig())
            if len(argv) == 5:
                opening = {"close": 0.0, "open": 1.0}.get(argv[4].lower())
                if opening is None:
                    opening = _num(argv[4], line)
                if not 0 <= opening <= 1:
                    raise ValueError(f"`{line}` 的夹爪开度必须在 [0, 1] 内")
                angles.append(opening)
            return cmd, tuple(angles), {}
        if argv:
            raise ValueError(f"`{line}` 不接受参数")
        return cmd, (), {}
    if cmd in ("straight", "vision-straight"):
        if not argv:
            raise ValueError(f"`{line}` 缺少距离")
        distance = _num(argv[0], line)
        if distance == 0 or (cmd == "vision-straight" and distance < 0):
            raise ValueError("直走距离不能为零，视觉直走距离必须为正")
        allowed = _COMMON | {"speed", "meters_per_count"}
        allowed |= _VISION if cmd == "vision-straight" else {
            "straight_kp_gap", "straight_ki_gap", "straight_kd_gap"}
        kwargs = {key: value for key, value in params.items() if key in allowed}
        rest = argv[1:]
        if rest and "=" not in rest[0]:
            kwargs["speed"] = _num(rest[0], line)
            rest = rest[1:]
        for token in rest:
            key, sep, value = token.partition("=")
            key = key.replace("-", "_")
            if cmd != "straight" or not sep or key not in ("kp_gap", "ki_gap", "kd_gap"):
                raise ValueError(f"`{line}` 的参数 {token!r} 不支持")
            gain = _num(value, line)
            _positive(gain, key, allow_zero=True)
            kwargs["straight_" + key] = gain
        _positive(kwargs.get("speed", settings.DEFAULTS["speed"]), "速度", maximum=1000)
        return cmd, (distance,), kwargs
    if cmd == "turn":
        if not 1 <= len(argv) <= 2:
            raise ValueError(f"`{line}` 需要角度和可选半径")
        angle = _num(argv[0], line)
        radius = _num(argv[1], line) if len(argv) == 2 else params.get("turn_radius", settings.DEFAULTS["turn_radius"])
        if angle == 0 or abs(angle) > 360:
            raise ValueError("单次转角必须非零且不超过 360°")
        _positive(radius, "半径", allow_zero=True)
        kwargs = {key: value for key, value in params.items()
                  if key in _COMMON or key.startswith("turn_") and key != "turn_speed"}
        kwargs["speed"] = params.get("turn_speed", TURN_SPEED_MM_S)
        arc_turn.make_plan(radius, angle, kwargs["speed"] / 1000,
                                    kwargs.get("turn_max_speed"),
                                    kwargs.get("turn_track_width", settings.DEFAULTS["turn_track_width"]))
        return cmd, (angle, radius), kwargs
    if cmd == "calibrate-position":
        kwargs = {key: value for key, value in params.items()
                  if key in _COMMON | {"speed"}}
        for token in argv:
            key, sep, value = token.partition("=")
            key = key.replace("-", "_")
            key = {"max_distance": "max_distance_m", "log": "log_path"}.get(key, key)
            if not sep or key not in _POSITION_NUMBERS | {"log_path"}:
                raise ValueError(f"`{line}` 的位置校准参数 {token!r} 不支持")
            if key in _POSITION_NUMBERS:
                value = _num(value, line)
                _positive(value, key, allow_zero=key in {"boot_wait", "kp_gap", "ki_gap", "kd_gap"},
                          maximum=1000 if key == "speed" else None)
            kwargs[key] = value
        return cmd, (), kwargs
    if cmd == "align":
        allowed = _COMMON | _VISION
        kwargs = {key: value for key, value in params.items()
                  if key in allowed or key.startswith("align_")}
        for token in argv:
            key, sep, value = token.partition("=")
            key = key.replace("-", "_")
            if not sep or key not in _ALIGN_PARAMETERS:
                raise ValueError(f"`{line}` 的对正参数 {token!r} 不支持")
            if key == "ref":
                if value not in ("e", "head", "mid"):
                    raise ValueError("ref 只能是 e / head / mid")
            elif key == "rate_src":
                if value not in ("auto", "gyro", "gap"):
                    raise ValueError("rate-src 只能是 auto / gyro / gap")
            else:
                value = _num(value, line)
                if key != "bias":
                    _positive(value, key, allow_zero=key in {"settle", "min_u"},
                              maximum=1000 if key in {"spin", "min_u"} else None)
            kwargs[_ALIGN_PARAMETERS[key]] = value
        vision_align.AlignCfg(**{
            field: kwargs["align_" + field] for field in settings._ALIGN_FIELDS
            if kwargs.get("align_" + field) is not None})
        return cmd, (), kwargs
    raise ValueError(f"未知任务命令 {cmd!r}；可用命令见 --help")


@dataclass(frozen=True)
class PreparedTask:
    lineno: int
    command: str
    args: tuple
    kwargs: dict
    text: str

    @property
    def target(self):
        if self.command in VISION_COMMANDS:
            return "vision"
        return "base" if self.command in BASE_COMMANDS else "arm"

    @property
    def needs_vision(self):
        return self.command in VISUAL_BASE_COMMANDS


def prepare_tasks(text, params=None, *, require_calibration=True):
    """Validate the complete list without opening hardware or importing ROS."""
    prepared = []
    calibrated = False
    for lineno, cmd, argv, line in parse_tasks(text):
        try:
            command, args, kwargs = _prepare_step(cmd, argv, line, params or {})
            if command == "arm-calibrate":
                calibrated = True
            elif require_calibration and command in ("arm-move", "arm-home", "home") and not calibrated:
                raise ValueError("机械臂动作前必须先执行 arm-calibrate（同一份清单）")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"清单第 {lineno} 行：{exc}") from exc
        prepared.append(PreparedTask(lineno, command, args, kwargs, line))
    if not prepared:
        raise ValueError("任务清单为空")
    return prepared
