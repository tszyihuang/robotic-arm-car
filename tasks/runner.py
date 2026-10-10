"""读取任务表的指定段落，在执行前检查整段指令。"""
from dataclasses import dataclass
import math
from pathlib import Path

from . import pause
from . import detect_ball
from . import detect_target
from . import calibrate_ball_position
from . import calibrate_object_position
from .mission import TASK_SECTIONS
from base.control import check_cancel


LOCK_TARGET_COMMANDS = {
    "lock-target-left": "left",
    "lock-target-middle": "middle",
    "lock-target-right": "right",
}


# 指令 → 设备、方法、允许的参数数量；不执行任务表中的 Python 代码。
COMMANDS = {
    "straight": ("base", "straight", (1, 2)),
    "turn": ("base", "turn", (1, 2)),
    "calibrate-position": ("base", "calibrate_position", (0,)),
    "calibrate-ball-position": ("base", "calibrate_ball_position", (0,)),
    "calibrate-object-position": ("base", "calibrate_object_position", (0,)),
    "align": ("base", "align", (0,)),
    "vision-straight": ("base", "vision_straight", (1,)),
    "arm-calibrate": ("arm", "calibrate", (0,)),
    "arm-disable": ("arm", "disable", (0,)),
    "arm-move": ("arm", "move_joints", (4,)),
    "arm-home": ("arm", "home", (0,)),
    "home": ("arm", "home", (0,)),
    "gripper-open": ("arm", "open_gripper", (0,)),
    "gripper-close": ("arm", "close_gripper", (0,)),
    "scan-qrcode": ("vision", "scan_qrcode", (0,)),
    "detect-balls": ("vision", "observe_balls", (0,)),
    "detect-targets": ("vision", "observe_targets", (0,)),
    **{command: ("arm", "lock_target", (0, 1)) for command in LOCK_TARGET_COMMANDS},
}


@dataclass(frozen=True)
class Step:
    line: int
    command: str
    args: tuple


def list_sections(path):
    """按文件顺序列出段落标题，包含空段落，同名标题只列一次。"""
    sections = []
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        text = raw.split("#", 1)[0].strip()
        if text.startswith("[") and text.endswith("]"):
            name = text[1:-1]
            if name and name not in sections:
                sections.append(name)
    return sections


def load_main(path):
    return load_section(path, "主线")


def load_section(path, section):
    path = Path(path)
    steps = []
    found = False
    for line, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        text = raw.split("#", 1)[0].strip()
        if not text:
            continue
        if text.startswith("[") and text.endswith("]"):
            if found:
                break
            found = text == f"[{section}]"
            continue
        if not found:
            continue
        command, *values = text.split()
        try:
            if command not in COMMANDS and command not in TASK_SECTIONS:
                raise ValueError(f"未知指令 {command!r}")
            counts = (0,) if command in TASK_SECTIONS else COMMANDS[command][2]
            if len(values) not in counts:
                expected = " 或 ".join(map(str, counts))
                raise ValueError(f"{command} 需要 {expected} 个参数，实际 {len(values)} 个")
            args = tuple(float(value) for value in values)
            if any(not math.isfinite(value) for value in args):
                raise ValueError("参数必须为有限数值")
            if command in LOCK_TARGET_COMMANDS and args and args[0] < 0:
                raise ValueError("锁靶超时必须为非负数，0 表示不限时")
        except ValueError as exc:
            raise ValueError(f"{path.name}:{line}：{exc}") from exc
        steps.append(Step(line, command, args))
    if not found:
        raise ValueError(f"{path.name} 缺少 [{section}] 段")
    if not steps:
        raise ValueError(f"{path.name} 的 [{section}] 没有可执行指令")
    return steps


def load_branches(path, steps, *, section="主线"):
    """出发前校验所有可能调用的动作段，拒绝缺段、语法错误及循环调用。"""
    branches = {}
    visiting = [section]

    def collect(current_steps):
        for step in current_steps:
            for name in TASK_SECTIONS.get(step.command, {}).values():
                if name in visiting:
                    chain = " → ".join((*visiting, name))
                    raise ValueError(f"{Path(path).name}:{step.line}：任务段调用循环：{chain}")
                if name in branches:
                    continue
                branch = load_section(path, name)
                branches[name] = branch
                visiting.append(name)
                collect(branch)
                visiting.pop()

    collect(steps)
    return branches


def execute(step, base, arm, vision, *, stop_event=None, target_color=None):
    check_cancel(stop_event)
    owner, method, _ = COMMANDS[step.command]
    device = {"base": base, "arm": arm, "vision": vision}[owner]
    args = step.args
    if step.command in ("align", "vision-straight"):
        args += (vision,)
    if step.command == "detect-balls":
        result = detect_ball.run(device)
    elif step.command == "detect-targets":
        result = detect_target.run(device, stop_event=stop_event)
    elif step.command == "calibrate-ball-position":
        result = calibrate_ball_position.run(base, vision, stop_event=stop_event)
    elif step.command == "calibrate-object-position":
        result = calibrate_object_position.run(base, vision, stop_event=stop_event)
    elif step.command in LOCK_TARGET_COMMANDS:
        from . import lock_target
        selector = LOCK_TARGET_COMMANDS[step.command] if target_color is None else target_color
        result = lock_target.run_task(arm, vision, selector,
                                      duration=args[0] if args else 0.0, stop_event=stop_event)
    elif step.command == "arm-calibrate":
        result = device.calibrate(hold_tool=True)
    else:
        result = getattr(device, method)(*args)
    check_cancel(stop_event)
    if step.command == "scan-qrcode":
        print(f"  二维码：{result}")
    pause(device)
    return result
