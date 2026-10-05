"""二维码的业务含义及三个画面位置到子任务名称的映射。"""
import re
from .tasks import BRANCH_NAMES

COLORS = ("red", "green", "blue")
SHAPES = ("cylinder", "cone", "drum")


def mission_code(data):
    if not isinstance(data, str) or not re.fullmatch(r"[123]{3}", data.strip()):
        raise ValueError("任务二维码必须是三位 1、2、3 的组合，例如 211")
    code = data.strip()
    return {"ball": COLORS[int(code[0]) - 1], "target": COLORS[int(code[1]) - 1],
            "object": SHAPES[int(code[2]) - 1]}


def selected_branch(kind, position):
    if type(position) is not int or position not in (0, 1, 2):
        raise ValueError("视觉位置必须为 0（左）、1（中）、2（右）")
    return BRANCH_NAMES[kind][position]
