"""视觉采集与底盘控制共享的边界反馈，不依赖模型或控制算法。"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class BoundarySample:
    info: dict | None
    frames: int | None
    seq: int
    state: str
    error: str
    t: float
    t_valid: float
    frame_dt: float = 0.0


def valid_geometry(info):
    if not isinstance(info, dict):
        return False

    def finite(value):
        return type(value) in (float, int) and math.isfinite(value)

    size = info.get("size")
    if size is not None and (not isinstance(size, (tuple, list)) or len(size) != 2
                             or any(not finite(v) or v < 1 or int(v) != v for v in size)):
        return False
    found = False
    for name in ("left", "right"):
        side = info.get(name)
        if side is None:
            continue
        if not isinstance(side, dict) or not all(finite(side.get(k)) for k in ("a", "b")):
            return False
        for key in ("near_y", "far_y", "angle_deg", "confidence"):
            if side.get(key) is not None and not finite(side[key]):
                return False
        found = True
    return found


def validate_keypoint_pairs(pairs):
    if (not isinstance(pairs, list) or len(pairs) != 2
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
            or any(type(index) is not int for pair in pairs for index in pair)
            or sorted(index for pair in pairs for index in pair) != [0, 1, 2, 3]):
        raise ValueError("左右边界各需两个点，索引 0、1、2、3 必须各出现一次")
    return [list(pair) for pair in pairs]
