"""任务表关节角的完整性和限位校验。"""
from .config import JOINT_IDS, finite


def validate_joints(joints, config):
    if set(joints) != set(JOINT_IDS):
        raise ValueError("关节角必须完整包含 ID1、ID2、ID3、ID4")
    result = {}
    for addr in JOINT_IDS:
        angle = finite(joints[addr], f"ID{addr} 角度")
        bounds = config.joint_limits[addr]
        if bounds is None:
            result[addr] = angle
            continue
        lo, hi = bounds
        if angle < lo - 1e-8 or angle > hi + 1e-8:
            raise ValueError(f"ID{addr} 目标 {angle:.3f}° 超出限位 [{lo}, {hi}]")
        result[addr] = min(hi, max(lo, angle))  # 仅消除浮点边界误差。
    return result
