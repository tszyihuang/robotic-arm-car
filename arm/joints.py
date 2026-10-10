"""关节角的完整性和有限数值校验；不设置软件角度限位。"""
from .config import JOINT_IDS, finite


def validate_joints(joints, config=None):
    if set(joints) != set(JOINT_IDS):
        raise ValueError("关节角必须完整包含 ID1、ID2、ID3、ID4")
    return {addr: finite(joints[addr], f"ID{addr} 角度") for addr in JOINT_IDS}
