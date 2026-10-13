"""根据救援物体校准前的位置，视觉直走到终点。"""
from .mission import POSITION_NAMES


DISTANCES = {"left": 1.84, "middle": 1.69, "right": 1.54}


def run(base, vision, *, position=None):
    if position is None and getattr(base, "dry_run", False):
        print("  走向终点：按救援物体校准前的位置选择视觉直走距离："
              "左边 1.84m，中间 1.69m，右边 1.54m；实际位置等待现场识别。", flush=True)
        return None
    if position not in DISTANCES:
        raise ValueError("go-to-finish 前需要完成人质任务，并确认救援物体校准前的左、中、右位置")
    distance = DISTANCES[position]
    print(f"  走向终点：救援物体原位置={POSITION_NAMES[position]}，视觉直走 {distance:g}m。", flush=True)
    return base.vision_straight(distance, vision)
