"""抬头扫码，解析比赛任务，再回到本次初始姿态。"""
from . import pause
from vision.qrcode import mission_code


def run(arm, vision):
    arm.move_joints(6, 0, 160, 1)
    pause(arm)
    data = vision.scan_qrcode()
    pause(arm)
    mission = mission_code(data)
    arm.home()
    pause(arm)
    return mission
