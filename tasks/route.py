"""比赛路线，逐条保留原任务表的顺序和数值。"""
from . import pause


def to_qrcode(base):
    base.straight(0.3)
    pause(base)


def to_ball(base, vision):
    base.turn(87, 0.38)
    pause(base)
    base.turn(-45, 0.52)
    pause(base)
    base.turn(-42, 0)
    pause(base)
    base.straight(0.7)
    pause(base)
    base.calibrate_position()
    pause(base)
    base.turn(44, 0.24)
    pause(base)
    base.turn(44, 0)
    pause(base)
    base.straight(0.46)
    pause(base)
    base.turn(-87, 0.36)
    pause(base)
    base.straight(0.27)
    pause(base)
    base.turn(-87, 0.36)
    pause(base)
    base.align(vision)
    pause(base)
    base.vision_straight(0.45, vision)
    pause(base)


def after_ball(base, vision):
    base.vision_straight(1.76, vision)
    pause(base)
    base.straight(0.13)
    pause(base)
    base.turn(-87, 0.43)
    pause(base)
    base.vision_straight(0.67, vision)
    pause(base)
    base.vision_straight(1.55, vision)


def to_target(base, vision):
    raise NotImplementedError("打靶路线未填写；请先补充实际路线参数")


def to_delivery(base, vision):
    raise NotImplementedError("取物、放物路线未填写；请先补充实际路线参数")
