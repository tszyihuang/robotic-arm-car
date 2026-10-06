"""夹球动作；关节角单位 °，左、中、右直接在这里修改。"""
from . import pause


def run(arm, vision, color):
    arm.move_joints(-90, 37, 110, 62)
    pause(arm)
    position = vision.observe_target("ball", color)
    pause(arm)
    if position is None and getattr(vision, "dry_run", False):
        print("  视觉确认后：左 → 未填写；中 → 以下 7 步；右 → 未填写。")
        print("  [条件分支预览：中间小球；干跑不选择实际位置]")
        middle(arm)
    elif position == "left":
        left(arm)
    elif position == "middle":
        middle(arm)
    elif position == "right":
        right(arm)
    else:
        raise ValueError(f"视觉未明确小球位置：{position!r}")


def left(arm):
    raise NotImplementedError("抓左边的小球：关节角和夹爪动作未填写，请补充 tasks/ball.py 的 left()")


def middle(arm):
    arm.open_gripper()
    pause(arm)
    arm.move_joints(-95, 133, 134, -83)
    pause(arm)
    arm.close_gripper()
    pause(arm)
    arm.move_joints(-90, 37, 110, 62)
    pause(arm)
    arm.move_joints(90, 82, 142, -56)
    pause(arm)
    arm.open_gripper()
    pause(arm)
    arm.home()
    pause(arm)


def right(arm):
    raise NotImplementedError("抓右边的小球：关节角和夹爪动作未填写，请补充 tasks/ball.py 的 right()")
