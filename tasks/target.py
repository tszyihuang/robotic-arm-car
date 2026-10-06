"""打靶任务位置；观察姿态、路线及三个位置动作待填写，尚未接入主线。"""


def run(arm, vision, color):
    # 接入主线前，需由调用者先执行实际观察姿态与到靶路线。
    position = vision.observe_target("target", color)
    if position == "left":
        left(arm)
    elif position == "middle":
        middle(arm)
    elif position == "right":
        right(arm)
    else:
        raise ValueError(f"视觉未明确靶位置：{position!r}")


def left(arm):
    raise NotImplementedError("打左边的靶：关节角和打靶动作未填写")


def middle(arm):
    raise NotImplementedError("打中间的靶：关节角和打靶动作未填写")


def right(arm):
    raise NotImplementedError("打右边的靶：关节角和打靶动作未填写")
