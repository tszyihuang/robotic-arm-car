"""取物、放物任务位置；路线、观察姿态和动作待填写，尚未接入主线。"""


def run(arm, vision, shape):
    position = vision.observe_target("object", shape)
    if position == "left":
        left(arm)
    elif position == "middle":
        middle(arm)
    elif position == "right":
        right(arm)
    else:
        raise ValueError(f"视觉未明确物体位置：{position!r}")


def left(arm):
    raise NotImplementedError("抓左边的物体：取物、放物关节角及夹爪动作未填写")


def middle(arm):
    raise NotImplementedError("抓中间的物体：取物、放物关节角及夹爪动作未填写")


def right(arm):
    raise NotImplementedError("抓右边的物体：取物、放物关节角及夹爪动作未填写")
