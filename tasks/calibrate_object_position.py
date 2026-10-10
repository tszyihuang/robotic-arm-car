"""任务表的物品位置校准入口；与小球共用干跑和停车逻辑。"""
from base.object_position import validate_config
from .calibrate_ball_position import _run


def run(base, vision, *, stop_event=None, config=None, log=print):
    return _run(base, vision, stop_event=stop_event, config=config, log=log,
                kind="object", label="物品", validator=validate_config)
