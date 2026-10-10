"""以三个物品的中间一个对齐画面中心，复用小球的连续位置控制。"""
from config import OBJECT_POSITION
from .ball_position import _calibrate_layout_position, _validate_config


def validate_config(config=None):
    return _validate_config(config, defaults=OBJECT_POSITION,
                            config_name="OBJECT_POSITION", label="物品")


def calibrate_object_position(board, vision, *, config=None, stop_event=None, log=print):
    """按画面从左到右取中间物品，偏左后退、偏右前进。"""
    return _calibrate_layout_position(board, vision, config=config, stop_event=stop_event, log=log,
                                      kind="object", label="物品", validator=validate_config)
