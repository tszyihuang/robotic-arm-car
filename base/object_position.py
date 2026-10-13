"""将指定形状物品对齐画面中心，复用小球的连续位置控制。"""
from config import OBJECT_POSITION
from .ball_position import _calibrate_layout_position, _validate_config


def validate_config(config=None):
    return _validate_config(config, defaults=OBJECT_POSITION,
                            config_name="OBJECT_POSITION", label="物品")


def calibrate_object_position(board, vision, *, target_shape=None, config=None, stop_event=None, log=print):
    """按二维码指定形状校准；独立调试未指定形状时仍取中间物品。"""
    return _calibrate_layout_position(board, vision, config=config, stop_event=stop_event, log=log,
                                      kind="object", label="物品", validator=validate_config,
                                      target_value=target_shape)
