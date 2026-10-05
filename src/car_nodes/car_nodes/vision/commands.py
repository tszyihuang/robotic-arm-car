"""视觉节点内部请求语法；不将调试命令暴露为主线运动指令。"""
import shlex
from ..planner.decision import COLORS, SHAPES


def internal_command(text):
    parts = shlex.split(text)
    if not parts or parts[0] not in ('start-boundary', 'start-camera', 'set-models', 'observe-target'):
        return None
    name, kwargs = parts[0], {}
    for token in parts[1:]:
        key, separator, value = token.partition('=')
        if not separator or key in kwargs:
            raise ValueError('参数必须为不重复的 key=value')
        kwargs[key] = value
    if name in ('start-camera', 'start-boundary'):
        if kwargs:
            raise ValueError(f'{name} 不接受参数')
    elif name == 'set-models':
        if set(kwargs) != {'boundary', 'objects'} or any(v not in ('true', 'false') for v in kwargs.values()):
            raise ValueError('set-models 需要 boundary=true/false objects=true/false')
        kwargs = {key: value == 'true' for key, value in kwargs.items()}
    elif (set(kwargs) != {'kind', 'value'} or kwargs['kind'] not in ('ball', 'target', 'object')
          or kwargs['value'] not in (SHAPES if kwargs['kind'] == 'object' else COLORS)):
        raise ValueError('observe-target 需要有效的 kind=ball/target/object 和 value=颜色或形状')
    return name, kwargs
