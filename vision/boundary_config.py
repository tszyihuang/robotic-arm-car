"""左右边界关键点配置；不加载模型或 GPU 库。"""
from pathlib import Path


def validate_keypoint_pairs(pairs):
    if (not isinstance(pairs, list) or len(pairs) != 2
            or any(not isinstance(pair, list) or len(pair) != 2 for pair in pairs)
            or any(type(index) is not int for pair in pairs for index in pair)
            or sorted(index for pair in pairs for index in pair) != [0, 1, 2, 3]):
        raise ValueError("左右边界各需两个点，索引 0、1、2、3 必须各出现一次")
    return [list(pair) for pair in pairs]


def read_args(path, task="pose"):
    import yaml
    try:
        with Path(path).open(encoding="utf-8") as stream:
            args = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"args.yaml 解析失败：{exc}") from exc
    if not isinstance(args, dict) or args.get("task") != task:
        raise ValueError(f"args.yaml must describe a YOLO {task} model")
    return args

