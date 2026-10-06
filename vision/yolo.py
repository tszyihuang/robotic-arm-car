"""两套 YOLO 模型共用的配置、BGR 输入校验与推理流程。"""
import math
from pathlib import Path
import time

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz


def read_args(path, task):
    import yaml
    try:
        with Path(path).open(encoding="utf-8") as stream:
            args = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"args.yaml 解析失败：{exc}") from exc
    if not isinstance(args, dict) or args.get("task") != task:
        raise ValueError(f"args.yaml must describe a YOLO {task} model")
    return args


class YoloPredictor:
    def __init__(self, weights, device, fp16, threshold, args, task):
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"找不到 YOLO 权重文件 {weights}")
        self.threshold = threshold if threshold is not None else args.get("conf")
        if self.threshold is None:
            self.threshold = 0.25
        if not math.isfinite(self.threshold) or not 0 < self.threshold < 1:
            raise ValueError("YOLO confidence threshold must be between 0 and 1")
        iou = args.get("iou", 0.7)
        if not math.isfinite(iou) or not 0 < iou <= 1:
            raise ValueError("YOLO IoU threshold must be between 0 and 1")
        imgsz = args.get("imgsz", 640)
        dims = [imgsz] if isinstance(imgsz, int) else imgsz
        if (not isinstance(dims, list) or len(dims) not in (1, 2)
                or any(type(value) is not int or value <= 0 for value in dims)):
            raise ValueError("YOLO imgsz must be a positive integer or [height, width]")
        max_det = args.get("max_det", 300)
        if type(max_det) is not int or max_det < 1:
            raise ValueError("YOLO max_det must be a positive integer")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; install matching PyTorch or use device='cpu'.")
        self.fp16 = bool(fp16 and self.device.type == "cuda")
        self.model = YOLO(str(weights), task=task)
        if self.model.task != task:
            raise ValueError(f"Expected a YOLO {task} checkpoint")
        self.imgsz = check_imgsz(imgsz, stride=int(self.model.model.stride.max()), min_dim=2)
        self.config = {"imgsz": self.imgsz, "conf": self.threshold, "iou": iou, "max_det": max_det}

    def predict(self, frame):
        """YOLO 负责 letterbox 和原图坐标还原，子类仅解码各自结果。"""
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or 0 in frame.shape):
            raise ValueError("Expected a nonempty uint8 BGR HxWx3 frame")
        start = time.perf_counter()
        result = self.model.predict(
            source=frame, **self.config, device=str(self.device),
            quantize="fp16" if self.fp16 else None, rect=True, save=False, verbose=False,
        )[0]
        info = self._decode(result, frame.shape[:2])
        height, width = frame.shape[:2]
        speed = result.speed
        return {"size": [width, height], "coordinate_units": "original_image_pixels", **info,
                "timing_ms": {"preprocess": float(speed.get("preprocess") or 0),
                              "network": float(speed.get("inference") or 0),
                              "postprocess": float(speed.get("postprocess") or 0),
                              "total": (time.perf_counter() - start) * 1000}}

    def _decode(self, result, size):
        raise NotImplementedError
