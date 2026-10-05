"""YOLO 物体检测：BGR 原始画面输入，原图坐标的检测框输出。"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from .paths import OBJECT_MODEL_DIR
from .boundary_config import read_args


class ObjectPredictor:
    def __init__(self, weights=OBJECT_MODEL_DIR / "weights.pt", device="cuda:0", fp16=True,
                 threshold=None, args_path=OBJECT_MODEL_DIR / "args.yaml"):
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"找不到 YOLO 权重文件 {weights}")
        args = read_args(args_path, task="detect")
        self.threshold = threshold if threshold is not None else args.get("conf")
        if self.threshold is None:
            self.threshold = 0.25
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
        self.model = YOLO(str(weights), task="detect")
        if self.model.task != "detect":
            raise ValueError("Expected a YOLO detection checkpoint")
        self.names = dict(self.model.names)
        self.imgsz = check_imgsz(imgsz, stride=int(self.model.model.stride.max()), min_dim=2)
        self.config = {"imgsz": self.imgsz, "conf": self.threshold, "iou": iou,
                       "max_det": max_det, "names": self.names}

    def _run(self, frame):
        return self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.threshold, iou=self.config["iou"],
            max_det=self.config["max_det"], device=str(self.device),
            quantize="fp16" if self.fp16 else None,
            rect=True, save=False, verbose=False,
        )[0]

    def warmup(self, count=5):
        frame = np.zeros((*self.imgsz, 3), dtype=np.uint8)
        for _ in range(count):
            self._run(frame)

    def predict(self, frame):
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or 0 in frame.shape):
            raise ValueError("Expected a nonempty uint8 BGR HxWx3 frame")
        start = time.perf_counter()
        result = self._run(frame)
        detections = []
        if result.boxes is not None:
            for box in result.boxes.data.cpu().numpy():
                xyxy, confidence, class_value = box[:4], float(box[-2]), float(box[-1])
                if (not np.isfinite(box).all() or confidence < self.threshold
                        or not class_value.is_integer() or xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]):
                    continue
                class_id = int(class_value)
                if class_id not in self.names:
                    continue
                detections.append({"class_id": class_id, "name": self.names[class_id],
                                   "confidence": confidence, "box": xyxy.tolist()})
        height, width = frame.shape[:2]
        speed = result.speed
        return {"size": [width, height], "coordinate_units": "original_image_pixels",
                "detections": detections,
                "timing_ms": {"preprocess": float(speed.get("preprocess") or 0),
                              "network": float(speed.get("inference") or 0),
                              "postprocess": float(speed.get("postprocess") or 0),
                              "total": (time.perf_counter() - start) * 1000}}
