"""YOLO Pose 跑道边界：BGR 画面输入，原图坐标的左右线段输出。"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.checks import check_imgsz

from config import VISION
from .boundary_config import read_args, validate_keypoint_pairs


def _boundary(points, confidence, width, height):
    """把同一侧的两个关键点按近到远排序，求 x=a*y+b。"""
    points = np.asarray(points, dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        return None
    if (points < 0).any() or (points > [width, height]).any():
        return None
    near, far = points[np.argsort(points[:, 1])[::-1]]
    span = near[1] - far[1]
    if span < 1.0:
        return None
    a = (near[0] - far[0]) / span
    b = far[0] - a * far[1]
    return {"points": [near.tolist(), far.tolist()], "a": float(a), "b": float(b),
            "confidence": float(confidence)}


def decode_boundaries(keypoints, confidence, width, height, pairs):
    """连接已指定的左右点对；相交、颠倒或无共同纵向范围时拒绝整对。"""
    keypoints = np.asarray(keypoints, dtype=float)
    if keypoints.shape != (4, 2):
        raise ValueError("Expected four 2D lane keypoints")
    left, right = [_boundary(keypoints[list(pair)], confidence, width, height)
                   for pair in pairs]
    warnings = [f"invalid_{side}_keypoints" for side, line in (("left", left), ("right", right))
                if line is None]
    if left is not None and right is not None:
        low = max(left["points"][1][1], right["points"][1][1])
        high = min(left["points"][0][1], right["points"][0][1])
        if high <= low:
            left = right = None
            warnings.append("no_common_y_span")
        elif any(left["a"] * y + left["b"] >= right["a"] * y + right["b"]
                 for y in (low, high)):
            left = right = None
            warnings.append("crossing_or_swapped_sides_rejected")
    return left, right, warnings


class BoundaryPredictor:
    def __init__(self, weights=VISION["boundary_weights"], device=VISION["infer_device"], fp16=VISION["fp16"],
                 threshold=None, args_path=VISION["boundary_args"], allow_unconfigured=False):
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"找不到 YOLO 权重文件 {weights}")
        args = read_args(args_path)
        # 训练参数里的 data/model/project/device 等不参与运行时推理。
        pairs = args.get("boundary_keypoint_pairs")
        if pairs is None and not allow_unconfigured:
            raise ValueError("请在 args.yaml 配置 boundary_keypoint_pairs")
        self.keypoint_pairs = validate_keypoint_pairs(pairs) if pairs is not None else None
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
        self.model = YOLO(str(weights), task="pose")
        if self.model.task != "pose" or self.model.model.yaml.get("kpt_shape") != [4, 2]:
            raise ValueError("Expected a YOLO pose checkpoint with kpt_shape=[4, 2]")
        if self.model.names != {0: "lane_pair"}:
            raise ValueError("Expected a YOLO pose checkpoint with the lane_pair class")
        self.imgsz = check_imgsz(imgsz, stride=int(self.model.model.stride.max()), min_dim=2)
        self.config = {"imgsz": self.imgsz, "conf": self.threshold, "iou": iou,
                       "max_det": max_det, "boundary_keypoint_pairs": pairs}

    def set_keypoint_pairs(self, pairs):
        pairs = validate_keypoint_pairs(pairs)
        self.config["boundary_keypoint_pairs"] = pairs
        self.keypoint_pairs = pairs

    def _run(self, frame):
        return self.model.predict(
            source=frame, imgsz=self.imgsz, conf=self.threshold, iou=self.config["iou"],
            max_det=self.config["max_det"], device=str(self.device),
            quantize="fp16" if self.fp16 else None,
            rect=True, save=False, verbose=False,
        )[0]

    def warmup(self, count=5):
        """用实际 BGR 预处理和推理流程热机；CPU 使用 FP32。"""
        frame = np.zeros((*self.imgsz, 3), dtype=np.uint8)
        for _ in range(count):
            self._run(frame)

    def predict(self, frame):
        """接收 OpenCV uint8 BGR HWC 画面；YOLO 负责 letterbox 和原图坐标还原。"""
        if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                or frame.ndim != 3 or frame.shape[2] != 3 or 0 in frame.shape):
            raise ValueError("Expected a nonempty uint8 BGR HxWx3 frame")
        start = time.perf_counter()
        pairs = self.keypoint_pairs
        result = self._run(frame)
        height, width = frame.shape[:2]
        left = right = None
        warnings = [] if pairs is not None else ["keypoint_pairs_not_configured"]
        detections = []
        if result.keypoints is not None and len(result.boxes):
            keypoints = result.keypoints.xy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()
            # 同一组中的四个点构成跑道；不拼接不同检测框的左右边界。
            for index in np.argsort(confidences)[::-1]:
                confidence = float(confidences[index])
                if not math.isfinite(confidence) or confidence < self.threshold:
                    continue
                detections.append({"box": boxes[index].tolist(), "confidence": confidence,
                                   "keypoints": keypoints[index].tolist()})
                if pairs is None:
                    continue
                candidate = decode_boundaries(keypoints[index], confidence, width, height,
                                              pairs)
                warnings.extend(candidate[2])
                if candidate[0] is not None or candidate[1] is not None:
                    left, right = candidate[:2]
                    break
        speed = result.speed
        finished = time.perf_counter()
        return {"size": [width, height], "coordinate_units": "original_image_pixels",
                "endpoint_order": "near_to_far", "left": left, "right": right,
                "warnings": warnings, "detections": detections,
                "timing_ms": {"preprocess": float(speed.get("preprocess") or 0),
                              "network": float(speed.get("inference") or 0),
                              "postprocess": float(speed.get("postprocess") or 0),
                              "total": (finished - start) * 1000}}
