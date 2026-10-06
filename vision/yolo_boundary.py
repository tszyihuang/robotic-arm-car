"""YOLO Pose 跑道边界：BGR 画面输入，原图坐标的左右线段输出。"""
from __future__ import annotations

import math
import numpy as np

from config import VISION
from .boundary import validate_keypoint_pairs
from .yolo import YoloPredictor, read_args


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
            "near_y": float(near[1]), "far_y": float(far[1]),
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


class BoundaryPredictor(YoloPredictor):
    def __init__(self, weights=VISION["boundary_weights"], device=VISION["infer_device"], fp16=VISION["fp16"],
                 threshold=None, args_path=VISION["boundary_args"]):
        args = read_args(args_path, "pose")
        self.keypoint_pairs = validate_keypoint_pairs(args.get("boundary_keypoint_pairs"))
        super().__init__(weights, device, fp16, threshold, args, "pose")
        if self.model.model.yaml.get("kpt_shape") != [4, 2]:
            raise ValueError("Expected a YOLO pose checkpoint with kpt_shape=[4, 2]")
        if self.model.names != {0: "lane_pair"}:
            raise ValueError("Expected a YOLO pose checkpoint with the lane_pair class")

    def _decode(self, result, size):
        height, width = size
        left = right = None
        warnings = []
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
                candidate = decode_boundaries(keypoints[index], confidence, width, height,
                                              self.keypoint_pairs)
                warnings.extend(candidate[2])
                if candidate[0] is not None or candidate[1] is not None:
                    left, right = candidate[:2]
                    break
        return {"endpoint_order": "near_to_far", "left": left, "right": right,
                "warnings": warnings, "detections": detections}
