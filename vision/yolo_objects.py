"""YOLO 物体检测：BGR 输入，输出按置信度去除重叠的原图检测框。"""
from __future__ import annotations

import numpy as np

from config import VISION
from .yolo import YoloPredictor, read_args


class ObjectPredictor(YoloPredictor):
    def __init__(self, weights=VISION["objects_weights"], device=VISION["infer_device"], fp16=VISION["fp16"],
                 threshold=None, args_path=VISION["objects_args"]):
        super().__init__(weights, device, fp16, threshold, read_args(args_path, "detect"), "detect")
        self.names = dict(self.model.names)

    def _decode(self, result, size):
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
        # 跨类别按置信度去重；有面积交集就抑制，边缘接触不算重叠。
        kept = []
        for detection in sorted(detections, key=lambda row: row["confidence"], reverse=True):
            x1, y1, x2, y2 = detection["box"]
            if any(max(x1, row["box"][0]) < min(x2, row["box"][2])
                   and max(y1, row["box"][1]) < min(y2, row["box"][3])
                   for row in kept):
                continue
            kept.append(detection)
        return {"detections": kept}
