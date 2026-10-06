"""模型使用假推理结果，验证共享推理入口和实际边界几何接口。"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from base.vision_straight import VisionCfg, track_view
from vision.boundary import valid_geometry
from vision.yolo_boundary import BoundaryPredictor, decode_boundaries
from vision.yolo_objects import ObjectPredictor


def tensor(values):
    return SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(values)))


class YoloTests(unittest.TestCase):
    def test_pose_and_detection_share_bgr_inference_and_timing(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for predictor_type, task, names in ((BoundaryPredictor, "pose", {0: "lane_pair"}),
                                           (ObjectPredictor, "detect", {0: "红球"})):
            with self.subTest(task=task):
                model = Mock(task=task, names=names)
                model.model.stride = np.array([32])
                model.model.yaml = {"kpt_shape": [4, 2]}
                model.predict.return_value = [SimpleNamespace(
                    boxes=None, keypoints=None, speed={"inference": 12.5})]
                with patch("vision.yolo.YOLO", return_value=model):
                    predictor = predictor_type(device="cpu", fp16=True)
                result = predictor.predict(frame)
                self.assertEqual(result["size"], [1280, 720])
                self.assertEqual(result["detections"], [])
                self.assertEqual(result["timing_ms"]["network"], 12.5)
                self.assertGreaterEqual(result["timing_ms"]["total"], 0)
                options = model.predict.call_args.kwargs
                self.assertIs(options["source"], frame)
                self.assertEqual(options["conf"], 0.25)
                self.assertIsNone(options["quantize"])
                self.assertFalse(predictor.fp16)
                for invalid in (None, np.zeros((2, 2)), np.zeros((2, 2, 3), dtype=float)):
                    with self.assertRaises(ValueError):
                        predictor.predict(invalid)
                self.assertEqual(model.predict.call_count, 1)

    def test_decoded_lane_limits_lookahead_to_detected_endpoints(self):
        left, right, warnings = decode_boundaries(
            [[340, 700], [500, 500], [940, 650], [780, 450]],
            0.9, 1280, 720, [[0, 1], [2, 3]])
        info = {"size": [1280, 720], "left": left, "right": right}
        self.assertTrue(valid_geometry(info))
        self.assertFalse(warnings)
        view = track_view(VisionCfg(lookahead=0.5), info)
        self.assertEqual((view.y_lo, view.y_hi, view.look_row), (500, 650, 500))

    def test_invalid_lane_pairs_and_single_valid_side(self):
        for points in ([[700, 700], [500, 300], [400, 700], [600, 300]],
                       [[340, 700], [500, 600], [940, 500], [780, 400]]):
            left, right, warnings = decode_boundaries(points, 0.9, 1280, 720, [[0, 1], [2, 3]])
            self.assertIsNone(left)
            self.assertIsNone(right)
            self.assertTrue(warnings)
        left, right, warnings = decode_boundaries(
            [[340, 700], [500, 300], [-1, 700], [780, 300]],
            0.9, 1280, 720, [[0, 1], [2, 3]])
        self.assertIsNotNone(left)
        self.assertIsNone(right)
        self.assertEqual(warnings, ["invalid_right_keypoints"])

    def test_object_decoder_rejects_invalid_boxes_and_unknown_classes(self):
        predictor = ObjectPredictor.__new__(ObjectPredictor)
        predictor.names, predictor.threshold = {0: "红球"}, 0.25
        result = SimpleNamespace(boxes=SimpleNamespace(data=tensor([
            [10, 20, 30, 40, 0.9, 0], [10, 20, 30, 40, 0.1, 0],
            [10, 20, 10, 40, 0.9, 0], [10, 20, 30, 40, 0.9, 1],
            [10, 20, 30, 40, 0.9, 0.5], [float("nan"), 20, 30, 40, 0.9, 0],
        ])))
        detections = predictor._decode(result, (720, 1280))["detections"]
        self.assertEqual(detections, [{"class_id": 0, "name": "红球", "confidence": 0.9,
                                       "box": [10, 20, 30, 40]}])
