"""标靶稳定排序、后台识别和位置记录；不连接实车。"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import main
from base.control import MotionCancelled
from config import ROOT
from debug.dry_run import DryVision
from tasks import detect_target
from tasks.runner import Step, execute, load_section
from tests.test_vision_api import ContinuousCamera
from vision.api import Vision
from vision.targets import colored_targets, observe


def candidates(colors):
    return list(reversed([{"value": color, "name": color,
                           "box": [i * 100, 10, i * 100 + 40, 50],
                           "center_x": i * 100 + 20}
                          for i, color in enumerate(colors)]))


class TargetDetectionTests(unittest.TestCase):
    def test_layout_rejects_stale_missing_duplicate_and_overlapping_targets(self):
        camera = Mock(condition=threading.Condition(), index=10)
        layouts = [candidates(colors) for colors in [
            ("green", "red", "blue"), ("red", "green"), ("red", "red", "blue"),
            ("red", "green", "blue"), ("red", "green", "blue"),
            ("green", "red", "blue"), ("green", "red", "blue"),
            ("green", "red", "blue")]]
        layouts[3][0]["center_x"] = layouts[3][1]["center_x"]
        now = time.time()
        source = Mock(side_effect=[(rows, index, now - 2 if index == 11 else now)
                                   for index, rows in enumerate(layouts, 11)])
        result = observe(camera, "target", None, stable_frames=3, candidate_source=source)
        self.assertEqual(result["frame_index"], 18)
        self.assertEqual([row["value"] for row in result["candidates"]],
                         ["green", "red", "blue"])
        self.assertEqual([call.kwargs["after"] for call in source.call_args_list],
                         list(range(10, 18)))
        camera.next_frame.assert_not_called()

    def test_task_records_real_background_target_coordinates_without_motion(self):
        camera, vision = ContinuousCamera(), Vision()
        base, arm = Mock(), Mock()
        threads = []

        def detect(frame, min_area_ratio):
            threads.append(threading.current_thread().name)
            return colored_targets(frame, min_area_ratio)

        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "target_positions.json"
                with patch("vision.api.CameraStream", return_value=camera), \
                        patch("vision.api.colored_targets", side_effect=detect), \
                        patch("tasks.detect_target.DEFAULT_POSITIONS_FILE", path), \
                        patch("tasks.runner.pause"), contextlib.redirect_stdout(io.StringIO()) as output:
                    result = execute(Step(1, "detect-targets", ()), base, arm, vision)
                self.assertEqual([row["position"] for row in result], ["left", "middle", "right"])
                self.assertEqual([row["color"] for row in result], ["green", "red", "blue"])
                self.assertEqual([(row["center_x"], row["center_y"]) for row in result],
                                 [(90.0, 120.0), (290.0, 120.0), (490.0, 120.0)])
                self.assertEqual(result[0]["box"], [30, 60, 150, 180])
                self.assertEqual(json.loads(path.read_text())["targets"], result)
                for expected in ("标靶位置：左，颜色：绿色", "标靶位置：中，颜色：红色",
                                 "标靶位置：右，颜色：蓝色", "标靶位置已记录"):
                    self.assertIn(expected, output.getvalue())
                self.assertEqual(set(threads), {"vision-inference"})
                self.assertIsNone(vision._object_model)
                self.assertIsNone(vision._boundary_model)
                self.assertFalse(vision._target_requested.is_set())
                self.assertEqual(base.mock_calls, [])
                self.assertEqual(arm.mock_calls, [])
        finally:
            vision.close()

    def test_missing_targets_timeout_preserves_record_and_stops_detection_request(self):
        camera, vision = ContinuousCamera(), Vision()
        camera.frame[60:180, 430:550] = 0
        vision.config["observe_timeout"] = 0.1
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "target_positions.json"
                path.write_text("previous record", encoding="utf-8")
                with patch("vision.api.CameraStream", return_value=camera), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    with self.assertRaisesRegex(TimeoutError, "当前识别到 2 个"):
                        detect_target.run(vision, output_path=path)
                self.assertEqual(path.read_text(), "previous record")
                self.assertEqual(output.getvalue(), "")
                self.assertFalse(vision._target_requested.is_set())
        finally:
            vision.close()

    def test_cancellation_after_observation_does_not_overwrite_record(self):
        stop = threading.Event()
        vision = Mock(stop_event=stop)

        def cancel():
            stop.set()
            return []

        vision.observe_targets.side_effect = cancel
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target_positions.json"
            path.write_text("previous record", encoding="utf-8")
            with self.assertRaises(MotionCancelled):
                detect_target.run(vision, output_path=path)
            self.assertEqual(path.read_text(), "previous record")

    def test_selected_section_starts_vision_and_dry_run_leaves_record_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "target_positions.json"
            path.write_text("previous record", encoding="utf-8")
            tasks_path = Path(folder) / "tasks.txt"
            tasks_path.write_text("[识别标靶]\ndetect-targets\n", encoding="utf-8")
            vision, base, arm = DryVision(), Mock(), Mock()
            with patch.object(vision, "start") as start, \
                    patch("tasks.detect_target.DEFAULT_POSITIONS_FILE", path), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                main.run(base, arm, vision, tasks_path=tasks_path, section="识别标靶")
            start.assert_called_once()
            self.assertIn("vision.observe_targets()", output.getvalue())
            self.assertNotIn("标靶位置已记录", output.getvalue())
            self.assertEqual(path.read_text(), "previous record")
            arm.calibrate.assert_not_called()
            arm.move_joints.assert_not_called()
            base.calibrate_ball_position.assert_not_called()
            base.straight.assert_not_called()
            steps = load_section(tasks_path, "识别标靶")
            self.assertEqual([step.command for step in steps], ["detect-targets"])
