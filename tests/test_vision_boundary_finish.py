"""以完整跑道边界的原图纵坐标关闭纠偏；模拟两个任务入口，不连接实车。"""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from base import vision_straight as vs
from base.api import Base
from config import BASE, RUNWAY_ENDPOINT, VISION_CONTROL
from tests.test_go_ahead import DETECTOR_CONFIG, publish, shared_camera
from tests.test_visual_control import camera, lane
from vision.boundary import BoundarySample


def boundary_info(left_far, right_far=None, *, height=720):
    info = lane(vp_x=620)
    info["size"][1] = height
    for name, far in (("left", left_far), ("right", left_far if right_far is None else right_far)):
        info[name].update(far_y=far, near_y=height - 1)
    return info


class BoundaryFinishGeometryTests(unittest.TestCase):
    def test_both_complete_segments_must_reach_15_percent_row_at_each_resolution(self):
        cfg = vs.VisionCfg()
        for height in (400, 720, 1080):
            threshold = height * 0.15
            for left, right, expected in ((threshold, threshold, True),
                                          (threshold + 10, threshold, True),
                                          (threshold - 1, threshold, False),
                                          (threshold, threshold - 1, False)):
                with self.subTest(height=height, left=left, right=right):
                    self.assertEqual(vs.boundaries_below_finish(
                        cfg, boundary_info(left, right, height=height)), expected)

    def test_short_segments_above_threshold_and_incomplete_or_invalid_segments_do_not_trigger(self):
        cfg = vs.VisionCfg(min_conf=0.5)
        cases = [None, {}]
        short = boundary_info(10)
        for side in ("left", "right"):
            short[side]["near_y"] = 50  # 长度不足 15% 也不能代替 y≥15%H 的位置判断。
        cases.append(short)
        for side in ("left", "right"):
            missing = boundary_info(108)
            missing[side] = None
            cases.append(missing)
            for key in ("far_y", "near_y"):
                for value in (None, float("nan"), float("inf"), "bad", 721):
                    info = boundary_info(108)
                    info[side][key] = value
                    cases.append(info)
            low_confidence = boundary_info(108)
            low_confidence[side]["confidence"] = 0.4
            cases.append(low_confidence)
        for info in cases:
            with self.subTest(info=info):
                self.assertFalse(vs.boundaries_below_finish(cfg, info))

    def test_threshold_can_be_disabled_and_invalid_ratios_are_rejected(self):
        self.assertFalse(vs.boundaries_below_finish(vs.VisionCfg(gap_finish_y_ratio=0),
                                                  boundary_info(700)))
        for value in (-0.1, 1.1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "GAP_FINISH_Y_RATIO"):
                vs.VisionCfg(gap_finish_y_ratio=value)

    def test_only_fresh_ready_frames_can_trigger(self):
        for state, error, age in (("ready", "", 1), ("error", "摄像头断开", 0)):
            cam = camera(boundary_info(108))
            cam.sample.return_value = BoundarySample(boundary_info(108), 1, 1, state, error,
                                                    10.0, 10.0 - age, age)
            heading = vs.VisionHeading(vs.VisionCfg(), cam)
            heading.step(now=10.0)
            self.assertFalse(heading.boundary_finish)

        cam = camera(boundary_info(108))
        heading = vs.VisionHeading(vs.VisionCfg(), cam)
        heading.step(now=10.0)
        self.assertTrue(heading.boundary_finish)
        heading.step(now=10.1)  # 重读同一帧。
        self.assertFalse(heading.boundary_finish)
        cam.update(boundary_info(108), frames=2, state="ready", now=9.9)
        heading.step(now=10.1)  # 序号新但采集时刻倒退。
        self.assertFalse(heading.boundary_finish)

    def test_valid_segments_trigger_even_without_a_usable_vanishing_point(self):
        info = boundary_info(108)
        info["left"]["a"] = info["right"]["a"] = 0
        cam = camera(info)
        heading = vs.VisionHeading(vs.VisionCfg(lookahead="vanishing"), cam)
        self.assertIsNone(heading.step(now=10.0))
        self.assertTrue(heading.boundary_finish)


class BoundaryFinishMotionTests(unittest.TestCase):
    def run_command(self, command, *, initially_blind=False, encoder_loss=False):
        clock = [10.0]
        cam = camera(boundary_info(107))
        raw_camera = shared_camera()
        cam.start_camera.return_value = raw_camera
        heading = vs.VisionHeading(vs.VisionCfg(kp_rate=0), cam)
        heading.step = Mock(wraps=heading.step)
        base = Base()
        board = base.motor = Mock()
        stopped, last_totals = [False], [[0] * 4]
        packets = [([0] * 4, None),
                   ([60, 60, 40, 40], None if initially_blind else boundary_info(107)),
                   ([120, 120, 80, 80], boundary_info(108)),
                   ([175, 175, 125, 125], boundary_info(50)),  # 新帧恢复长边界也不重启纠偏。
                   ([215, 215, 185, 185], None),
                   ([1020, 1020, 980, 980], None)]
        if encoder_loss:
            packets = packets[:3] + [(None, None)] * 30
        pending = iter(enumerate(packets))

        def feedback(*args):
            clock[0] += 0.02
            if stopped[0]:
                return last_totals[0], None
            tick, (totals, info) = next(pending)
            if totals is not None:
                last_totals[0] = totals
            if tick:
                cam.sample.return_value = BoundarySample(
                    info, tick + 1, tick + 1, "ready", "", clock[0], clock[0])
                publish(raw_camera, 292 if tick >= 5 else 270, clock[0], height=720)
            return totals, None

        def speed(*values):
            if values == (0, 0, 0, 0):
                stopped[0] = True

        board.feedback.side_effect = feedback
        board.spd.side_effect = speed
        with tempfile.TemporaryDirectory() as folder, \
                contextlib.redirect_stdout(io.StringIO()) as output, \
                patch.dict(BASE, meters_per_count=0.001), \
                patch.dict(VISION_CONTROL, LOG_DIR=Path(folder)), \
                patch.dict(RUNWAY_ENDPOINT, DETECTOR_CONFIG), \
                patch.object(vs.time, "time", side_effect=lambda: clock[0]), \
                patch.object(base, "_heading", return_value=heading):
            if command == "vision-straight":
                result = base.vision_straight(1.0, cam)
            else:
                result = base.go_ahead(cam)
            with result["vision"]["log_path"].open() as stream:
                rows = list(csv.DictReader(stream))
        return result, board, heading, rows, output.getvalue()

    def test_both_commands_latch_current_heading_and_keep_their_stop_condition(self):
        for command in ("vision-straight", "go-ahead"):
            with self.subTest(command=command):
                result, board, heading, rows, messages = self.run_command(command)
                info = result["vision"]
                self.assertTrue(result["ok"])
                self.assertEqual(info["heading_mode"], "gap")
                self.assertEqual(info["finish_gap_trigger"], "boundary-y")
                self.assertEqual(info["finish_gap_ref_mm"], 40)
                self.assertEqual(info["vision_recoveries"], 0)
                self.assertEqual(heading.step.call_count, 2)
                self.assertEqual([row["src"] for row in rows], ["both", "gap", "gap", "gap"])
                commands = board.spd.call_args_list
                self.assertGreater(commands[0].args[2], commands[0].args[0])
                self.assertEqual(commands[1].args[0], commands[1].args[2])
                self.assertGreater(commands[2].args[2], commands[2].args[0])
                self.assertLess(commands[3].args[2], commands[3].args[0])
                self.assertEqual(commands[-1].args, (0, 0, 0, 0))
                self.assertEqual(messages.count("关闭视觉纠偏"), 1)
                if command == "vision-straight":
                    self.assertEqual(result["distance_m"], 1.0)
                else:
                    self.assertTrue(info["endpoint"]["valid"])
                    self.assertEqual(info["endpoint"]["point"][1], 292)

    def test_boundary_threshold_overrides_temporary_blind_recovery(self):
        result, _, _, rows, _ = self.run_command("vision-straight", initially_blind=True)
        self.assertEqual(result["vision"]["heading_mode"], "gap")
        self.assertEqual(result["vision"]["finish_gap_ref_mm"], 40)
        self.assertEqual(result["vision"]["blind_switches"], 1)
        self.assertEqual(result["vision"]["vision_recoveries"], 0)
        self.assertEqual([row["src"] for row in rows], ["gap-blind", "gap", "gap", "gap"])

    def test_encoder_loss_after_visual_cutoff_still_fails_command(self):
        with self.assertRaisesRegex(RuntimeError, "编码器数据中断"):
            self.run_command("vision-straight", encoder_loss=True)
