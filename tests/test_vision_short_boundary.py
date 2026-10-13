"""任一边界短于 60px 时跳过整帧，验证两个任务入口和单帧恢复。"""
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
from tests.test_visual_control import camera
from vision.boundary import BoundarySample


def boundary_info(left_length=200, right_length=200, *, near_y=700, height=720):
    def edge(x, length):
        return {"a": 0.0, "b": x, "near_y": near_y, "far_y": near_y - length,
                "confidence": 1.0}
    return {"size": [1280, height], "left": edge(340, left_length),
            "right": edge(900, right_length)}


class ShortBoundaryGeometryTests(unittest.TestCase):
    def test_either_short_side_rejects_the_frame_and_exactly_60_is_usable(self):
        cfg = vs.VisionCfg(min_boundary_length_px=60)
        for left, right, expected in ((59, 200, True), (200, 59, True), (59, 59, True),
                                      (60, 200, False), (200, 60, False), (60, 60, False)):
            with self.subTest(left=left, right=right):
                self.assertEqual(vs.has_short_boundary(cfg, boundary_info(left, right)), expected)

    def test_length_is_euclidean_in_original_pixels_and_independent_of_y_and_resolution(self):
        cfg = vs.VisionCfg(min_boundary_length_px=60)
        for height in (400, 720, 1080):
            for near_y in (80, height - 1):
                with self.subTest(height=height, near_y=near_y):
                    self.assertTrue(vs.has_short_boundary(
                        cfg, boundary_info(59, 200, near_y=near_y, height=height)))
        info = boundary_info(50)
        info["left"]["a"] = 1  # 纵向跨度 50px，实际长度约 70.7px，仍可使用。
        self.assertFalse(vs.has_short_boundary(cfg, info))
        info["left"].update(a=0.75, far_y=652)  # 36²+48²=60²。
        self.assertFalse(vs.has_short_boundary(cfg, info))
        info["left"]["far_y"] = 653
        self.assertTrue(vs.has_short_boundary(cfg, info))

    def test_single_detected_side_is_checked(self):
        for side in ("left", "right"):
            info = boundary_info(59, 59)
            info[side] = None
            with self.subTest(missing=side):
                self.assertTrue(vs.has_short_boundary(vs.VisionCfg(), info))

    def test_threshold_can_be_disabled_and_invalid_values_are_rejected(self):
        self.assertFalse(vs.has_short_boundary(vs.VisionCfg(min_boundary_length_px=0),
                                              boundary_info(1, 1)))
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "MIN_BOUNDARY_LENGTH_PX"):
                vs.VisionCfg(min_boundary_length_px=value)

    def test_short_frame_clears_cached_visual_error_until_a_new_usable_frame(self):
        cam = camera(boundary_info())
        heading = vs.VisionHeading(vs.VisionCfg(min_boundary_length_px=60), cam)
        self.assertIsNotNone(heading.step(now=10.0, reject_short=True))
        cam.update(boundary_info(59), frames=2, state="ready", now=10.1)
        self.assertIsNone(heading.step(now=10.1, reject_short=True))
        self.assertTrue(heading.short_boundary)
        self.assertIsNone(heading.view)
        self.assertIsNone(heading.step(now=10.2, reject_short=True))
        self.assertEqual(heading.n_short, 1)  # 重读不重复计数，也不能沿用旧视觉误差。
        cam.update(boundary_info(60), frames=3, state="ready", now=10.3)
        self.assertIsNotNone(heading.step(now=10.3, reject_short=True))
        self.assertFalse(heading.short_boundary)

    def test_short_frame_does_not_update_cached_lane_width_or_affect_align(self):
        cam = camera(boundary_info())
        heading = vs.VisionHeading(vs.VisionCfg(), cam)
        heading.step(now=10.0, reject_short=True)
        width = heading.half_px
        info = boundary_info(59)
        info["right"]["b"] = 1000
        cam.update(info, frames=2, state="ready", now=10.1)
        self.assertIsNone(heading.step(now=10.1, reject_short=True))
        self.assertEqual(heading.half_px, width)
        align_heading = vs.VisionHeading(vs.VisionCfg(), cam)
        self.assertIsNotNone(align_heading.step(now=10.1))


class ShortBoundaryMotionTests(unittest.TestCase):
    def run_command(self, command, updates, *, encoder_loss=False, camera_error=False, rate_src="gap"):
        clock = [10.0]
        cam = camera(boundary_info())
        raw_camera = shared_camera()
        cam.start_camera.return_value = raw_camera
        heading = vs.VisionHeading(vs.VisionCfg(kp_rate=0, rate_src=rate_src,
                                              min_boundary_length_px=60), cam)
        base = Base()
        base.config["go_ahead_stop_y"] = 292
        board = base.motor = Mock()
        imu = Mock(yaw=0.0, yaw_rate=0.0)
        imu.age.return_value = 0
        stopped, last_totals = [False], [[0] * 4]
        packets = [([0] * 4, None)]
        for tick, update in enumerate(updates, 1):
            packets.append(([50 * tick + 20, 50 * tick + 20,
                             50 * tick - 20, 50 * tick - 20], update))
        if encoder_loss:
            packets.extend([(None, None)] * 30)
        else:
            packets.append(([1020, 1020, 980, 980], None))
            if command == "go-ahead":
                packets.extend([([1020, 1020, 980, 980], None)] * 30)
        pending = iter(enumerate(packets))

        def feedback(*args):
            clock[0] += 0.02
            if stopped[0]:
                return last_totals[0], None
            tick, (totals, update) = next(pending)
            if totals is not None:
                last_totals[0] = totals
            if tick:
                if update is not None:
                    cam.sample.return_value = BoundarySample(
                        update["info"], tick + 1, tick + 1, "ready", "", clock[0],
                        clock[0] - update.get("age", 0.0), update.get("age", 0.0))
                publish(raw_camera, 292 if tick > len(updates) else 270, clock[0], height=720)
                if camera_error and tick > len(updates):
                    raw_camera.error = "摄像头断开"
            return totals, None

        def speed(*values):
            if values == (0, 0, 0, 0):
                stopped[0] = True

        motion_control = vs.vision_straight

        def control(*args, **kwargs):
            return motion_control(*args, imu=imu if rate_src == "gyro" else None, **kwargs)

        board.feedback.side_effect = feedback
        board.spd.side_effect = speed
        try:
            with tempfile.TemporaryDirectory() as folder, \
                    contextlib.redirect_stdout(io.StringIO()) as output, \
                    patch.dict(BASE, meters_per_count=0.001), \
                    patch.dict(VISION_CONTROL, LOG_DIR=Path(folder)), \
                    patch.dict(RUNWAY_ENDPOINT, DETECTOR_CONFIG), \
                    patch.object(vs.time, "time", side_effect=lambda: clock[0]), \
                    patch.object(base, "_heading", return_value=heading), \
                    patch("base.api.vision_straight.vision_straight", wraps=control):
                if command == "vision-straight":
                    result = base.vision_straight(1.0, cam)
                else:
                    result = base.go_ahead(cam)
                with result["vision"]["log_path"].open() as stream:
                    rows = list(csv.DictReader(stream))
        finally:
            self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))
        return result, board, heading, rows, output.getvalue()

    def test_both_commands_skip_only_short_frames_and_restore_on_the_next_good_frame(self):
        for command in ("vision-straight", "go-ahead"):
            for rate_src in ("gap", "gyro"):
                with self.subTest(command=command, rate_src=rate_src):
                    result, board, _, rows, _ = self.run_command(command, [
                        {"info": boundary_info()}, {"info": boundary_info(59)},
                        None, {"info": boundary_info(60)}, {"info": boundary_info(200, 59)},
                        {"info": boundary_info()},
                    ], rate_src=rate_src)
                    self.assertTrue(result["ok"])
                    self.assertEqual(result["vision"]["short_frames"], 2)
                    self.assertEqual(result["vision"]["heading_mode"], "vision")
                    self.assertIsNone(result["vision"]["finish_gap_ref_mm"])
                    self.assertEqual([row["src"] for row in rows[:6]],
                                     ["both", "gap-short", "gap-short", "both", "gap-short", "both"])
                    self.assertEqual(result["vision"]["vision_recoveries"], 2)
                    for row in rows:
                        if row["src"] == "gap-short":
                            self.assertEqual(float(row["trim_mms"]), 0)
                            self.assertEqual((row["lat_px"], row["target_x"], row["offset_px"]), ("", "", ""))
                        else:
                            self.assertGreater(float(row["trim_mms"]), 0)
                    self.assertTrue(all(command.args != (0, 0, 0, 0)
                                        for command in board.spd.call_args_list[:-1]))
                    if command == "vision-straight":
                        self.assertEqual(result["distance_m"], 1.0)
                    else:
                        self.assertTrue(result["vision"]["endpoint"]["valid"])
                        self.assertEqual(result["vision"]["endpoint"]["point"][1], 292)
                        self.assertEqual(float(rows[-1]["endpoint_target_mms"]), 0)
                        self.assertEqual(float(rows[-1]["endpoint_error_px"]), 0)

    def test_old_y_condition_no_longer_disables_visual_control(self):
        _, _, _, rows, _ = self.run_command("vision-straight", [{"info": boundary_info()}])
        self.assertEqual(rows[0]["src"], "both")  # 两侧全在 y≥108，长度合格仍纠偏。

    def test_short_frames_do_not_satisfy_recovery_after_a_real_visual_outage(self):
        result, _, _, rows, _ = self.run_command("vision-straight", [
            {"info": None, "age": 1}, {"info": boundary_info(59)},
            {"info": boundary_info()}, {"info": boundary_info()},
        ])
        self.assertEqual([row["src"] for row in rows], ["gap-blind"] * 3 + ["both"])
        self.assertEqual(result["vision"]["vision_recoveries"], 1)

    def test_outage_following_a_short_frame_still_requires_normal_recovery(self):
        result, _, _, rows, _ = self.run_command("vision-straight", [
            {"info": boundary_info(59)}, {"info": None, "age": 1},
            {"info": boundary_info()}, {"info": boundary_info()},
        ])
        self.assertEqual([row["src"] for row in rows], ["gap-short", "gap-blind", "gap-blind", "both"])
        self.assertEqual(result["vision"]["vision_recoveries"], 1)

    def test_encoder_loss_and_go_ahead_camera_error_still_stop(self):
        with self.assertRaisesRegex(RuntimeError, "编码器数据中断"):
            self.run_command("vision-straight", [{"info": boundary_info(59)}], encoder_loss=True)
        with self.assertRaisesRegex(RuntimeError, "摄像头断开"):
            self.run_command("go-ahead", [{"info": boundary_info(59)}] * 2, camera_error=True)
