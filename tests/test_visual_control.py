import math
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, call
from car_nodes.base import vision_align as va, vision_straight as vs

def heading():
    def edge(x):
        return {'a': 0.0, 'b': x, 'near_y': 719.0, 'far_y': 100.0,
                'bottom_x': x, 'far_x': x, 'confidence': 1.0}
    info = {'size': [1280, 720], 'left': edge(340), 'right': edge(940), 'predict_ms': 0.0}
    cam = Mock()
    cam.sample.return_value = vs.BoundarySample(info, 1, 1, 'ready', '', time.time(), time.time())
    cam.age.return_value = 0
    return vs.VisionHeading(vs.VisionCfg(rate_src='gap'), cam)

class VisionControlTests(unittest.TestCase):
    def test_actual_straight_loop_reaches_distance_and_writes_log(self):
            board = Mock()
            board.feedback.side_effect = [([0]*4, None), ([10]*4, None), *[([1000]*4, None)]*4]
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'straight.csv'
                distance, info = vs.vision_straight(board, heading(), goal_mm=100,
                                                   log_path=path, log=lambda _: None)
                self.assertGreater(distance, 100)
                self.assertFalse(info['reason'])
                self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))
                self.assertTrue(path.read_text().startswith('t,dist_mm,e_deg'))

    def test_actual_align_loop_settles_and_writes_log(self):
            board = Mock()
            board.feedback.return_value = ([0]*4, None)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'align.csv'
                result = va.vision_align(board, heading(), va.AlignCfg(settle=0),
                                         log_path=path, log=lambda _: None)
                self.assertTrue(result['ok'])
                self.assertEqual(result['err'], 0)
                self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))
                self.assertTrue(path.read_text().startswith('t,err_deg'))

    def test_lost_vision_stops_straight_loop(self):
            board = Mock()
            board.feedback.return_value = ([0]*4, None)
            hd = heading()
            hd.cam.age.side_effect = [0, hd.cfg.lost_stop + 1]
            _, info = vs.vision_straight(board, hd, goal_mm=100, log=lambda _: None)
            self.assertIn('没有新边界', info['reason'])
            self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_settling_feedback_failure_closes_motion_log(self):
            board = Mock()
            board.feedback.side_effect = [([0]*4, None), ([1000]*4, None), OSError('disconnected')]
            run_log = Mock()
            with patch.object(vs, '_RunLog', return_value=run_log), self.assertRaises(OSError):
                vs.vision_straight(board, heading(), goal_mm=100, log=lambda _: None)
            run_log.close.assert_called_once()
            self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_interrupt_stops_both_control_loops(self):
            for action in (vs.vision_straight, va.vision_align):
                with self.subTest(action=action):
                    board = Mock()
                    board.feedback.side_effect = [([0]*4, None), KeyboardInterrupt()]
                    with self.assertRaises(KeyboardInterrupt):
                        action(board, heading(), log=lambda _: None)
                    self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

class VanishingGeometryTests(unittest.TestCase):
    def test_intersection_outside_image_is_used_without_clamping(self):
        cfg = vs.VisionCfg(lookahead="vanishing")
        # 交点来自直线本身，不能依赖上游缓存的灭点字段。
        info = lane(vp_x=700, vp_y=-200)
        info["vanishing_point"] = [float("nan"), 0]
        view = vs.track_view(cfg, info)
        self.assertEqual((view.x_look, view.look_row), (700, -200))
        self.assertAlmostEqual(view.e_deg, math.degrees(math.atan2(-60, cfg.focal(1280))))
        self.assertEqual(view.e_deg, view.head_deg)

    def test_vanishing_heading_does_not_correct_lateral_offset(self):
        centered = lane()
        offset = lane(left_a=-0.7, right_a=0.3)
        cfg = vs.VisionCfg(lookahead="vanishing")
        self.assertAlmostEqual(vs.track_view(cfg, centered).e_deg, 0)
        self.assertAlmostEqual(vs.track_view(cfg, offset).e_deg, 0)
        # 数值前视模式仍能看到相同航向下的横向偏移。
        view = vs.track_view(vs.VisionCfg(lookahead=0.25), offset)
        self.assertAlmostEqual(view.look_row, 180)
        self.assertAlmostEqual(view.x_look, 614)
        self.assertGreater(view.e_deg, 0)

    def test_missing_side_or_unstable_intersection_is_unusable(self):
        cfg = vs.VisionCfg(lookahead="vanishing")
        for side in ("left", "right"):
            info = lane()
            info[side] = None
            with self.subTest(side=side):
                self.assertIsNone(vs.track_view(cfg, info, half_px=100))
        for slope in (0.5, 0.5 + 1e-8):
            with self.subTest(slope=slope):
                self.assertIsNone(vs.track_view(cfg, lane(left_a=0.5, right_a=slope)))
        info = lane()
        info["left"]["b"], info["right"]["b"] = 1e308, -1e308
        self.assertIsNone(vs.track_view(cfg, info))

    def test_invalid_new_frame_clears_previous_heading(self):
        cam = camera(lane(vp_x=700))
        heading = vs.VisionHeading(vs.VisionCfg(lookahead="vanishing"), cam)
        self.assertIsNotNone(heading.step(now=10.0))
        info = lane()
        info["right"] = None
        cam.update(info, frames=2, state="ready", now=10.1)
        self.assertIsNone(heading.step(now=10.1))
        self.assertIsNone(heading.view)
        self.assertIsNone(heading.step(now=10.2))
        cam.update(lane(), frames=3, state="ready", now=10.3)
        self.assertIsNotNone(heading.step(now=10.3))

    def test_loss_of_intersection_stops_actual_motion_loop(self):
        cam = camera(lane(vp_x=620))
        heading = vs.VisionHeading(vs.VisionCfg(lookahead="vanishing"), cam)
        self.assertIsNotNone(heading.step(now=10.0))
        cam.update(lane(left_a=0.5, right_a=0.5), frames=2, state="ready", now=10.1)
        board = Mock()
        board.feedback.return_value = ([0] * 4, None)
        with patch.object(vs.time, "time", return_value=10.1):
            _, info = vs.vision_straight(board, heading, goal_mm=100, log=lambda _: None)
        self.assertIn("没有可用灭点", info["reason"])
        self.assertEqual(board.spd.call_args_list, [call(0, 0, 0, 0)])

    def test_actual_motion_turns_toward_vanishing_point(self):
        for vp_x, direction in ((620, 1), (660, -1)):
            with self.subTest(vp_x=vp_x):
                cam = camera(lane(vp_x=vp_x))
                heading = vs.VisionHeading(vs.VisionCfg(lookahead="vanishing"), cam)
                board = Mock()
                board.feedback.side_effect = [([0] * 4, None), ([10] * 4, None),
                                              *[([1000] * 4, None)] * 4]
                with patch.object(vs.time, "time", return_value=10.1):
                    _, info = vs.vision_straight(board, heading, goal_mm=100, log=lambda _: None)
                self.assertFalse(info["reason"])
                left, _, right, _ = board.spd.call_args_list[0].args
                self.assertGreater(direction * (right - left), 0)
                self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))


def lane(vp_x=640.0, vp_y=50.0, left_a=-0.5, right_a=0.5):
    def edge(a):
        return {"a": a, "b": vp_x - a * vp_y,
                "near_y": 719.0, "far_y": 100.0, "confidence": 1.0}
    return {"size": [1280, 720], "left": edge(left_a), "right": edge(right_a)}

def camera(info):
    cam = Mock()
    def update(info, frames, state, now):
        cam.sample.return_value = vs.BoundarySample(info, frames, frames, state, '', now, now)
    cam.update.side_effect = update
    cam.age.side_effect = lambda now=None: (time.time() if now is None else now) - cam.sample().t_valid
    cam.update(info, frames=1, state="ready", now=10.0)
    return cam
