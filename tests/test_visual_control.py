import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch, call
from base import vision_align as va, vision_straight as vs
from base.api import Base
from config import BASE
from vision.boundary import BoundarySample

def heading(gap_finish_mm=0.0):
    def edge(x):
        return {'a': 0.0, 'b': x, 'near_y': 719.0, 'far_y': 100.0,
                'bottom_x': x, 'far_x': x, 'confidence': 1.0}
    info = {'size': [1280, 720], 'left': edge(340), 'right': edge(940), 'predict_ms': 0.0}
    cam = Mock()
    cam.sample.return_value = BoundarySample(info, 1, 1, 'ready', '', time.time(), time.time())
    cam.age.return_value = 0
    return vs.VisionHeading(vs.VisionCfg(rate_src='gap', gap_finish_mm=gap_finish_mm), cam)


class GapFinishTests(unittest.TestCase):
    def test_switch_at_300mm_holds_current_heading_and_ignores_lost_vision(self):
        for rate_src in ('gap', 'gyro'):
            with self.subTest(rate_src=rate_src), tempfile.TemporaryDirectory() as folder:
                hd = heading(gap_finish_mm=300)
                hd.cfg.rate_src = rate_src
                hd.cfg.dir_sign = -1  # gap 的物理方向不受视觉方向参数影响。
                hd.cfg.kp_rate = 0
                hd.step = Mock(wraps=hd.step)
                imu = Mock(yaw=0.0, yaw_rate=0.0)
                imu.age.return_value = 0
                clock = [100.0]
                frames = iter([
                    [0] * 4,
                    [719, 719, 679, 679],  # 剩余 301mm，仍需视觉。
                    [720, 720, 680, 680],  # 剩余 300mm，以 gap=40mm 为新基准。
                    [719, 719, 679, 679],  # 编码器微小回退，不重新切回视觉。
                    [745, 745, 695, 695],  # gap 比基准多 10mm，应向左修正。
                    *[[1020, 1020, 980, 980]] * 4,
                ])

                def feedback(*args):
                    clock[0] += 0.01
                    totals = next(frames)
                    if (totals[0] + totals[2]) / 2 >= 700:
                        hd.cam.sample.return_value = BoundarySample(None, 2, 2, 'error', '摄像头断开',
                                                                    clock[0], 0.0)
                        hd.cam.age.return_value = 999
                        imu.age.return_value = 999
                    return totals, None

                board = Mock()
                board.feedback.side_effect = feedback
                path = Path(folder) / 'switch.csv'
                with patch.dict(BASE, meters_per_count=0.001), \
                        patch.object(vs.time, 'time', side_effect=lambda: clock[0]):
                    distance, info = vs.vision_straight(board, hd, imu=imu if rate_src == 'gyro' else None,
                                                       goal_mm=1000, log_path=path, log=lambda _: None)
                self.assertEqual(distance, 1000)
                self.assertFalse(info['reason'])
                self.assertEqual(info['finish_gap_ref_mm'], 40)
                self.assertEqual(info['heading_mode'], 'gap')
                hd.step.assert_called_once()
                commands = board.spd.call_args_list
                self.assertEqual(commands[1].args[0], commands[1].args[2])
                self.assertEqual(commands[2].args[0], commands[2].args[2])
                self.assertGreater(commands[3].args[2], commands[3].args[0])
                self.assertEqual(commands[-1].args, (0, 0, 0, 0))
                rows = path.read_text().splitlines()[1:]
                self.assertEqual([row.rsplit(',', 1)[1] for row in rows], ['both', 'gap', 'gap', 'gap'])
                if rate_src == 'gyro':
                    imu.age.assert_called_once()

    def test_short_command_uses_gap_without_camera_and_corrects_both_directions(self):
        for goal in (200, 300):
            for direction in (1, -1):
                with self.subTest(goal=goal, direction=direction):
                    cam = Mock()
                    cam.sample.side_effect = AssertionError('短距离不应读取视觉')
                    cfg = vs.VisionCfg(gap_finish_mm=300, dir_sign=-1, kp_rate=0)
                    hd = vs.VisionHeading(cfg, cam)
                    board = Mock()
                    drift = [10, 10, 0, 0] if direction == 1 else [0, 0, 10, 10]
                    board.feedback.side_effect = [([0] * 4, None), (drift, None),
                                                  *[([goal] * 4, None)] * 4]
                    with patch.dict(BASE, meters_per_count=0.001):
                        distance, info = vs.vision_straight(board, hd, goal_mm=goal, log=lambda _: None)
                    self.assertEqual(distance, goal)
                    self.assertFalse(info['reason'])
                    self.assertEqual(info['finish_gap_ref_mm'], 0)
                    first = board.spd.call_args_list[0].args
                    self.assertGreater(direction * (first[2] - first[0]), 0)
                    cam.sample.assert_not_called()
                    cam.age.assert_not_called()
                    self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_encoder_loss_still_stops_in_gap_mode(self):
        hd = heading(gap_finish_mm=300)
        clock = [100.0]
        frames = iter([([0] * 4, None), ([10] * 4, None)])

        def feedback(*args):
            clock[0] += 0.1
            return next(frames, (None, None))

        board = Mock()
        board.feedback.side_effect = feedback
        with patch.object(vs.time, 'time', side_effect=lambda: clock[0]):
            _, info = vs.vision_straight(board, hd, goal_mm=200, log=lambda _: None)
        self.assertIn('编码器数据中断', info['reason'])
        self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_base_skips_visual_readiness_only_for_short_commands(self):
        for distance in (0.2, 0.3, 0.301):
            with self.subTest(distance=distance):
                base, vision = Base(), Mock()
                base.motor, base._imu_checked = Mock(), True
                hd = Mock()
                with patch.object(base, '_heading', return_value=hd) as prepare, \
                        patch.object(vs, 'vision_straight', return_value=(distance * 1000, {'reason': ''})):
                    self.assertTrue(base.vision_straight(distance, vision)['ok'])
                if distance <= 0.3:
                    prepare.assert_not_called()
                    vision.enable_boundary.assert_not_called()
                    vision.wait_ready.assert_not_called()
                else:
                    prepare.assert_called_once_with(vision)
                    hd.wait_track.assert_called_once()

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
            with patch.object(vs, 'CsvLog', return_value=run_log), self.assertRaises(OSError):
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

    def test_log_open_failure_stops_both_control_loops(self):
        for action in (vs.vision_straight, va.vision_align):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as folder:
                board = Mock()
                board.feedback.return_value = ([0] * 4, None)
                path = Path(folder) / 'missing' / 'motion.csv'
                with self.assertRaises(FileNotFoundError):
                    action(board, heading(), log_path=path, log=lambda _: None)
                self.assertEqual(board.spd.call_args_list, [call(0, 0, 0, 0)])

    def test_gyro_feedback_preserves_turn_direction_in_both_loops(self):
        for action in (vs.vision_straight, va.vision_align):
            with self.subTest(action=action):
                hd = heading()
                hd.cfg.rate_src = 'gyro'
                hd.cam.sample().info['left']['b'] += 20
                hd.cam.sample().info['right']['b'] += 20
                imu = Mock(yaw=0.0, yaw_rate=-1.0)
                imu.age.return_value = 0
                board = Mock()
                board.feedback.side_effect = [([0] * 4, None), ([10] * 4, None),
                                              *[([1000] * 4, None)] * 4]
                if action is vs.vision_straight:
                    _, info = action(board, hd, imu=imu, goal_mm=100, log=lambda _: None)
                    self.assertFalse(info['reason'])
                else:
                    info = action(board, hd, va.AlignCfg(settle=0, tol=20),
                                  imu=imu, log=lambda _: None)
                    self.assertTrue(info['ok'])
                left, _, right, _ = board.spd.call_args_list[0].args
                self.assertGreater(left, right)
                self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_stale_gyro_stops_before_sending_motion(self):
        hd = heading()
        hd.cfg.rate_src = 'gyro'
        imu = Mock(yaw=0.0, yaw_rate=0.0)
        imu.age.return_value = vs.IMU_STALE + 0.1
        board = Mock()
        board.feedback.return_value = ([0] * 4, None)
        _, info = vs.vision_straight(board, hd, imu=imu, goal_mm=100, log=lambda _: None)
        self.assertIn('IMU 数据中断', info['reason'])
        self.assertEqual(board.spd.call_args_list, [call(0, 0, 0, 0)])

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
        heading = vs.VisionHeading(vs.VisionCfg(lookahead="vanishing", gap_finish_mm=0), cam)
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
                heading = vs.VisionHeading(vs.VisionCfg(lookahead="vanishing", gap_finish_mm=0), cam)
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
        cam.sample.return_value = BoundarySample(info, frames, frames, state, '', now, now)
    cam.update.side_effect = update
    cam.age.side_effect = lambda now=None: (time.time() if now is None else now) - cam.sample().t_valid
    cam.update(info, frames=1, state="ready", now=10.0)
    return cam
