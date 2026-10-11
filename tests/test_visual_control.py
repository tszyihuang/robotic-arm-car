import csv
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch, call
from base import vision_align as va, vision_straight as vs
from base.api import Base
from config import BASE, VISION_CONTROL
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
                cfg = vs.VisionCfg(gap_finish_mm=300)
                with patch.dict(VISION_CONTROL, LOG_DIR=None), \
                        patch.object(vs, 'VisionCfg', return_value=cfg), \
                        patch.object(base, '_heading', return_value=hd) as prepare, \
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

class LaneCenteringTests(unittest.TestCase):
    def test_task_api_saves_lateral_error_and_actual_wheel_commands(self):
        base = Base()
        board = base.motor = Mock()
        board.feedback.side_effect = [([0] * 4, None), ([10] * 4, None), *[([200] * 4, None)] * 4]
        hd = heading()
        for side in ('left', 'right'):
            hd.cam.sample().info[side]['b'] -= 40
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(BASE, meters_per_count=0.001), \
                patch.dict(VISION_CONTROL, LOG_DIR=Path(folder)), \
                patch.object(base, '_heading', return_value=hd):
            result = base.vision_straight(0.2, Mock())
            self.assertTrue(result['ok'])
            paths = list(Path(folder).glob('*.csv'))
            self.assertEqual(len(paths), 1)
            self.assertEqual(result['vision']['log_path'], paths[0])
            with paths[0].open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(float(rows[0]['lat_px']), -40)
            self.assertEqual(float(rows[0]['offset_px']), 100)
            self.assertEqual(float(rows[0]['target_x']), 740)
            self.assertGreater(float(rows[0]['cmd_r']), float(rows[0]['cmd_l']))
            self.assertEqual(rows[0]['src'], 'both')
        self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_default_straight_corrects_lateral_offset_for_short_and_long_commands(self):
        for goal in (200, 1000):
            for direction in (1, -1):
                with self.subTest(goal=goal, direction=direction), tempfile.TemporaryDirectory() as folder:
                    # 横向偏移改变两侧斜率，但灭点仍在画面中央，纯航向误差为零。
                    cam = camera(lane(left_a=-0.5 - 0.2 * direction,
                                      right_a=0.5 - 0.2 * direction))
                    hd = vs.VisionHeading(vs.VisionCfg(), cam)
                    board = Mock()
                    board.feedback.side_effect = [([0] * 4, None), ([10] * 4, None),
                                                  ([goal - 20] * 4, None), *[([goal] * 4, None)] * 4]
                    path = Path(folder) / 'center.csv'
                    with patch.dict(BASE, meters_per_count=0.001), \
                            patch.object(vs.time, 'time', return_value=10.1):
                        distance, info = vs.vision_straight(board, hd, goal_mm=goal,
                                                           log_path=path, log=lambda _: None)
                    self.assertEqual(distance, goal)
                    self.assertFalse(info['reason'])
                    self.assertEqual(info['heading_mode'], 'vision')
                    self.assertIsNone(info['finish_gap_ref_mm'])
                    self.assertAlmostEqual(hd.view.head_deg, 0)
                    commands = board.spd.call_args_list
                    self.assertEqual(len(commands), 3)
                    for command in commands[:-1]:
                        left, _, right, _ = command.args
                        self.assertGreater(direction * (right - left), 0)
                    self.assertEqual(commands[-1].args, (0, 0, 0, 0))
                    rows = path.read_text().splitlines()[1:]
                    self.assertEqual([row.rsplit(',', 1)[1] for row in rows], ['both', 'both'])

    def test_default_straight_corrects_new_lateral_offset_in_last_300mm(self):
        for direction in (1, -1):
            with self.subTest(direction=direction):
                cam = camera(lane())
                hd = vs.VisionHeading(vs.VisionCfg(), cam)
                clock = [10.0]
                frames = iter([[0] * 4, [10] * 4, [800] * 4, *[[1000] * 4] * 4])

                def feedback(*args):
                    clock[0] += 0.01
                    totals = next(frames)
                    if totals[0] == 800:
                        cam.update(lane(left_a=-0.5 - 0.2 * direction,
                                        right_a=0.5 - 0.2 * direction),
                                   frames=2, state='ready', now=clock[0])
                    return totals, None

                board = Mock()
                board.feedback.side_effect = feedback
                with patch.dict(BASE, meters_per_count=0.001), \
                        patch.object(vs.time, 'time', side_effect=lambda: clock[0]):
                    _, info = vs.vision_straight(board, hd, goal_mm=1000, log=lambda _: None)
                self.assertFalse(info['reason'])
                first, second = board.spd.call_args_list[:2]
                self.assertEqual(first.args[0], first.args[2])
                self.assertGreater(direction * (second.args[2] - second.args[0]), 0)
                self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_default_straight_still_stops_for_stale_vision_near_finish(self):
        cam = camera(lane())
        hd = vs.VisionHeading(vs.VisionCfg(), cam)
        clock = [10.0]
        frames = iter([[0] * 4, [10] * 4, [800] * 4, *[[1000] * 4] * 4])

        def feedback(*args):
            clock[0] += 0.01
            totals = next(frames)
            if totals[0] == 800:
                cam.update(None, frames=2, state='error', now=clock[0] - 1)
            return totals, None

        board = Mock()
        board.feedback.side_effect = feedback
        with patch.dict(BASE, meters_per_count=0.001), \
                patch.object(vs.time, 'time', side_effect=lambda: clock[0]):
            _, info = vs.vision_straight(board, hd, goal_mm=1000, log=lambda _: None)
        self.assertIn('没有新边界', info['reason'])
        self.assertEqual(len(board.spd.call_args_list), 2)
        self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_base_default_requires_vision_for_every_distance(self):
        for distance in (0.2, 0.3, 0.35, 1.76):
            with self.subTest(distance=distance):
                base, vision = Base(), Mock()
                base.motor, base._imu_checked = Mock(), True
                hd = Mock()
                with patch.dict(VISION_CONTROL, LOG_DIR=None), \
                        patch.object(base, '_heading', return_value=hd) as prepare, \
                        patch.object(vs, 'vision_straight', return_value=(distance * 1000, {'reason': ''})):
                    self.assertTrue(base.vision_straight(distance, vision)['ok'])
                prepare.assert_called_once_with(vision)
                hd.wait_track.assert_called_once()


class LateralCalibrationTests(unittest.TestCase):
    def test_centered_car_turns_left_or_right_for_requested_offset(self):
        neutral = vs.track_view(vs.VisionCfg(), lane())
        for offset in (-50, 0, 50):
            with self.subTest(offset=offset):
                cfg = vs.VisionCfg(lane_width_mm=300, lateral_offset_mm=offset)
                view = vs.track_view(cfg, lane())
                self.assertAlmostEqual(view.offset_px, 310 * offset / 300)
                self.assertAlmostEqual(view.target_x, 640 + view.offset_px)
                self.assertAlmostEqual(view.e_deg,
                                       math.degrees(math.atan2(view.offset_px, cfg.focal(1280))))
                self.assertEqual(view.head_deg, neutral.head_deg)
                self.assertEqual(view.lat_px, neutral.lat_px)

    def test_requested_physical_offset_is_zero_error_with_parallel_heading(self):
        for offset in (-50, 50):
            with self.subTest(offset=offset):
                # 两侧斜率同加偏移，灭点仍居中，表示车头平行而车身横移。
                info = lane(left_a=-0.5 + offset / 300, right_a=0.5 + offset / 300)
                view = vs.track_view(vs.VisionCfg(lateral_offset_mm=offset), info)
                self.assertAlmostEqual(view.e_deg, 0)
                self.assertAlmostEqual(view.head_deg, 0)
                self.assertAlmostEqual(view.lat_px, view.offset_px)

    def test_compensation_scales_with_perspective_width_and_image_resolution(self):
        for lookahead in (0.25, 0.5):
            views = []
            for scale in (0.5, 1, 2):
                info = lane()
                info['size'] = [1280 * scale, 720 * scale]
                for side in ('left', 'right'):
                    for key in ('b', 'near_y', 'far_y'):
                        info[side][key] *= scale
                cfg = vs.VisionCfg(lookahead=lookahead, lateral_offset_mm=50)
                view = vs.track_view(cfg, info)
                self.assertAlmostEqual(view.offset_px, 2 * view.half_px / 6)
                self.assertAlmostEqual(view.offset_px / scale, (lookahead * 720 - 50) / 6)
                views.append(view)
            self.assertAlmostEqual(views[0].e_deg, views[1].e_deg)
            self.assertAlmostEqual(views[1].e_deg, views[2].e_deg)

    def test_single_side_uses_cached_width_and_requires_a_width_reference(self):
        cfg = vs.VisionCfg(lateral_offset_mm=50)
        both = vs.track_view(cfg, lane())
        for side in ('left', 'right'):
            with self.subTest(side=side):
                info = lane()
                info[side] = None
                self.assertIsNone(vs.track_view(cfg, info))
                single = vs.track_view(cfg, info, half_px=both.half_px)
                self.assertTrue(single.absolute)
                self.assertAlmostEqual(single.e_deg, both.e_deg)
                self.assertAlmostEqual(single.offset_px, both.offset_px)

    def test_invalid_new_width_clears_previous_heading_and_stops_motion(self):
        cam = camera(lane())
        hd = vs.VisionHeading(vs.VisionCfg(lateral_offset_mm=50), cam)
        self.assertIsNotNone(hd.step(now=10.0))
        invalid = lane()
        invalid['left'], invalid['right'] = invalid['right'], invalid['left']
        cam.update(invalid, frames=2, state='ready', now=10.1)
        self.assertIsNone(hd.step(now=10.1))
        self.assertIsNone(hd.view)
        board = Mock()
        board.feedback.return_value = ([0] * 4, None)
        with patch.object(vs.time, 'time', return_value=10.1):
            _, result = vs.vision_straight(board, hd, goal_mm=100, log=lambda _: None)
        self.assertIn('没有可用中心线或路宽', result['reason'])
        self.assertEqual(board.spd.call_args_list, [call(0, 0, 0, 0)])

    def test_invalid_calibration_is_rejected_before_connecting_hardware(self):
        for params in ({'lane_width_mm': 0}, {'lane_width_mm': float('nan')},
                       {'lateral_offset_mm': float('inf')},
                       {'lookahead': 'vanishing', 'lateral_offset_mm': 50}):
            with self.subTest(params=params), self.assertRaises(ValueError):
                vs.VisionCfg(**params)
        base = Base()
        with patch.dict(VISION_CONTROL, LATERAL_OFFSET_MM=float('nan')), \
                patch.object(base, '_connect') as connect, self.assertRaises(ValueError):
            base.vision_straight(1, Mock())
        connect.assert_not_called()

    def test_align_keeps_original_reference_despite_straight_calibration(self):
        base, vision = Base(), Mock()
        vision.wait_ready.return_value = True
        with patch.dict(VISION_CONTROL, LATERAL_OFFSET_MM=50), \
                patch.object(base, '_connect'), \
                patch.object(va, 'vision_align', return_value={'ok': True}) as align:
            base.align(vision)
        cfg = align.call_args.args[1].cfg
        self.assertEqual(cfg.lateral_offset_mm, 0)
        self.assertAlmostEqual(vs.track_view(cfg, lane()).e_deg, 0)


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
