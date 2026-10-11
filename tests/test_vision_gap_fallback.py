"""视觉直走丢失边界时的编码器盲走，以及连续新帧恢复。"""
import csv
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from base import vision_straight as vs
from base.control import MotionCancelled
from config import BASE
from tests.test_visual_control import camera, lane
from vision.boundary import BoundarySample


class VisionGapFallbackTests(unittest.TestCase):
    def drive(self, packets, *, cfg=None, imu=None, stop_event=None):
        """用真实控制环消费定时的编码器/视觉反馈，返回指令与 CSV。"""
        cam = camera(lane())
        cfg = cfg or vs.VisionCfg(lookahead=0.5, speed=150, kp_rate=0)
        heading = vs.VisionHeading(cfg, cam)
        clock = [10.0]
        # 第一条建立编码器原点；末尾到达目标并反馈停车后的相同里程。
        pending = iter([([0] * 4, None), *packets, *[([1000] * 4, None)] * 4])

        def feedback(*args):
            totals, update = next(pending)
            clock[0] += 0.1
            if update is not None:
                if update.get('cancel'):
                    stop_event.set()
                else:
                    seq = update.get('seq', cam.sample().seq + 1)
                    age = update.get('age', 0.0)
                    cam.sample.return_value = BoundarySample(
                        update['info'], seq, seq, 'ready', '', clock[0], clock[0] - age, age)
            return totals, None

        board = Mock()
        board.feedback.side_effect = feedback
        messages = []
        with tempfile.TemporaryDirectory() as folder, \
                patch.dict(BASE, meters_per_count=0.001), \
                patch.object(vs.time, 'time', side_effect=lambda: clock[0]):
            path = Path(folder) / 'gap.csv'
            try:
                distance, info = vs.vision_straight(board, heading, imu=imu, goal_mm=1000,
                                                   log=messages.append, log_path=path,
                                                   stop_event=stop_event)
            finally:
                self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
        return distance, info, board, rows, messages

    def test_frozen_boundary_switches_once_and_completes_distance(self):
        distance, info, board, rows, messages = self.drive(
            [([40 * i] * 4, None) for i in range(1, 21)])
        self.assertEqual(distance, 1000)
        self.assertFalse(info['reason'])
        self.assertEqual(info['heading_mode'], 'gap-blind')
        self.assertEqual(info['blind_switches'], 1)
        self.assertEqual(info['vision_recoveries'], 0)
        self.assertGreater(info['lost_max'], 0.6)
        self.assertEqual(rows[0]['src'], 'both')
        self.assertEqual(rows[-1]['src'], 'gap-blind')
        self.assertEqual(sum('切换 gap 盲走' in message for message in messages), 1)
        for row in rows:
            if row['src'] == 'gap-blind':
                self.assertEqual((row['lat_px'], row['target_x'], row['offset_px']), ('', '', ''))
        self.assertTrue(all(any(value != 0 for value in command.args)
                            for command in board.spd.call_args_list[:-1]))

    def test_blind_reference_is_current_gap_and_corrects_both_directions(self):
        for rate_src in ('gap', 'gyro'):
            for dir_sign in (-1, 1):
                with self.subTest(rate_src=rate_src, dir_sign=dir_sign):
                    cfg = vs.VisionCfg(lookahead=0.5, speed=150, kp_rate=0,
                                       rate_src=rate_src, dir_sign=dir_sign)
                    imu = Mock(yaw=0.0, yaw_rate=0.0)
                    imu.age.return_value = 0
                    distance, info, board, rows, _ = self.drive([
                        ([60, 60, 40, 40], None),
                        ([120, 120, 80, 80], {'info': None, 'age': 1}),
                        ([175, 175, 125, 125], None),
                        ([215, 215, 185, 185], None),
                    ], cfg=cfg, imu=imu if rate_src == 'gyro' else None)
                    self.assertEqual(distance, 1000)
                    self.assertFalse(info['reason'])
                    self.assertEqual(info['blind_gap_ref_mm'], 40)
                    self.assertEqual([row['src'] for row in rows],
                                     ['both', 'gap-blind', 'gap-blind', 'gap-blind'])
                    commands = board.spd.call_args_list
                    self.assertEqual(commands[1].args[0], commands[1].args[2])
                    self.assertGreater(commands[2].args[2], commands[2].args[0])
                    self.assertLess(commands[3].args[2], commands[3].args[0])

    def test_two_fresh_complete_frames_required_and_missing_frame_resets_confirmation(self):
        cfg = vs.VisionCfg(lookahead=0.5, speed=150, lateral_offset_mm=50, kp_rate=0)
        _, info, board, rows, messages = self.drive([
            ([50] * 4, {'info': None, 'age': 1}),
            ([100] * 4, {'info': lane()}),
            ([150] * 4, None),  # 重读同一帧不能作为第二次确认。
            ([200] * 4, {'info': None}),  # 新的缺边界帧打断连续确认。
            ([250] * 4, {'info': lane()}),
            ([300] * 4, None),
            ([350] * 4, {'info': lane()}),
        ], cfg=cfg)
        self.assertFalse(info['reason'])
        self.assertEqual(info['heading_mode'], 'vision')
        self.assertEqual(info['blind_switches'], 1)
        self.assertEqual(info['vision_recoveries'], 1)
        self.assertEqual([row['src'] for row in rows], ['gap-blind'] * 6 + ['both'])
        self.assertGreater(board.spd.call_args_list[6].args[2], board.spd.call_args_list[6].args[0])
        self.assertGreater(float(rows[-1]['offset_px']), 0)
        self.assertTrue(any('恢复视觉纠偏' in message for message in messages))

    def test_repeated_or_out_of_order_capture_does_not_confirm_recovery(self):
        _, info, _, rows, _ = self.drive([
            ([50] * 4, {'info': None, 'age': 1}),
            ([100] * 4, {'info': lane()}),
            ([150] * 4, {'info': lane(), 'age': 0.1}),  # 新 seq，但采集时刻重复。
            ([200] * 4, {'info': lane(), 'age': 0.3}),  # 更老的采集时刻。
            ([250] * 4, {'info': lane()}),
            ([300] * 4, {'info': lane()}),
        ])
        self.assertFalse(info['reason'])
        self.assertEqual(info['vision_recoveries'], 1)
        self.assertEqual([row['src'] for row in rows], ['gap-blind'] * 5 + ['both'])

    def test_repeated_losses_latch_a_new_gap_reference_each_time(self):
        _, info, _, rows, _ = self.drive([
            ([60, 60, 40, 40], {'info': None, 'age': 1}),
            ([110, 110, 90, 90], {'info': lane()}),
            ([160, 160, 140, 140], {'info': lane()}),
            ([230, 230, 170, 170], {'info': None, 'age': 1}),
            ([280, 280, 220, 220], None),
        ])
        self.assertFalse(info['reason'])
        self.assertEqual(info['blind_switches'], 2)
        self.assertEqual(info['vision_recoveries'], 1)
        self.assertEqual(info['blind_gap_ref_mm'], 60)
        self.assertEqual([row['src'] for row in rows],
                         ['gap-blind', 'gap-blind', 'both', 'gap-blind', 'gap-blind'])
        self.assertEqual(float(rows[3]['e_raw_deg']), 0)

    def test_blind_mode_uses_encoder_rate_even_when_gyro_is_stale(self):
        cfg = vs.VisionCfg(lookahead=0.5, speed=150, rate_src='gyro', kp_rate=0)
        imu = Mock(yaw=0.0, yaw_rate=999.0)
        imu.age.return_value = 999
        distance, info, _, _, _ = self.drive([
            ([50] * 4, {'info': None, 'age': 1}), ([100] * 4, None),
        ], cfg=cfg, imu=imu)
        self.assertEqual(distance, 1000)
        self.assertFalse(info['reason'])
        self.assertEqual(info['heading_mode'], 'gap-blind')
        imu.age.assert_not_called()

    def test_blind_mode_still_stops_on_encoder_loss_or_excess_heading_error(self):
        for packets, fault in (([([50] * 4, {'info': None, 'age': 1}),
                                 *[(None, None)] * 7], '编码器数据中断'),
                               ([([50] * 4, {'info': None, 'age': 1}),
                                 ([600, 600, 200, 200], None)], 'gap 航向误差')):
            with self.subTest(fault=fault):
                _, info, _, _, _ = self.drive(packets)
                self.assertIn(fault, info['reason'])

    def test_blind_mode_still_times_out_if_wheels_make_no_progress(self):
        _, info, _, _, _ = self.drive([
            ([0] * 4, {'info': None, 'age': 1}), *[([0] * 4, None)] * 260,
        ])
        self.assertIn('总时长超时', info['reason'])

    def test_cancellation_stops_blind_drive(self):
        event = threading.Event()
        with self.assertRaises(MotionCancelled):
            self.drive([([50] * 4, {'info': None, 'age': 1}),
                        ([100] * 4, {'cancel': True})], stop_event=event)

    def test_permanent_finish_gap_does_not_resume_visual_after_recovery(self):
        cfg = vs.VisionCfg(lookahead=0.5, speed=150, gap_finish_mm=300, kp_rate=0)
        _, info, _, rows, _ = self.drive([
            ([100] * 4, {'info': None, 'age': 1}),
            ([720] * 4, None),
            ([800] * 4, {'info': lane()}),
            ([850] * 4, {'info': lane()}),
        ], cfg=cfg)
        self.assertFalse(info['reason'])
        self.assertEqual(info['heading_mode'], 'gap')
        self.assertEqual(info['vision_recoveries'], 0)
        self.assertEqual([row['src'] for row in rows], ['gap-blind', 'gap', 'gap', 'gap'])

    def test_recovery_frame_count_must_be_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                vs.VisionCfg(gap_recovery_frames=value)
