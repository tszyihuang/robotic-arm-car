import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from car_nodes.arm.session import ArmSession
from car_nodes.base.sensor_source import SensorSource
from car_nodes.base.board import ControllerBoard
from car_nodes.base import controller, straight_pid
from car_nodes.common.control import MotionCancelled
from car_nodes.planner.tasks import default_task_file, prepare_tasks

ROOT = Path(__file__).resolve().parents[1]


def header(stamp):
    sec, nanosec = divmod(round(stamp * 1e9), 1_000_000_000)
    return NS(stamp=NS(sec=sec, nanosec=nanosec))


def imu(stamp, delta=0.0):
    return NS(yaw_stamp=stamp, accel_stamp=stamp, yaw_deg=42.0,
              yaw_rate_dps=3.0, gyro_rate_dps=4.0, acceleration=[0, 0, 0],
              accel_sequence=1, accel_delta=delta)


class MissionTests(unittest.TestCase):
    def test_all_shipped_tables_validate(self):
        self.assertEqual(default_task_file(), ROOT / 'tasks.txt')
        for table in ROOT.glob('*.txt'):
            self.assertTrue(prepare_tasks(table.read_text()))
        main = prepare_tasks((ROOT / 'tasks.txt').read_text())
        self.assertIn('抓球任务', [s.command for s in main])
        self.assertNotIn('gripper-open', [s.command for s in main])

    def test_invalid_later_step_rejects_complete_mission(self):
        for text in ('straight 0.3\nunknown', 'arm-move 0 0 160 24',
                     'arm-calibrate\narm-move 0 0 170 24', 'straight 0.3\nturn nan',
                     'scan-qrcode timeout=0', 'scan-qrcode device=-1', 'straight 0'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                prepare_tasks(text)

    def test_source_cli_works_from_other_directory_without_ros(self):
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'plan.py'), '--dry-run'],
                                cwd='/tmp', capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('vision: scan-qrcode', result.stdout)

    def test_arm_task1_runs_in_simulation_with_persistent_zero(self):
        actions = {'arm-calibrate': 'calibrate', 'arm-move': 'move_joints',
                   'gripper-open': 'open_gripper', 'gripper-close': 'close_gripper',
                   'arm-home': 'home', 'arm-disable': 'disable'}
        session = ArmSession(simulate=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                for step in prepare_tasks('arm-calibrate\ngripper-open\narm-move -95 133 134 -83\n'
                                          'gripper-close\narm-move 90 82 142 -56\ngripper-open\narm-home'):
                    self.assertTrue(getattr(session, actions[step.command])(*step.args)['ok'])
            self.assertTrue(session.calibrated)
            self.assertEqual(session._arm.get_joints(), session.config.joint_offsets_deg)
        finally:
            session.close()


class SensorTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.1
        self.source = SensorSource(clock=lambda: self.now)

    def test_duplicate_imu_cannot_renew_watchdog(self):
        self.source.ingest_imu(imu(100.0))
        self.now = 101.0
        self.source.ingest_imu(imu(100.0))
        self.assertFalse(self.source.has_data())
        self.assertGreater(self.source.accel_state().age, 0.3)

    def test_single_sample_impact_remains_latched_after_calm_samples(self):
        self.source.ingest_imu(imu(100.0))
        self.source.arm_impact()
        self.source.ingest_imu(imu(100.01, 151.0))
        self.source.ingest_imu(imu(100.02, 0.0))
        self.assertTrue(self.source.accel_state().hit)
        self.assertEqual(self.source.accel_state().peak, 151.0)
        self.source.disarm_impact()

    def test_stale_frames_cannot_start_calibration(self):
        self.source.ingest_imu(imu(98.0, 999.0))
        with self.assertRaises(RuntimeError):
            self.source.arm_impact()

    def test_encoder_is_consumed_once_and_keeps_capture_age(self):
        message = NS(header=header(100), sequence=1, totals=[20] * 4,
                     increments=[1] * 4, has_increments=True)
        self.source.ingest_encoders(message)
        board = ControllerBoard(NS(error=None), self.source)
        self.assertEqual(board.feedback(0), ([20] * 4, [1] * 4))
        self.assertEqual(board.feedback(0), (None, None))
        self.now = 101.0
        self.source.ingest_encoders(message)
        self.assertEqual(ControllerBoard(NS(error=None), self.source).feedback(0), (None, None))


class ControllerTests(unittest.TestCase):
    def test_actual_gap_loop_corrects_left_lead_in_both_directions(self):
        for direction in (1, -1):
            board = Mock()
            board.feedback.side_effect = [([2000] * 4, None),
                ([2000 + direction * 100, 2000 + direction * 100,
                  2000 + direction * 80, 2000 + direction * 80], None),
                *[([2000 + direction * 1000] * 4, None)] * 5]
            board.stop.side_effect = lambda: board.spd(0, 0, 0, 0)
            _, info = straight_pid.straight(board, direction * 100, 100, log=lambda _: None)
            self.assertFalse(info['reason'])
            command = next(c.args for c in board.spd.call_args_list if any(c.args))
            self.assertGreater((command[2] - command[0]) * direction, 0)
            self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))

    def test_controller_failure_and_cancel_stop_motor(self):
        step = prepare_tasks('straight 0.3')[0]
        for error in (OSError('disconnected'), MotionCancelled('cancel')):
            board = Mock()
            with patch.object(straight_pid, 'straight', side_effect=error), self.assertRaises(type(error)):
                controller.execute(step, board, None, None, threading.Event())
            board.stop.assert_called()


class QrTests(unittest.TestCase):
    def test_real_decoder_scans_shared_frame(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('需要本机 OpenCV/numpy')
        from car_nodes.vision.qrcode import scan_qrcode
        qr = cv2.QRCodeEncoder_create().encode('car:123+321')
        qr = cv2.resize(qr, (400, 400), interpolation=cv2.INTER_NEAREST)
        frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        frame[340:740, 760:1160] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        camera = Mock()
        camera.next_frame.return_value = frame, 1, time.time()
        self.assertEqual(scan_qrcode(camera), 'car:123+321')
        camera.close.assert_not_called()

    def test_scan_cancel_does_not_wait_for_frame(self):
        from car_nodes.vision.qrcode import scan_qrcode
        stop = threading.Event()
        stop.set()
        camera = Mock()
        with self.assertRaises(MotionCancelled):
            scan_qrcode(camera, stop_event=stop)
        camera.next_frame.assert_not_called()
