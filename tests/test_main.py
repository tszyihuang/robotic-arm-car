"""任务表主线、独立抓球动作与退出清理验证；不连接实车。"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import main
from tasks import ball, target, delivery
from arm.api import Arm
from base.control import MotionCancelled

ROOT = Path(__file__).resolve().parents[1]


class Recorder:
    dry_run = True  # 测试中只去掉步骤间等待，视觉结果由测试明确给出。

    def __init__(self, trace, position="middle", qr="211"):
        self.trace, self.position, self.qr = trace, position, qr
        self.cleaned = []

    def start(self):
        pass

    def __getattr__(self, name):
        def call(*args):
            if name in ("stop", "cancel", "close"):
                self.cleaned.append(name)
                return
            command = {"calibrate": "arm-calibrate", "move_joints": "arm-move",
                       "home": "arm-home", "open_gripper": "gripper-open",
                       "close_gripper": "gripper-close", "calibrate_position": "calibrate-position",
                       "vision_straight": "vision-straight", "scan_qrcode": "scan-qrcode",
                       "observe_target": "抓球任务"}.get(name, name)
            recorded = () if name in ("observe_target", "align") else args[:1] if name == "vision_straight" else args
            self.trace.append((command, recorded))
            if name == "scan_qrcode":
                return self.qr
            if name == "observe_target":
                return self.position
        return call


class MissionTests(unittest.TestCase):
    def run_tasks(self, text, *devices):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text(text, encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main.run(*devices, tasks_path=path)

    def test_main_executes_only_selected_section_in_file_order(self):
        text = """[主线]
calibrate-position
turn 44 0.24
align
straight 0.48 # 使用任务表中的新距离
vision-straight 0.45
arm-calibrate
arm-move -90 37 110 62
gripper-open
gripper-close
home
arm-home
scan-qrcode
[抓中间的小球]
gripper-open
这段不应解析或执行
"""
        trace = []
        devices = [Recorder(trace) for _ in range(3)]
        self.run_tasks(text, *devices)
        self.assertEqual(trace, [
            ("calibrate-position", ()), ("turn", (44, 0.24)), ("align", ()),
            ("straight", (0.48,)), ("vision-straight", (0.45,)),
            ("arm-calibrate", ()), ("arm-move", (-90, 37, 110, 62)),
            ("gripper-open", ()), ("gripper-close", ()),
            ("arm-home", ()), ("arm-home", ()), ("scan-qrcode", ()),
        ])
        self.assertEqual([d.cleaned for d in devices], [["stop", "close"], ["cancel", "close"], ["close"]])

    def test_main_does_not_insert_calibration_scan_or_ball_actions(self):
        trace = []
        devices = [Recorder(trace) for _ in range(3)]
        self.run_tasks("[主线]\nstraight 0.48\n[抓中间的小球]\ngripper-open\n", *devices)
        self.assertEqual(trace, [("straight", (0.48,))])

    def test_invalid_later_command_prevents_all_motion(self):
        trace = []
        devices = [Recorder(trace) for _ in range(3)]
        with self.assertRaisesRegex(ValueError, "tasks.txt:3"):
            self.run_tasks("[主线]\nstraight 0.48\nturn 44 错误\n", *devices)
        self.assertEqual(trace, [])
        self.assertTrue(all(d.cleaned for d in devices))

    def test_cancel_before_step_does_not_send_motion(self):
        devices = [Mock() for _ in range(3)]
        stop = threading.Event()
        stop.set()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[主线]\nstraight 0.48\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(MotionCancelled):
                main.run(*devices, tasks_path=path, stop_event=stop)
        devices[0].straight.assert_not_called()
        devices[2].start.assert_not_called()

    def test_background_vision_starts_before_first_step_and_start_failure_cleans_up(self):
        base, arm, vision = Mock(), Mock(), Mock()
        calls = Mock()
        calls.attach_mock(vision.start, "start")
        calls.attach_mock(base.straight, "straight")
        self.run_tasks("[主线]\nstraight 0.48\n", base, arm, vision)
        self.assertEqual([call[0] for call in calls.mock_calls], ["start", "straight"])
        vision.start.side_effect = OSError("视觉启动失败")
        base.straight.reset_mock()
        with self.assertRaisesRegex(OSError, "视觉启动失败"):
            self.run_tasks("[主线]\nstraight 0.48\n", base, arm, vision)
        base.straight.assert_not_called()
        self.assertEqual(vision.close.call_count, 2)

    def test_middle_ball_7_steps_match_baseline(self):
        fixture = Path(__file__).with_name("fixtures") / "mission_actions.json"
        data = json.loads(fixture.read_text())
        expected = [(command, tuple(args)) for command, args in data["middle_ball"]]
        self.assertEqual(len(expected), 7)
        trace = []
        ball.middle(Recorder(trace))
        self.assertEqual(trace, expected)

    def test_missing_ball_branch_stops_before_remaining_route(self):
        for position in ("left", "right"):
            trace = []
            devices = [Recorder(trace, position) for _ in range(3)]
            with self.subTest(position=position), self.assertRaises(NotImplementedError):
                ball.run(devices[1], devices[2], "green")
            self.assertNotIn(("vision-straight", (1.76,)), trace)

    def test_all_eight_missing_actions_fail_explicitly(self):
        missing = [ball.left, ball.right, target.left, target.middle, target.right,
                   delivery.left, delivery.middle, delivery.right]
        for action in missing:
            with self.subTest(action=action), self.assertRaisesRegex(NotImplementedError, "未填写"):
                action(Mock())

    def test_qr_failure_cannot_reach_ball_route(self):
        for error in (TimeoutError("扫码超时"), ValueError("二维码无效")):
            base, arm, vision = Mock(), Mock(), Mock()
            vision.scan_qrcode.side_effect = error
            with self.assertRaises(type(error)):
                self.run_tasks("[主线]\nscan-qrcode\nturn 87 0.38\n", base, arm, vision)
            base.turn.assert_not_called()
            base.stop.assert_called_once()
            vision.close.assert_called_once()

    def test_cancel_and_device_error_preserve_error_and_try_every_cleanup(self):
        for original in (KeyboardInterrupt(), MotionCancelled("取消"), OSError("设备断开")):
            base, arm, vision = Mock(), Mock(), Mock()
            arm.calibrate.side_effect = original
            base.stop.side_effect = OSError("停车失败")
            arm.cancel.side_effect = OSError("停止失败")
            base.close.side_effect = OSError("串口关闭失败")
            arm.close.side_effect = OSError("舵机关闭失败")
            with contextlib.redirect_stderr(io.StringIO()) as output:
                with self.assertRaises(type(original)) as failed:
                    self.run_tasks("[主线]\narm-calibrate\n", base, arm, vision)
            self.assertIs(failed.exception, original)
            vision.close.assert_called_once()
            self.assertEqual(output.getvalue().count("清理失败"), 4)

    def test_dry_run_from_other_directory_without_hardware_or_ros_imports(self):
        script = '''
import importlib.abc, sys, runpy
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'serial', 'cv2', 'torch', 'ultralytics', 'rclpy', 'car_interfaces'}:
            raise ImportError('干跑不应导入设备依赖：' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[1] + '/main.py', '--dry-run']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT)],
                                cwd="/tmp", capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("干跑：按 tasks.txt 的 [主线]", result.stdout)
        steps = main.load_main(ROOT / "tasks.txt")
        for index, step in enumerate(steps, 1):
            self.assertIn(f"[主线 {index}/{len(steps)}] {step.command}", result.stdout)
        self.assertNotIn("条件分支预览", result.stdout)

    def test_actual_sigint_cancels_main_and_closes_all_devices(self):
        script = '''
import main, os, signal, threading, time
from unittest.mock import Mock, patch
from tasks.runner import Step
base, arm, vision = Mock(), Mock(), Mock()
arm.calibrate.side_effect = lambda: time.sleep(10)
base.stop.side_effect = lambda: print('base stopped')
arm.cancel.side_effect = lambda: print('arm stopped')
base.close.side_effect = lambda: print('base closed')
arm.close.side_effect = lambda: print('arm closed')
vision.close.side_effect = lambda: print('vision closed')
with patch('base.api.Base', return_value=base), patch('arm.api.Arm', return_value=arm), patch('vision.api.Vision', return_value=vision), patch('main.load_main', return_value=[Step(1, 'arm-calibrate', ())]):
    threading.Timer(0.2, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
    raise SystemExit(main.main([]))
'''
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=5)
        self.assertEqual(result.returncode, 130, result.stderr)
        for message in ("base stopped", "arm stopped", "base closed", "arm closed", "vision closed"):
            self.assertIn(message, result.stdout)


class ArmTests(unittest.TestCase):
    def test_middle_ball_simulation_preserves_zero_and_home(self):
        arm = Arm(simulate=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()), patch("tasks.ball.pause"):
                arm.calibrate()
                initial = dict(arm._arm.encoder_zero_deg)
                ball.middle(arm)
            self.assertEqual(arm._arm.encoder_zero_deg, initial)
            self.assertEqual(arm._arm.get_joints(), arm.config.joint_offsets_deg)
        finally:
            arm.close()

    def test_calibration_is_read_only_and_close_does_not_release_gripper(self):
        arm = Arm(simulate=True)
        with contextlib.redirect_stdout(io.StringIO()):
            arm.calibrate()
            self.assertTrue(all(not m.enabled for m in arm._bus.motors.values()))
            arm.close_gripper()
        servo = arm._servo
        servo.move_angle = Mock(wraps=servo.move_angle)
        servo.disable_torque = Mock(wraps=servo.disable_torque)
        arm.cancel()
        arm.close()
        servo.move_angle.assert_not_called()
        servo.disable_torque.assert_not_called()
