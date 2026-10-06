"""直接调用的主线、迁移顺序、分支失败和清理验证；不连接实车。"""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

import main
from tasks import ball, target, delivery
from arm.api import Arm
from control import MotionCancelled

ROOT = Path(__file__).resolve().parents[1]


class Recorder:
    dry_run = True  # 测试中只去掉步骤间等待，视觉结果由测试明确给出。

    def __init__(self, trace, position="middle", qr="211"):
        self.trace, self.position, self.qr = trace, position, qr
        self.cleaned = []

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
    def test_main_25_steps_and_middle_7_steps_match_baseline(self):
        # 固定迁移时的动作基准，仅用于回归测试。
        fixture = Path(__file__).with_name("fixtures") / "mission_actions.json"
        data = json.loads(fixture.read_text())
        actions = {name: [(command, tuple(args)) for command, args in rows]
                   for name, rows in data.items()}
        self.assertEqual(len(actions["main"]), 25)
        self.assertEqual(len(actions["middle_ball"]), 7)
        expected = []
        for action in actions["main"]:
            expected.append(action)
            if action[0] == "抓球任务":
                expected.extend(actions["middle_ball"])
        trace = []
        devices = [Recorder(trace) for _ in range(3)]
        main.run(*devices)
        self.assertEqual(trace, expected)
        self.assertEqual([d.cleaned for d in devices], [["stop", "close"], ["cancel", "close"], ["close"]])

    def test_missing_ball_branch_stops_before_remaining_route(self):
        for position in ("left", "right"):
            trace = []
            devices = [Recorder(trace, position) for _ in range(3)]
            with self.subTest(position=position), self.assertRaises(NotImplementedError):
                main.run(*devices)
            self.assertNotIn(("vision-straight", (1.76,)), trace)
            self.assertTrue(all(d.cleaned for d in devices))

    def test_all_eight_missing_actions_fail_explicitly(self):
        missing = [ball.left, ball.right, target.left, target.middle, target.right,
                   delivery.left, delivery.middle, delivery.right]
        for action in missing:
            with self.subTest(action=action), self.assertRaisesRegex(NotImplementedError, "未填写"):
                action(Mock())

    def test_qr_failure_cannot_reach_ball_route(self):
        for error in (TimeoutError("扫码超时"), ValueError("二维码无效")):
            base, arm, vision = Mock(), Mock(), Mock()
            with patch("tasks.pause"), patch("tasks.scan.pause"), patch("tasks.route.pause"):
                vision.scan_qrcode.side_effect = error
                with self.assertRaises(type(error)):
                    main.run(base, arm, vision)
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
                    main.run(base, arm, vision)
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
        self.assertIn("base.vision_straight(1.55 m", result.stdout)
        self.assertIn("干跑不选择实际位置", result.stdout)

    def test_actual_sigint_cancels_main_and_closes_all_devices(self):
        script = '''
import main, os, signal, threading, time
from unittest.mock import Mock, patch
base, arm, vision = Mock(), Mock(), Mock()
arm.calibrate.side_effect = lambda: time.sleep(10)
base.stop.side_effect = lambda: print('base stopped')
arm.cancel.side_effect = lambda: print('arm stopped')
base.close.side_effect = lambda: print('base closed')
arm.close.side_effect = lambda: print('arm closed')
vision.close.side_effect = lambda: print('vision closed')
with patch('base.api.Base', return_value=base), patch('arm.api.Arm', return_value=arm), patch('vision.api.Vision', return_value=vision):
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
