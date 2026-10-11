"""任务表锁靶入口：共享设备、到位返回、取消失能及干跑。"""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import main
from arm.api import Arm
from base.control import MotionCancelled
from config import ROOT
from tasks.lock_target import run_task
from tasks.runner import load_section
from tests.test_target_lock import Clock, image, scene, target_sample


SECTIONS = (("打左边的靶", "left"), ("打中间的靶", "middle"), ("打右边的靶", "right"))


class TargetLockTaskTests(unittest.TestCase):
    def test_three_sections_reuse_shared_devices_saved_aim_and_background_yolo(self):
        for section, target in SECTIONS:
            with self.subTest(section=section):
                arm = Arm(simulate=True)
                bus = arm._ensure_bus()
                vision, base = Mock(), Mock()
                camera = vision.start_camera.return_value
                web, aim = Mock(port=8080), Mock()
                with patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                        patch("tasks.lock_target.AimPoint", return_value=aim), \
                        patch("tasks.lock_target.run", return_value=None) as control, \
                        contextlib.redirect_stdout(io.StringIO()):
                    main.run(base, arm, vision, section=section)
                self.assertEqual(control.call_args.args, (camera, bus, target))
                self.assertEqual(control.call_args.kwargs["duration"], 0)
                self.assertEqual(control.call_args.kwargs["complete_after"], 2)
                self.assertEqual(control.call_args.kwargs["complete_tolerance_px"], 8)
                self.assertIs(control.call_args.kwargs["aim"], aim)
                self.assertEqual(control.call_args.kwargs["on_update"], web.update)
                self.assertEqual(control.call_args.kwargs["candidate_source"], vision.wait_target_layout)
                vision.enable_objects.assert_called_once()
                web.start.assert_called_once_with(camera)
                web.close.assert_called_once()
                vision.start.assert_not_called()
                vision.start_camera.assert_called_once()
                vision.close.assert_called_once()
                self.assertTrue(bus.closed)
                self.assertFalse(arm.calibrated)
                self.assertEqual([step.command for step in load_section(ROOT / "tasks.txt", section)],
                                 [f"lock-target-{target}", "gripper-open", "gripper-close"])

    def test_task_timeout_without_alignment_stops_instead_of_reporting_completion(self):
        for selector, direction in (("left", -1), ("middle", 1), ("right", 1)):
            with self.subTest(selector=selector):
                clock, arm = Clock(), Arm(simulate=True)
                self.addCleanup(arm.close)
                bus = arm._ensure_bus()
                frame = image(x=40)
                frame.detections.extend(image("green", 180, 110).detections
                                        + image("blue", 270, 170).detections)
                for motor in bus.motors.values():
                    motor.move = Mock(wraps=motor.move)
                index = 0
                camera = Mock()

                def next_frame(**kwargs):
                    nonlocal index
                    index += 1
                    return target_sample(frame, index, clock.wall())

                vision, web, aim = Mock(), Mock(port=8080), Mock()
                vision.wait_target_layout.side_effect = next_frame
                vision.start_camera.return_value = camera
                aim.point.return_value = (0.5, 0.5)
                with patch("tasks.lock_target.time.monotonic", clock.monotonic), \
                        patch("tasks.lock_target.time.time", clock.wall), \
                        patch("tasks.lock_target.wait_cancelable", clock.wait), \
                        patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                        patch("tasks.lock_target.AimPoint", return_value=aim):
                    with self.assertRaises(TimeoutError):
                        run_task(arm, vision, selector, duration=0.03, log=lambda _: None)
                self.assertGreater(bus.motors[1].move.call_args_list[0].args[0] * direction, 0)
                bus.motors[2].move.assert_called_once()
                self.assertEqual(bus.motors[2].angle, 12)
                self.assertTrue(bus.motors[2].enabled)
                bus.motors[3].move.assert_not_called()
                self.assertFalse(bus.closed)
                camera.close.assert_not_called()
                vision.close.assert_not_called()
                web.update.assert_called()
                web.close.assert_called_once()

    def test_aligned_task_returns_and_executes_next_instruction(self):
        clock, arm = Clock(), Arm(simulate=True)
        bus = arm._ensure_bus()
        vision, camera, web, aim = Mock(), Mock(), Mock(port=8080), Mock()
        vision.start_camera.return_value = camera
        aim.point.return_value = (0.5, 0.5)
        frame = image(x=168, y=128)  # 两轴误差恰好为 8px，同样允许完成。
        frame.detections.extend(image("green", 220, 110).detections + image("blue", 280, 110).detections)
        index = 0

        def next_frame(**kwargs):
            nonlocal index
            index += 1
            return target_sample(frame, index, clock.wall())

        vision.wait_target_layout.side_effect = next_frame
        original_disable = arm.disable

        def disable_after_restore():
            for addr in (1, 4):
                self.assertAlmostEqual(bus.motors[addr].speed_ki, 0.005)
            return original_disable()

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[打靶测试]\nlock-target-left\narm-disable\n", encoding="utf-8")
            with patch.dict("tasks.lock_target.TARGET_LOCK", {"hz": 2}), \
                    patch("tasks.lock_target.time.monotonic", clock.monotonic), \
                    patch("tasks.lock_target.time.time", clock.wall), \
                    patch("tasks.lock_target.wait_cancelable", clock.wait), \
                    patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                    patch("tasks.lock_target.AimPoint", return_value=aim), \
                    patch("tasks.runner.pause"), patch.object(arm, "disable", side_effect=disable_after_restore) as disable, \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                main.run(Mock(), arm, vision, section="打靶测试", tasks_path=path)
        self.assertEqual(clock.now, 3)
        self.assertEqual(index, 7)
        disable.assert_called_once()
        self.assertIn("锁靶完成", output.getvalue())
        self.assertIn("[打靶测试 2/2] arm-disable", output.getvalue())
        self.assertTrue(all(not motor.enabled for motor in bus.motors.values()))
        self.assertTrue(bus.closed)
        web.close.assert_called_once()

    def test_cancellation_disables_all_motors_and_propagates_to_task_runner(self):
        for error in (KeyboardInterrupt(), MotionCancelled("取消")):
            with self.subTest(error=error):
                arm, vision, web = Arm(simulate=True), Mock(), Mock(port=8080)
                self.addCleanup(arm.close)
                bus = arm._ensure_bus()
                for motor in bus.motors.values():
                    motor.move(10)
                with patch("tasks.lock_target.AimPoint"), \
                        patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                        patch("tasks.lock_target.run", side_effect=error), \
                        self.assertRaises(type(error)) as failed:
                    run_task(arm, vision, log=lambda _: None)
                self.assertIs(failed.exception, error)
                self.assertTrue(all(not motor.enabled for motor in bus.motors.values()))
                self.assertFalse(bus.closed)
                web.close.assert_called_once()

    def test_lost_task_reuses_calibration_restores_pose_and_does_not_execute_next_step(self):
        clock, arm = Clock(), Arm(simulate=True)
        with contextlib.redirect_stdout(io.StringIO()):
            arm.calibrate()
            arm.move_joints(-90, 0, 160, 4)
        controller, bus = arm._arm, arm._ensure_bus()
        zero = dict(controller.encoder_zero_deg)
        vision, camera, web, aim = Mock(), Mock(), Mock(port=8080), Mock()
        vision.start_camera.return_value = camera
        aim.point.return_value = (0.5, 0.5)
        index = 0

        def next_frame(**kwargs):
            nonlocal index
            index += 1
            return target_sample(scene() if index <= 3 else image(None), index, clock.wall())

        vision.wait_target_layout.side_effect = next_frame
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[丢靶测试]\nlock-target-middle 0.08\ngripper-open\n", encoding="utf-8")
            with patch("tasks.lock_target.time.monotonic", clock.monotonic), \
                    patch("tasks.lock_target.time.time", clock.wall), \
                    patch("tasks.lock_target.wait_cancelable", clock.wait), \
                    patch("tasks.lock_target.TargetLockWeb", return_value=web), \
                    patch("tasks.lock_target.AimPoint", return_value=aim), \
                    patch.object(arm, "calibrate") as calibrate, \
                    patch.object(arm, "open_gripper") as gripper, \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(TimeoutError, "锁靶超时"):
                    main.run(Mock(), arm, vision, section="丢靶测试", tasks_path=path)
        calibrate.assert_not_called()
        gripper.assert_not_called()
        self.assertEqual(controller.encoder_zero_deg, zero)
        self.assertEqual({addr: motor.angle for addr, motor in bus.motors.items()},
                         {1: -90, 2: 0, 3: 0, 4: -20})
        self.assertTrue(bus.closed)
        web.close.assert_called_once()

    def test_precancel_and_web_failure_do_not_open_devices(self):
        arm, vision = Arm(simulate=True), Mock()
        self.addCleanup(arm.close)
        stop = threading.Event()
        stop.set()
        with patch("tasks.lock_target.TargetLockWeb") as web:
            with self.assertRaises(MotionCancelled):
                run_task(arm, vision, stop_event=stop)
            web.assert_not_called()
        with patch("tasks.lock_target.AimPoint"), \
                patch("tasks.lock_target.TargetLockWeb", side_effect=OSError("端口已占用")):
            with self.assertRaisesRegex(OSError, "端口已占用"):
                run_task(arm, vision)
        self.assertIsNone(arm._bus)
        vision.start_camera.assert_not_called()

    def test_all_three_cli_sections_dry_run_without_hardware_imports(self):
        script = '''
import importlib.abc, runpy, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'serial', 'cv2', 'torch', 'ultralytics'}:
            raise ImportError('干跑不应导入设备依赖：' + fullname)
sys.meta_path.insert(0, Block())
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[1] + '/main.py', '--' + sys.argv[2], '--dry-run']
runpy.run_path(sys.argv[0], run_name='__main__')
'''
        with tempfile.TemporaryDirectory() as folder:
            for section, target in SECTIONS:
                with self.subTest(section=section):
                    result = subprocess.run([sys.executable, "-B", "-c", script, str(ROOT), section],
                                            cwd=folder, capture_output=True, text=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"lock_target.run(target='{target}')", result.stdout)
                    self.assertIn("横纵误差均 ≤ 8px 连续保持 2 秒后完成", result.stdout)
                    self.assertNotIn("[主线 ", result.stdout)
