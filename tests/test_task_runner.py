"""任务表段落边界、注释及执行前语法校验。"""
import contextlib
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch, call

import main
from base.control import MotionCancelled
from config import BASE
from debug.dry_run import DryBase, DryArm, DryVision
from tasks.runner import load_main, load_section, list_sections


class TaskParserTests(unittest.TestCase):
    def test_section_discovery_includes_empty_sections_and_ignores_comments(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'tasks.txt'
            path.write_text('\ufeff[主线] # 默认\n未知指令\n'
                            '# [不要执行]\n[抓左边的小球]\n'
                            '[新任务] # 空段也可选\n[主线]\n', encoding='utf-8')
            self.assertEqual(list_sections(path), ['主线', '抓左边的小球', '新任务'])

    def parse(self, text, section="主线"):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text(text, encoding="utf-8")
            return load_main(path) if section == "主线" else load_section(path, section)

    def test_selected_section_preserves_lines_and_ignores_other_sections(self):
        steps = self.parse("[主线]\n未知指令\n[抓中间的小球] # 独立执行\n"
                           "arm-calibrate\ngripper-open\narm-move -95 133 134 -83\n"
                           "[抓左边的小球]\n未知指令\n", "抓中间的小球")
        self.assertEqual([(step.command, step.args) for step in steps], [
            ("arm-calibrate", ()), ("gripper-open", ()),
            ("arm-move", (-95, 133, 134, -83))])
        self.assertEqual([step.line for step in steps], [4, 5, 6])

    def test_missing_empty_and_invalid_selected_section_report_errors(self):
        for text, message in (("[主线]\narm-calibrate\n", "缺少.*抓中间的小球"),
                              ("[抓中间的小球]\n# 空\n[其他]\n", "没有可执行指令"),
                              ("[抓中间的小球]\ngripper-open\narm-move 1 2\n",
                               "tasks.txt:3")):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                self.parse(text, "抓中间的小球")

    def test_bom_comments_and_other_sections_are_ignored(self):
        steps = self.parse("\ufeff[其他]\n未知指令\n[主线] # 按顺序\n"
                           "# 注释\n\nstraight -0.48 150 # 倒车\nturn 44\n"
                           "[抓中间的小球]\n未知指令\narm-calibrate\n")
        self.assertEqual([(s.command, s.args) for s in steps],
                         [("straight", (-0.48, 150)), ("turn", (44,))])
        self.assertEqual([s.line for s in steps], [6, 7])

    def test_missing_or_empty_main_section_is_an_error(self):
        for text in ("straight 0.48", "[其他]\nstraight 0.48\n",
                     "[主线]\n# 空\n[其他]\nstraight 0.48"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_invalid_command_arity_and_nonfinite_numbers_report_line(self):
        for command in ("未知 1", "straight", "align 1", "turn 44 0.24 150",
                        "sleep", "sleep 1 2", "sleep -0.5", "sleep nan", "sleep inf", "sleep 错误",
                        "pause 1", "light 353",
                        "vision-straight", "vision-straight 1 600 700",
                        "vision-straight 1 0", "vision-straight 1 -1", "vision-straight 1 1001",
                        "vision-straight 1 nan", "vision-straight 1 inf", "vision-straight 1 错误",
                        "arm-move 0 0 160", "arm-disable 1", "calibrate-ball-position 1", "detect-targets 1",
                        "lock-target-left -1", "lock-target-middle 1 2", "lock-target-right nan",
                        "小球任务 1", "打靶任务 2", "人质任务 3", "go-to-finish 1.69",
                        "straight nan", "turn inf", "turn 错误"):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "tasks.txt:3"):
                self.parse(f"[主线]\nstraight 0.48\n{command}\n")

    def test_sleep_accepts_seconds_zero_and_comments(self):
        steps = self.parse("[主线]\nsleep 0.5 # 等待半秒\nsleep 2\nsleep 0\n")
        self.assertEqual([(step.command, step.args) for step in steps],
                         [("sleep", (0.5,)), ("sleep", (2.0,)), ("sleep", (0.0,))])

    def test_pause_accepts_no_arguments_and_comments(self):
        steps = self.parse("[主线]\npause # 按回车继续\nstraight 0.3\n")
        self.assertEqual([(step.command, step.args) for step in steps],
                         [("pause", ()), ("straight", (0.3,))])

    def test_vision_straight_accepts_optional_speed_in_mm_per_second(self):
        steps = self.parse("[主线]\nvision-straight 1 600 # 600 mm/s\n"
                           "vision-straight 0.4\nvision-straight 1 1000\n")
        self.assertEqual([(step.command, step.args) for step in steps], [
            ("vision-straight", (1.0, 600.0)), ("vision-straight", (0.4,)),
            ("vision-straight", (1.0, 1000.0))])

    def test_mission_directives_are_parsed_without_arguments(self):
        steps = self.parse("[主线]\n小球任务 # 按颜色选择\n打靶任务\n人质任务\ngo-to-finish\n")
        self.assertEqual([(step.command, step.args) for step in steps],
                         [("小球任务", ()), ("打靶任务", ()), ("人质任务", ()), ("go-to-finish", ())])


class VisionStraightExecutionTests(unittest.TestCase):
    def run_tasks(self, base, arm, vision):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[视觉测试]\nvision-straight 1 600\nvision-straight 0.4\n",
                            encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main.run(base, arm, vision, tasks_path=path, section="视觉测试")
            return output.getvalue()

    def test_task_execution_passes_speed_and_shared_vision(self):
        base, vision = Mock(dry_run=True), Mock()
        self.run_tasks(base, Mock(), vision)
        self.assertEqual(base.vision_straight.call_args_list,
                         [call(1.0, vision, 600.0), call(0.4, vision)])

    def test_dry_run_prints_custom_speed_then_default_speed(self):
        output = self.run_tasks(DryBase(), DryArm(), DryVision())
        self.assertIn("base.vision_straight(1 m, 600 mm/s)", output)
        self.assertIn(f"base.vision_straight(0.4 m, {BASE['speed']:g} mm/s)", output)


class PauseExecutionTests(unittest.TestCase):
    def run_tasks(self, base, arm, vision, *, stop_event=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[暂停测试]\npause # 按回车继续\nstraight 0.3\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main.run(base, arm, vision, tasks_path=path,
                         section="暂停测试", stop_event=stop_event)

    def test_enter_resumes_before_next_command_without_extra_step_gap(self):
        base = Mock(dry_run=False, stop_event=None)
        arm, vision = Mock(), Mock()
        trace = []

        def press_enter(fd):
            base.straight.assert_not_called()
            base.close.assert_not_called()
            trace.append("enter")
            return True

        base.straight.side_effect = lambda distance: trace.append(("straight", distance))
        with patch("tasks.wait_continue._enter_pressed", side_effect=press_enter) as console, \
                patch("tasks.runner.pause") as pause:
            self.run_tasks(base, arm, vision)
        self.assertEqual(trace, ["enter", ("straight", 0.3)])
        console.assert_called_once()
        pause.assert_called_once_with(base)
        vision.start.assert_not_called()

    def test_closed_input_or_ctrl_c_prevents_next_command_and_closes_devices(self):
        for error in (EOFError(), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                base = Mock(dry_run=False, stop_event=None)
                arm, vision = Mock(), Mock()
                with patch("tasks.wait_continue._enter_pressed", side_effect=error), self.assertRaises(type(error)):
                    self.run_tasks(base, arm, vision)
                base.straight.assert_not_called()
                base.stop.assert_called_once_with()
                base.close.assert_called_once_with(release_motors=False)
                arm.close.assert_called_once_with()
                vision.close.assert_called_once_with()

    def test_cancellation_before_enter_returns_prevents_next_command(self):
        base = Mock(dry_run=False)
        stop = threading.Event()
        with patch("tasks.wait_continue._enter_pressed", side_effect=lambda fd: stop.set()), \
                self.assertRaises(MotionCancelled):
            self.run_tasks(base, Mock(), Mock(), stop_event=stop)
        base.straight.assert_not_called()

    def test_dry_run_does_not_wait_for_input(self):
        base = Mock(dry_run=True)
        arm = Mock()
        with patch("tasks.wait_continue.run") as console, patch("tasks.runner.pause"):
            self.run_tasks(base, arm, Mock())
        console.assert_not_called()
        arm.read_servo_angle.assert_not_called()
        base.straight.assert_called_once_with(0.3)

    def test_servo_movement_resumes_before_next_command(self):
        base, arm = Mock(dry_run=False, stop_event=None), Mock()
        arm.read_servo_angle.side_effect = [100.0, 105.0]
        with patch("tasks.wait_continue._enter_pressed", return_value=False), \
                patch("tasks.wait_continue.time.monotonic", side_effect=[10.0, 10.01, 10.4, 10.41]), \
                patch("tasks.runner.pause") as pause:
            self.run_tasks(base, arm, Mock())
        base.straight.assert_called_once_with(0.3)
        self.assertEqual(arm.read_servo_angle.call_args_list,
                         [call(2, stop_event=None), call(2, stop_event=None)])
        pause.assert_called_once_with(base)


class SleepExecutionTests(unittest.TestCase):
    def run_tasks(self, base, *, stop_event=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text("[延时测试]\nsleep 0.5\nstraight 0.3\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                main.run(base, Mock(), Mock(), tasks_path=path,
                         section="延时测试", stop_event=stop_event)

    def test_wait_finishes_before_next_command_without_extra_step_gap(self):
        trace = []
        base = Mock(dry_run=False, stop_event=None)
        base.straight.side_effect = lambda distance: trace.append(("straight", distance))
        with patch("base.control.time.sleep", side_effect=lambda seconds: trace.append(("sleep", seconds))), \
                patch("tasks.runner.pause") as pause:
            self.run_tasks(base)
        self.assertEqual(trace, [("sleep", 0.5), ("straight", 0.3)])
        pause.assert_called_once_with(base)

    def test_cancellation_during_sleep_prevents_next_command(self):
        base = Mock(dry_run=False)
        stop = threading.Event()

        def cancel_wait(seconds):
            self.assertEqual(seconds, 0.5)
            stop.set()
            return True

        with patch.object(stop, "wait", side_effect=cancel_wait), self.assertRaises(MotionCancelled):
            self.run_tasks(base, stop_event=stop)
        base.straight.assert_not_called()
        base.stop.assert_called_once()
        base.close.assert_called_once_with(release_motors=False)

    def test_dry_run_does_not_wait(self):
        base = Mock(dry_run=True)
        with patch("tasks.runner.wait_cancelable") as wait, patch("tasks.runner.pause"):
            self.run_tasks(base)
        wait.assert_not_called()
        base.straight.assert_called_once_with(0.3)
