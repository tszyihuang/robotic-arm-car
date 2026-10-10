"""三位任务码、现场排列及任务表动作段的端到端选择；不连接实车。"""
import contextlib
import io
from itertools import permutations, product
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, patch

import main
from arm.api import Arm
from base.control import MotionCancelled
from tasks.lock_target import TargetLock


SECTIONS = ("抓左边的小球", "抓中间的小球", "抓右边的小球",
            "打左边的靶", "打中间的靶", "打右边的靶",
            "抓左边的物体", "抓中间的物体", "抓右边的物体")
ANGLES = (11, 12, 13, 21, 22, 23, 31, 32, 33)
POSITIONS = ("left", "middle", "right")
COLORS = ("red", "green", "blue")
SHAPES = ("cylinder", "cone", "drum")


def task_table(commands):
    text = f"[主线]\n{commands}\n"
    for section, angle in zip(SECTIONS, ANGLES):
        text += f"[{section}]\narm-calibrate\narm-move {angle} 0 160 24\narm-home\n"
    return text


def layout(colors):
    # 返回顺序故意与左右顺序不同，选择应使用已收集的 position。
    return list(reversed([{"position": position, "color": color,
                           "box": [i * 100, 10, i * 100 + 40, 50],
                           "center_x": i * 100 + 20, "center_y": 30}
                          for i, (position, color) in enumerate(zip(POSITIONS, colors))]))


class MissionDispatchTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "tasks.txt"
        self.positions_path = Path(folder.name) / "target_positions.json"

    def devices(self):
        base = Mock(dry_run=True, stop_event=threading.Event())
        arm = Mock(dry_run=True, calibrated=False)
        vision = Mock(dry_run=False, stop_event=threading.Event())
        vision.scan_qrcode.return_value = "211"
        vision.observe_balls.return_value = layout(COLORS)
        vision.observe_targets.return_value = layout(COLORS)
        vision.observe_target.return_value = "middle"
        return base, arm, vision

    def run_plan(self, text, base, arm, vision, *, stop_event=None):
        self.path.write_text(text, encoding="utf-8")
        with patch("tasks.runner.pause"), \
                patch("tasks.detect_target.DEFAULT_POSITIONS_FILE", self.positions_path), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            main.run(base, arm, vision, tasks_path=self.path, stop_event=stop_event)
        return output.getvalue()

    def assert_cleaned(self, base, arm, vision):
        base.stop.assert_called_once_with()
        base.close.assert_called_once_with(release_motors=False)
        arm.close.assert_called_once_with()
        vision.close.assert_called_once_with()

    def test_all_codes_and_arrangements_execute_one_branch_per_task_then_resume(self):
        table = task_table("scan-qrcode\ndetect-balls\n小球任务\nstraight 0.25\n"
                           "detect-targets\n打靶任务\nstraight 0.5\n人质任务\nstraight 0.75")
        color_orders, shape_orders = list(permutations(COLORS)), list(permutations(SHAPES))
        for digits in product("123", repeat=3):
            for i, balls in enumerate(color_orders):
                targets, objects = color_orders[5 - i], shape_orders[(i + 2) % 6]
                qr = "".join(digits)
                with self.subTest(qr=qr, balls=balls, targets=targets, objects=objects):
                    base, arm, vision = self.devices()
                    vision.scan_qrcode.return_value = qr
                    vision.observe_balls.return_value = layout(balls)
                    vision.observe_targets.return_value = layout(targets)
                    shape = SHAPES[int(digits[2]) - 1]
                    object_index = objects.index(shape)
                    vision.observe_target.return_value = POSITIONS[object_index]
                    trace = Mock()
                    for device, method, name in ((vision, "scan_qrcode", "scan"),
                                                  (vision, "observe_balls", "balls"),
                                                  (vision, "observe_targets", "targets"),
                                                  (vision, "observe_target", "object"),
                                                  (arm, "move_joints", "move"),
                                                  (base, "straight", "drive")):
                        trace.attach_mock(getattr(device, method), name)
                    self.run_plan(table, base, arm, vision)
                    ball_angle = 11 + balls.index(COLORS[int(digits[0]) - 1])
                    target_angle = 21 + targets.index(COLORS[int(digits[1]) - 1])
                    self.assertEqual(trace.mock_calls, [
                        call.scan(), call.balls(), call.move(ball_angle, 0, 160, 24),
                        call.drive(0.25), call.targets(), call.move(target_angle, 0, 160, 24),
                        call.drive(0.5), call.object("object", shape),
                        call.move(31 + object_index, 0, 160, 24), call.drive(0.75)])
                    vision.start.assert_called_once_with()
                    self.assert_cleaned(base, arm, vision)

    def test_invalid_qr_payload_stops_before_next_route_action(self):
        for qr in (None, 211, "012", "234", "11", "1111", "２１１", "123+321"):
            with self.subTest(qr=qr):
                base, arm, vision = self.devices()
                vision.scan_qrcode.return_value = qr
                with self.assertRaisesRegex(ValueError, "三位"):
                    self.run_plan("[主线]\nscan-qrcode\nstraight 1\n", base, arm, vision)
                base.straight.assert_not_called()
                self.assert_cleaned(base, arm, vision)

    def test_missing_scan_or_detection_stops_before_dispatch(self):
        for commands, message in (("小球任务", "scan-qrcode"),
                                  ("打靶任务", "scan-qrcode"),
                                  ("人质任务", "scan-qrcode"),
                                  ("scan-qrcode\n小球任务", "detect-balls"),
                                  ("scan-qrcode\n打靶任务", "detect-targets")):
            with self.subTest(commands=commands):
                base, arm, vision = self.devices()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_plan(task_table(commands + "\nstraight 1"), base, arm, vision)
                arm.move_joints.assert_not_called()
                base.straight.assert_not_called()
                self.assert_cleaned(base, arm, vision)

    def test_incomplete_or_ambiguous_color_layout_never_dispatches(self):
        incomplete = layout(COLORS)[:2]
        duplicates = layout(("red", "red", "blue"))
        repeated_position = layout(COLORS)
        repeated_position[0]["position"] = repeated_position[1]["position"]
        for command, method, task in (("detect-balls", "observe_balls", "小球任务"),
                                      ("detect-targets", "observe_targets", "打靶任务")):
            for rows in (incomplete, duplicates, repeated_position, None):
                with self.subTest(task=task, rows=rows):
                    base, arm, vision = self.devices()
                    getattr(vision, method).return_value = rows
                    # 隔离打印和文件记录，让主线校验直接接收异常排列。
                    wrapper = "detect_ball" if task == "小球任务" else "detect_target"
                    with patch(f"tasks.runner.{wrapper}.run", return_value=rows), \
                            self.assertRaisesRegex(ValueError, "现场排列"):
                        self.run_plan(task_table(f"scan-qrcode\n{command}\n{task}\nstraight 1"),
                                      base, arm, vision)
                    arm.move_joints.assert_not_called()
                    base.straight.assert_not_called()
                    self.assert_cleaned(base, arm, vision)

    def test_rescanning_clears_previously_collected_layouts(self):
        base, arm, vision = self.devices()
        vision.scan_qrcode.side_effect = ["211", "311"]
        with self.assertRaisesRegex(ValueError, "detect-balls"):
            self.run_plan(task_table("scan-qrcode\ndetect-balls\nscan-qrcode\n小球任务\nstraight 1"),
                          base, arm, vision)
        arm.move_joints.assert_not_called()
        base.straight.assert_not_called()
        self.assert_cleaned(base, arm, vision)

    def test_branch_errors_and_cycles_are_rejected_before_starting_devices(self):
        valid = task_table("straight 1\nscan-qrcode\ndetect-balls\n小球任务")
        missing = valid.replace("[抓右边的小球]", "[无关段落]")
        invalid = valid.replace("arm-move 13 0 160 24", "arm-move 1 2")
        cycle = valid.replace("arm-move 13 0 160 24", "小球任务")
        for text, message in ((missing, "缺少.*抓右边的小球"),
                              (invalid, "tasks.txt:[0-9]+"), (cycle, "调用循环")):
            with self.subTest(message=message):
                base, arm, vision = self.devices()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_plan(text, base, arm, vision)
                vision.start.assert_not_called()
                vision.scan_qrcode.assert_not_called()
                base.straight.assert_not_called()
                arm.move_joints.assert_not_called()
                self.assert_cleaned(base, arm, vision)

    def test_failed_rescue_observation_stops_before_grab_or_route(self):
        for position, error in ((None, None), ("unknown", None), (None, TimeoutError("观察超时"))):
            with self.subTest(position=position, error=error):
                base, arm, vision = self.devices()
                vision.observe_target.return_value = position
                vision.observe_target.side_effect = error
                with self.assertRaises(TimeoutError if error else ValueError):
                    self.run_plan(task_table("scan-qrcode\n人质任务\nstraight 1"), base, arm, vision)
                arm.move_joints.assert_not_called()
                base.straight.assert_not_called()
                self.assert_cleaned(base, arm, vision)

    def test_cancel_after_rescue_observation_prevents_branch_motion(self):
        base, arm, vision = self.devices()
        stop = threading.Event()

        def cancel(*args):
            stop.set()
            return "middle"

        vision.observe_target.side_effect = cancel
        with self.assertRaises(MotionCancelled):
            self.run_plan(task_table("scan-qrcode\n人质任务\nstraight 1"),
                          base, arm, vision, stop_event=stop)
        arm.move_joints.assert_not_called()
        base.straight.assert_not_called()
        self.assert_cleaned(base, arm, vision)

    def test_action_failure_stops_parent_route_and_cleans_up_shared_devices(self):
        base, arm, vision = self.devices()
        original = OSError("机械臂断开")
        arm.move_joints.side_effect = original
        with self.assertRaises(OSError) as failed:
            self.run_plan(task_table("scan-qrcode\ndetect-balls\n小球任务\nstraight 1"), base, arm, vision)
        self.assertIs(failed.exception, original)
        base.straight.assert_not_called()
        self.assert_cleaned(base, arm, vision)

    def test_called_grab_section_preserves_calibrated_zero_at_observation_pose(self):
        base, _, vision = self.devices()
        vision.observe_balls.return_value = layout(("green", "blue", "red"))
        arm = Arm(simulate=True)
        arm.dry_run = True
        zero = []
        calibrate = arm.calibrate

        def record_zero(**kwargs):
            result = calibrate(**kwargs)
            zero.append(dict(arm._arm.encoder_zero_deg))
            return result

        text = task_table("arm-calibrate\narm-move -90 55 74 89\nscan-qrcode\n"
                          "detect-balls\n小球任务")
        with patch.object(arm, "calibrate", side_effect=record_zero) as calibration, \
                patch.object(arm, "home", wraps=arm.home) as home:
            self.run_plan(text, base, arm, vision)
        calibration.assert_called_once_with(hold_tool=True)
        self.assertEqual(zero, [{1: 0, 2: 0, 3: 0, 4: 0}])
        home.assert_called_once_with()

    def test_actual_main_211_uses_task_table_branches_with_simulated_arm(self):
        base, _, vision = self.devices()
        vision.observe_balls.return_value = layout(("blue", "green", "red"))
        vision.observe_targets.return_value = layout(("red", "blue", "green"))
        vision.observe_target.return_value = "right"
        arm = Arm(simulate=True)
        arm.dry_run = True
        text = main.TASKS_FILE.read_text(encoding="utf-8")
        with patch("tasks.lock_target.run_task", return_value={"ok": True}) as lock, \
                patch.object(arm, "calibrate", wraps=arm.calibrate) as calibration, \
                patch.object(arm, "move_joints", wraps=arm.move_joints) as moves:
            output = self.run_plan(text, base, arm, vision)
        for selected in ("抓中间的小球", "打左边的靶", "抓右边的物体"):
            self.assertIn(f"[{selected} 1/", output)
        for unselected in ("抓左边的小球", "抓右边的小球", "打中间的靶",
                           "打右边的靶", "抓左边的物体", "抓中间的物体"):
            self.assertNotIn(f"[{unselected} 1/", output)
        self.assertEqual(calibration.call_count, 3)
        self.assertIn(call(-98, 119, 105.4, 2.3), moves.call_args_list)
        self.assertIn(call(-63, 109, 134, -83), moves.call_args_list)
        self.assertEqual(lock.call_args.args, (arm, vision, "red"))
        vision.observe_target.assert_called_once_with("object", "cylinder")
        self.assertIn("[主线 38/38] vision-straight 1.55", output)

    def test_target_branch_locks_qr_color_when_another_target_is_closer_to_frame_center(self):
        base, arm, vision = self.devices()
        vision.observe_targets.return_value = layout(("green", "red", "blue"))
        text = task_table("scan-qrcode\ndetect-targets\n打靶任务")
        text = text.replace("arm-move 22 0 160 24", "lock-target-middle")
        observed = []

        def lock_on_new_frame(arm, vision, selector, **kwargs):
            rows = [{"value": color, "box": [x - 20, 280, x + 20, 320]}
                    for color, x in (("green", 490), ("red", 640), ("blue", 860))]
            result = TargetLock(selector).step(rows, (1000, 600), 0.05)
            observed.append(result["target"]["value"])
            return result

        with patch("tasks.lock_target.run_task", side_effect=lock_on_new_frame) as lock:
            output = self.run_plan(text, base, arm, vision)
        self.assertIn("[打中间的靶 1/", output)
        self.assertEqual(observed, ["red"])
        self.assertEqual(lock.call_args.args, (arm, vision, "red"))
        self.assert_cleaned(base, arm, vision)
