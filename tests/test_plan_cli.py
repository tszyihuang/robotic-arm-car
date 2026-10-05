"""分段选择与 ROS 提交内容的回归，不连接设备。"""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from car_nodes.planner.client import main
from car_nodes.planner.tasks import prepare_tasks, select_task_section

ROOT = Path(__file__).resolve().parents[1]
TABLE = """# 任务表
[主线]
straight 0.3
scan-qrcode
[抓左边的小球] # 标题后的注释
gripper-open
gripper-close
[抓右边的小球]
# 还没有填写
[其他任务]
unknown-command
"""


class PlanCliTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.table = Path(self.directory.name) / 'tests.txt'
        self.table.write_text(TABLE, encoding='utf-8')

    def cli(self, *args):
        return subprocess.run(
            [sys.executable, '-B', str(ROOT / 'plan.py'), '-f', str(self.table), *args],
            cwd=self.directory.name, capture_output=True, text=True, timeout=10)

    def test_default_and_explicit_main_only_validate_main(self):
        default = self.cli('--dry-run')
        explicit = self.cli('--主线', '--dry-run')
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(explicit.returncode, 0, explicit.stderr)
        self.assertEqual(default.stdout, explicit.stdout)
        self.assertIn('[1/2] base: straight 0.3', default.stdout)
        self.assertIn('[2/2] vision: scan-qrcode', default.stdout)
        self.assertNotIn('gripper', default.stdout)

    def test_named_section_excludes_main_and_other_sections(self):
        result = self.cli('--抓左边的小球', '--list')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('[1/2] arm: gripper-open', result.stdout)
        self.assertIn('[2/2] arm: gripper-close', result.stdout)
        self.assertNotIn('straight', result.stdout)
        self.assertNotIn('scan-qrcode', result.stdout)

    def test_new_heading_automatically_becomes_an_option(self):
        with self.table.open('a', encoding='utf-8') as table:
            table.write('[新增任务]\nturn 90 0.3\n')
        result = self.cli('--新增任务', '--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('[1/1] base: turn 90 0.3', result.stdout)

    def test_unknown_abbreviated_and_multiple_sections_are_rejected(self):
        for args in (('--不存在',), ('--抓左边',), ('--主线', '--抓左边的小球')):
            with self.subTest(args=args):
                result = self.cli(*args)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn('ROS 环境未加载', result.stderr)

    def test_empty_section_reports_its_name_before_connecting_ros(self):
        result = self.cli('--抓右边的小球')
        self.assertEqual(result.returncode, 2)
        self.assertIn('任务分段 [抓右边的小球] 为空', result.stderr)
        self.assertNotIn('ROS 环境未加载', result.stderr)

    def test_missing_main_does_not_run_another_section(self):
        self.table.write_text('[抓左边的小球]\ngripper-open\n', encoding='utf-8')
        result = self.cli('--dry-run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('找不到任务分段 [主线]', result.stderr)
        self.assertEqual(self.cli('--抓左边的小球', '--dry-run').returncode, 0)

    def test_invalid_selected_command_keeps_original_line_number(self):
        result = self.cli('--其他任务', '--dry-run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('清单第 11 行', result.stderr)
        self.assertIn('unknown-command', result.stderr)

    def test_duplicate_headings_are_rejected(self):
        self.table.write_text(TABLE + '[主线]\nturn 90\n', encoding='utf-8')
        result = self.cli('--dry-run')
        self.assertEqual(result.returncode, 2)
        self.assertIn('重复的任务分段 [主线]', result.stderr)

    def test_plain_files_and_command_text_remain_supported(self):
        self.table.write_text('gripper-open\ngripper-close\n', encoding='utf-8')
        result = self.cli('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('共 2 步', result.stdout)
        result = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'plan.py'), '--command', 'straight 0.4', '--dry-run'],
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('共 1 步', result.stdout)
        self.assertEqual(self.cli('--主线', '--dry-run').returncode, 2)

    def test_help_lists_current_sections(self):
        result = self.cli('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--主线', result.stdout)
        self.assertIn('--抓左边的小球', result.stdout)
        self.assertIn('--其他任务', result.stdout)

    def test_selection_does_not_import_calibration_from_main(self):
        selected = select_task_section(
            '[主线]\narm-calibrate\n[抓左边的小球]\narm-home', '抓左边的小球')
        with self.assertRaisesRegex(ValueError, '同一份清单'):
            prepare_tasks(selected)

    def test_ros_receives_only_selected_tasks_and_preserves_ros_arguments(self):
        result = NS(success=True, message='任务完成', details_json='{}')
        child = Mock(accepted=True)
        child.get_result_async.return_value = NS(done=lambda: True, result=lambda: NS(result=result))
        client = Mock()
        client.send_goal_async.return_value = NS(done=lambda: True, result=lambda: child)
        ros = NS(init=Mock(), ok=lambda: True, shutdown=Mock())
        node = Mock()
        modules = {
            'rclpy': ros,
            'rclpy.action': NS(ActionClient=Mock(return_value=client)),
            'rclpy.node': NS(Node=Mock(return_value=node)),
            'rclpy.qos': NS(QoSProfile=Mock(), DurabilityPolicy=NS(TRANSIENT_LOCAL=1)),
            'rclpy.signals': NS(SignalHandlerOptions=NS(NO=0)),
            'std_msgs.msg': NS(Bool=Mock()),
            'std_srvs.srv': NS(Trigger=Mock()),
            'car_interfaces.action': NS(RunTasks=NS(Goal=lambda **kwargs: NS(**kwargs))),
        }
        ros_args = ['--ros-args', '-r', '__ns:=/section_test']
        with patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()):
            code = main(['-f', str(self.table), '--抓左边的小球', '--ros-dry-run', *ros_args])
        self.assertEqual(code, 0)
        sent = client.send_goal_async.call_args.args[0]
        self.assertEqual([step.text for step in prepare_tasks(sent.tasks)],
                         ['gripper-open', 'gripper-close'])
        self.assertTrue(sent.dry_run)
        ros.init.assert_called_once_with(args=ros_args, signal_handler_options=0)
        node.destroy_node.assert_called_once()
        ros.shutdown.assert_called_once()
