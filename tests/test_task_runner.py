"""任务表段落边界、注释及执行前语法校验。"""
from pathlib import Path
import tempfile
import unittest

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
                        "arm-move 0 0 160", "arm-disable 1", "calibrate-ball-position 1",
                        "straight nan", "turn inf", "turn 错误"):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "tasks.txt:3"):
                self.parse(f"[主线]\nstraight 0.48\n{command}\n")
