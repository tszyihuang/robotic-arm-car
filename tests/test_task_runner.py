"""任务表段落边界、注释及执行前语法校验。"""
from pathlib import Path
import tempfile
import unittest

from tasks.runner import load_main


class TaskParserTests(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tasks.txt"
            path.write_text(text, encoding="utf-8")
            return load_main(path)

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
                        "arm-move 0 0 160", "straight nan", "turn inf", "turn 错误"):
            with self.subTest(command=command), self.assertRaisesRegex(ValueError, "tasks.txt:3"):
                self.parse(f"[主线]\nstraight 0.48\n{command}\n")
