"""从其他目录直接启动调试脚本，不借助 PYTHONPATH 或连接硬件。"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class DebugEntrypointTests(unittest.TestCase):
    def test_scripts_start_from_another_directory_in_isolated_python(self):
        with tempfile.TemporaryDirectory() as folder:
            for script in ('camera_web.py', 'angles.py'):
                with self.subTest(script=script):
                    result = subprocess.run(
                        [sys.executable, '-I', '-B', str(ROOT / 'debug' / script), '--help'],
                        cwd=folder, capture_output=True, text=True, timeout=10,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('usage:', result.stdout)

    def test_module_entrypoints_still_start_from_project_root(self):
        for module in ('debug.camera_web', 'debug.angles'):
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, '-B', '-m', module, '--help'],
                    cwd=ROOT, capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)
