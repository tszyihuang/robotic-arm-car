"""角度调试的启动零点、逻辑方向、纯数值输出及退出清理。"""
import contextlib
import io
import json
import unittest
from unittest.mock import patch

from arm.api import Arm
from debug import angles


class AnglesDebugTests(unittest.TestCase):
    def test_missing_logical_feedback_keeps_numeric_columns(self):
        data = {'motors': {'1': {'encoder_deg': 999}, '2': {'joint_deg': 12.5}},
                'servos': {'2': {'angle_deg': 243}}}
        self.assertEqual(angles.format_angles(data), 'nan 12.50 nan nan nan 243.00')

    def test_each_start_rebases_current_encoders_and_prints_logical_angles(self):
        for json_mode in (False, True):
            with self.subTest(json_mode=json_mode):
                arm = Arm(simulate=True)
                # 即使配置有旧零点，也应该用启动时当前位置建立基准。
                arm.config.encoder_zero_deg = {1: 100, 2: 200, 3: 300, 4: 400}
                bus = arm._ensure_bus()
                # 归零之后改变位置，验证输出包含安装偏置与关节方向。
                read_angles = arm.angles

                def sample():
                    self.assertTrue(arm.calibrated)
                    for addr, delta in enumerate((10, 20, 30, 40), 1):
                        bus.motors[addr].angle = arm._arm.encoder_zero_deg[addr] + delta
                    return read_angles()

                args = ['--duration', '0.1'] + (['--json'] if json_mode else [])
                clock = [0.0]

                def advance(seconds):
                    clock[0] += seconds

                with patch('debug.angles.Arm', return_value=arm), \
                        patch.object(arm, 'angles', side_effect=sample), \
                        patch('debug.angles.time.monotonic', side_effect=lambda: clock[0]), \
                        patch('debug.angles.time.sleep', side_effect=advance), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    result = angles.main(args)
                self.assertEqual(result, 0)
                lines = output.getvalue().splitlines()
                self.assertEqual(len(lines), 1)
                if json_mode:
                    data = json.loads(lines[0])
                    self.assertTrue(data['calibrated'])
                    values = [data['motors'][str(addr)]['joint_deg'] for addr in (1, 2, 3, 4)]
                else:
                    columns = lines[0].split()
                    self.assertEqual(len(columns), 6)
                    values = [float(value) for value in columns[:4]]
                self.assertEqual(values, [10, 20, 130, 64])
                self.assertTrue(all(not motor.enabled for motor in bus.motors.values()))
                self.assertIsNone(arm._bus)
