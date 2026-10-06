"""主线选择、视觉位置和只读反馈的行为回归。"""
import contextlib
import io
import threading
import time
import unittest
from unittest.mock import Mock

from arm.api import Arm as ArmSession
from control import MotionCancelled
from vision.qrcode import mission_code
from vision.targets import candidates_from_detections, choose_position, colored_targets, observe


def balls(names=('红球', '绿球', '蓝球')):
    return [{'name': name, 'box': [i * 100, 10, i * 100 + 40, 50], 'confidence': 0.99}
            for i, name in enumerate(names)]


class DecisionTests(unittest.TestCase):
    def test_three_digits_are_independent(self):
        for ball in '123':
            for target in '123':
                for shape in '123':
                    code = mission_code(ball + target + shape)
                    self.assertEqual(code['ball'], ('red', 'green', 'blue')[int(ball)-1])
                    self.assertEqual(code['target'], ('red', 'green', 'blue')[int(target)-1])
                    self.assertEqual(code['object'], ('cylinder', 'cone', 'drum')[int(shape)-1])
        self.assertEqual(mission_code(' 211\n'), {'ball': 'green', 'target': 'red', 'object': 'cylinder'})
        for invalid in (None, 211, '123+321', '012', '234', '11', '1111', '２１１'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                mission_code(invalid)



class TargetTests(unittest.TestCase):
    def test_order_uses_centers_and_ignores_other_kinds(self):
        detections = list(reversed(balls())) + [{'name': '圆柱', 'box': [1, 1, 3, 3]}]
        position, candidates = choose_position(candidates_from_detections(detections, 'ball'), 'green')
        self.assertEqual(position, 1)
        self.assertEqual([c['value'] for c in candidates], ['red', 'green', 'blue'])
        self.assertEqual(len(candidates_from_detections(detections, 'object')), 1)

    def test_missing_duplicated_and_extra_targets_do_not_guess(self):
        for names, value in ((('红球', '绿球'), 'green'), (('绿球', '绿球', '蓝球'), 'green'),
                             (('红球', '红球', '蓝球'), 'green'),
                             (('红球', '绿球', '蓝球', '红球'), 'green')):
            with self.subTest(names=names), self.assertRaises(ValueError):
                choose_position(candidates_from_detections(balls(names), 'ball'), value)

    def test_colored_targets_use_real_opencv_contours(self):
        import numpy as np
        frame = np.zeros((240, 600, 3), dtype=np.uint8)
        frame[60:180, 30:150] = (0, 255, 0)
        frame[60:180, 230:350] = (0, 0, 255)
        frame[60:180, 430:550] = (255, 0, 0)
        position, candidates = choose_position(colored_targets(frame), 'red')
        self.assertEqual(position, 1)
        self.assertEqual([c['value'] for c in candidates], ['green', 'red', 'blue'])

    def test_observation_requires_new_and_stable_frames(self):
        camera = Mock(condition=threading.Condition(), index=10)
        camera.next_frame.side_effect = [(object(), i, time.time()) for i in range(11, 17)]
        predictor = Mock()
        predictor.predict.side_effect = [{'detections': balls(names)} for names in
            [('红球', '绿球', '蓝球'), ('绿球', '红球', '蓝球'), ('红球', '绿球', '蓝球'),
             ('红球', '绿球', '蓝球'), ('红球', '绿球', '蓝球')]]
        result = observe(camera, 'ball', 'green', predictor)
        self.assertEqual(result['position'], 1)
        self.assertEqual(result['frame_index'], 15)
        self.assertEqual(camera.next_frame.call_args_list[0].kwargs['after'], 10)
        stop = threading.Event()
        stop.set()
        with self.assertRaises(MotionCancelled):
            observe(camera, 'ball', 'green', predictor, stop_event=stop)


class AngleReadTests(unittest.TestCase):
    def test_read_only_feedback_reuses_connections_and_preserves_servo_ids(self):
        session = ArmSession(simulate=True)
        try:
            data = session.angles()
            self.assertFalse(data['calibrated'])
            self.assertTrue(all(not motor.enabled for motor in session._bus.motors.values()))
            self.assertFalse(session._servo._motion_commanded)
            servo1 = data['servos']['1']['angle_deg']
            bus, servo = session._bus, session._servo
            with contextlib.redirect_stdout(io.StringIO()):
                session.calibrate()
                session.move_joints(20, 30, 140, 40, 'open')
            data = session.angles()
            self.assertIs(session._arm._bus, bus)
            self.assertIs(session._arm._gripper, servo)
            self.assertEqual(session._servo.servo_id, 2)
            self.assertAlmostEqual(data['servos']['1']['angle_deg'], servo1)
            self.assertAlmostEqual(data['servos']['2']['angle_deg'], 291, delta=0.1)
            for addr, angle in enumerate((20, 30, 140, 40), 1):
                self.assertEqual(data['motors'][str(addr)]['joint_deg'], angle)
        finally:
            session.close()

    def test_one_failed_device_does_not_hide_other_angles(self):
        session = ArmSession(simulate=True)
        try:
            session.angles()
            session._bus.motors[2].read_status = Mock(side_effect=OSError('offline'))
            data = session.angles()
            self.assertEqual(data['motors']['2']['error'], 'offline')
            self.assertIn('encoder_deg', data['motors']['1'])
            self.assertIn('angle_deg', data['servos']['1'])
        finally:
            session.close()
