import threading
import time
import unittest
from unittest.mock import Mock
from base import straight_pid
from base.control import MotionCancelled


class ControllerTests(unittest.TestCase):
    def test_actual_gap_loop_corrects_left_lead_in_both_directions(self):
        for direction in (1, -1):
            board = Mock()
            board.feedback.side_effect = [([2000] * 4, None),
                ([2000 + direction * 100, 2000 + direction * 100,
                  2000 + direction * 80, 2000 + direction * 80], None),
                *[([2000 + direction * 1000] * 4, None)] * 5]
            board.stop.side_effect = lambda: board.spd(0, 0, 0, 0)
            _, info = straight_pid.straight(board, direction * 100, 100, log=lambda _: None)
            self.assertFalse(info['reason'])
            command = next(c.args for c in board.spd.call_args_list if any(c.args))
            self.assertGreater((command[2] - command[0]) * direction, 0)
            self.assertEqual(board.spd.call_args.args, (0, 0, 0, 0))



class QrTests(unittest.TestCase):
    def test_real_decoder_scans_shared_frame(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest('需要本机 OpenCV/numpy')
        from vision.qrcode import scan_qrcode
        qr = cv2.QRCodeEncoder_create().encode('car:123+321')
        qr = cv2.resize(qr, (400, 400), interpolation=cv2.INTER_NEAREST)
        frame = np.full((1080, 1920, 3), 255, dtype=np.uint8)
        frame[340:740, 760:1160] = cv2.cvtColor(qr, cv2.COLOR_GRAY2BGR)
        camera = Mock()
        camera.next_frame.return_value = frame, 1, time.time()
        self.assertEqual(scan_qrcode(camera), 'car:123+321')
        camera.close.assert_not_called()

    def test_scan_cancel_does_not_wait_for_frame(self):
        from vision.qrcode import scan_qrcode
        stop = threading.Event()
        stop.set()
        camera = Mock()
        with self.assertRaises(MotionCancelled):
            scan_qrcode(camera, stop_event=stop)
        camera.next_frame.assert_not_called()
