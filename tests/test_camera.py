"""以假摄像头验证后台采集、分辨率切换与新帧等待，不访问真实设备。"""
from collections import deque
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2

from vision.camera import CameraStream


class FakeCapture:
    def __init__(self):
        self.width, self.height = 1280, 720
        self.closed = self.reading = False
        self.reject_1080p = False
        self.buffered = deque()

    def isOpened(self):
        return not self.closed

    def set(self, key, value):
        if self.reading:
            raise AssertionError('不能在读取画面期间修改摄像头属性')
        if self.reject_1080p and value in (1920, 1080):
            return False
        if key == cv2.CAP_PROP_FRAME_WIDTH:
            if value != self.width:
                self.buffered.append(SimpleNamespace(shape=(self.height, self.width, 3)))
            self.width = value
        elif key == cv2.CAP_PROP_FRAME_HEIGHT:
            self.height = value
        return True

    def get(self, key):
        return self.width if key == cv2.CAP_PROP_FRAME_WIDTH else self.height

    def read(self):
        self.reading = True
        try:
            time.sleep(0.002)
            if self.closed:
                return False, None
            frame = self.buffered.popleft() if self.buffered else SimpleNamespace(
                shape=(self.height, self.width, 3))
            return True, frame
        finally:
            self.reading = False

    def release(self):
        if self.reading:
            raise AssertionError('不能在读取画面期间释放摄像头')
        self.closed = True


class CameraTests(unittest.TestCase):
    def test_close_waits_for_active_read_before_release(self):
        capture = FakeCapture()
        entered, finish = threading.Event(), threading.Event()
        errors = []

        def read():
            capture.reading = True
            entered.set()
            try:
                if not finish.wait(2.0):
                    raise TimeoutError('测试未解除摄像头读取等待')
                return True, SimpleNamespace(shape=(capture.height, capture.width, 3))
            finally:
                capture.reading = False

        capture.read = read
        with patch('cv2.VideoCapture', return_value=capture):
            camera = CameraStream()

        def close():
            try:
                camera.close()
            except BaseException as exc:
                errors.append(exc)

        closer = threading.Thread(target=close)
        try:
            self.assertTrue(entered.wait(1.0))
            closer.start()
            self.assertTrue(camera.stop.wait(1.0))
            self.assertFalse(capture.closed)
        finally:
            finish.set()
            if closer.ident is not None:
                closer.join(timeout=3.0)
            else:
                camera.close()
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(capture.closed)
        self.assertFalse(camera.thread.is_alive())

    def test_resolution_switch_drops_buffered_old_size_and_publishes_fresh_frame(self):
        capture = FakeCapture()
        with patch('cv2.VideoCapture', return_value=capture):
            camera = CameraStream()
        try:
            frame, index, stamp = camera.next_frame()
            self.assertEqual(frame.shape, (720, 1280, 3))
            for size in ((1920, 1080), (1280, 720)):
                camera.set_resolution(*size)
                frame, newer, captured = camera.next_frame(after=index, after_stamp=stamp)
                self.assertEqual(frame.shape, (size[1], size[0], 3))
                self.assertGreater(newer, index)
                self.assertGreater(captured, stamp)
                index, stamp = newer, captured
        finally:
            camera.close()
        self.assertTrue(capture.closed)
        self.assertFalse(camera.thread.is_alive())

    def test_unsupported_resolution_can_restore_and_continue_capture(self):
        capture = FakeCapture()
        capture.reject_1080p = True
        with patch('cv2.VideoCapture', return_value=capture):
            camera = CameraStream()
        try:
            _, index, _ = camera.next_frame()
            with self.assertRaisesRegex(RuntimeError, '不支持 1920×1080'):
                camera.set_resolution(1920, 1080)
            camera.set_resolution(1280, 720)
            frame, _, _ = camera.next_frame(after=index)
            self.assertEqual(frame.shape, (720, 1280, 3))
            self.assertEqual(camera.error, '')
        finally:
            camera.close()

    def test_timestamp_wait_does_not_repeat_same_frame_and_honors_close(self):
        camera = CameraStream.__new__(CameraStream)
        camera.condition = threading.Condition()
        camera.stop = threading.Event()
        camera.frame = object()
        camera.index, camera.capture_stamp, camera.error = 1, time.time(), ''
        with self.assertRaises(TimeoutError):
            camera.next_frame(after_stamp=camera.capture_stamp, timeout=0)
        camera.stop.set()
        with self.assertRaisesRegex(RuntimeError, '已关闭'):
            camera.next_frame(timeout=0)
