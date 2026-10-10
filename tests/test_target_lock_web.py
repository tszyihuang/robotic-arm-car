"""真实 HTTP、合成画面和校正文件验证；不连接摄像头或机械臂。"""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import cv2
import numpy as np

from base.control import MotionCancelled
from tasks.target_lock_web import AimPoint, TargetLockWeb

urlopen = build_opener(ProxyHandler({})).open


class AimPointTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "aim.json"

    def test_unsaved_changes_are_session_only_and_saved_changes_reload(self):
        aim = AimPoint(self.path)
        self.assertEqual(aim.point(), (0.5, 0.5))
        self.assertTrue(aim.set(0.6, 0.3)["dirty"])
        self.assertFalse(self.path.exists())
        self.assertEqual(AimPoint(self.path).point(), (0.5, 0.5))
        self.assertFalse(aim.save()["dirty"])
        self.assertEqual(AimPoint(self.path).point(), (0.6, 0.3))
        aim.set(0.5, 0.5)
        aim.save()
        self.assertEqual(AimPoint(self.path).point(), (0.5, 0.5))

    def test_failed_save_keeps_the_previous_file_and_no_temporary_file(self):
        aim = AimPoint(self.path)
        aim.save()
        before = self.path.read_bytes()
        aim.set(0.2, 0.3)
        with patch("tasks.target_lock_web.os.replace", side_effect=OSError("写入失败")), \
                self.assertRaises(OSError):
            aim.save()
        self.assertEqual(self.path.read_bytes(), before)
        self.assertTrue(aim.status()["dirty"])
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_change_during_save_does_not_block_control_or_mark_new_point_saved(self):
        aim = AimPoint(self.path)
        aim.set(0.6, 0.3)

        def change_while_saving(_):
            # 磁盘写入期间仍可从控制线程取值/设置新点，文件保存的是先前快照。
            self.assertEqual(aim.point(), (0.6, 0.3))
            aim.set(0.7, 0.4)

        with patch("tasks.target_lock_web.os.fsync", side_effect=change_while_saving):
            result = aim.save()
        self.assertTrue(result["dirty"])
        self.assertEqual(aim.point(), (0.7, 0.4))
        self.assertEqual(AimPoint(self.path).point(), (0.6, 0.3))

    def test_invalid_coordinates_or_files_are_rejected(self):
        aim = AimPoint(self.path)
        for x, y in ((-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5),
                     (0.5, float("inf")), (True, 0.5), ("0.5", 0.5)):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                aim.set(x, y)
        self.assertEqual(aim.point(), (0.5, 0.5))
        for data in ({"x": 0.4}, {"x": 0.4, "y": 1.5}, [], {"x": False, "y": 0.5}):
            self.path.write_text(json.dumps(data))
            with self.subTest(data=data), self.assertRaises(ValueError):
                AimPoint(self.path)


class SyntheticCamera:
    def __init__(self):
        self.frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.frame[70:90, 210:230] = (0, 0, 255)

    def next_frame(self, *, after=0, timeout=0.2, stop_event=None):
        if stop_event.wait(0.01):
            raise MotionCancelled("预览已结束")
        return self.frame, after + 1, time.time()


class TargetLockWebTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "aim.json"
        self.aim = AimPoint(self.path)
        self.web = TargetLockWeb(self.aim, "127.0.0.1", 0)
        self.addCleanup(self.web.close)
        self.web.start(SyntheticCamera())
        self.url = "http://127.0.0.1:" + str(self.web.port)

    def post(self, path, body):
        return urlopen(Request(self.url + path, json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"}), timeout=3)

    def test_live_page_jpeg_and_tracking_status(self):
        with urlopen(self.url, timeout=3) as response:
            page = response.read().decode()
        self.assertIn("保存校正", page)
        self.assertIn("/api/aim", page)
        with urlopen(self.url + "/frame.jpg", timeout=3) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (240, 320, 3))
        self.web.update("锁定 red", {"size": (320, 240), "errors": {1: 60, 4: -40},
                                    "target": {"value": "red", "box": [210, 70, 230, 90]}})
        with urlopen(self.url + "/api/status", timeout=3) as response:
            status = json.load(response)
        self.assertEqual(status["size"], [320, 240])
        self.assertEqual(status["state"], "锁定 red")
        self.assertEqual(status["target"]["color"], "red")
        self.assertEqual(status["target"]["box"], [210 / 320, 70 / 240, 230 / 320, 90 / 240])
        self.assertEqual(status["errors"], {"1": 60, "4": -40})
        self.assertEqual(status["aim"], {"x": 0.5, "y": 0.5, "dirty": False})

    def test_http_aim_apply_save_and_reset(self):
        with self.post("/api/aim", {"x": 0.7, "y": 0.3}) as response:
            result = json.load(response)
        self.assertTrue(result["ok"])
        self.assertTrue(result["aim"]["dirty"])
        self.assertEqual(self.aim.point(), (0.7, 0.3))
        self.assertFalse(self.path.exists())
        with self.post("/api/aim/save", {}) as response:
            self.assertFalse(json.load(response)["aim"]["dirty"])
        self.assertEqual(AimPoint(self.path).point(), (0.7, 0.3))
        with self.post("/api/aim/reset", {}) as response:
            self.assertTrue(json.load(response)["aim"]["dirty"])
        self.assertEqual(self.aim.point(), (0.5, 0.5))
        self.assertEqual(AimPoint(self.path).point(), (0.7, 0.3))

    def test_invalid_requests_cannot_change_or_save_aim(self):
        for path, body in (("/api/aim", {"x": 0.6}), ("/api/aim", []),
                           ("/api/aim", {"x": -1, "y": 0.5}),
                           ("/api/aim", {"x": True, "y": 0.5}),
                           ("/api/aim", {"x": float("nan"), "y": 0.5}),
                           ("/api/aim/save", {"path": "/tmp/unexpected.json"}),
                           ("/api/aim/reset", {"x": 0.8})):
            with self.subTest(path=path, body=body), self.assertRaises(HTTPError) as failed:
                self.post(path, body)
            self.assertEqual(failed.exception.code, 400)
            failed.exception.close()
        self.assertEqual(self.aim.point(), (0.5, 0.5))
        self.assertFalse(self.path.exists())

    def test_stream_shares_encoded_frames_and_shutdown_ends_readers(self):
        with urlopen(self.url + "/stream.mjpg", timeout=3) as response:
            self.assertEqual(response.headers["Content-Type"],
                             "multipart/x-mixed-replace; boundary=frame")
            for _ in range(2):
                self.assertEqual(response.readline(), b"--frame\r\n")
                self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
                size = int(response.readline().decode().split(":")[1])
                self.assertEqual(response.readline(), b"\r\n")
                self.assertTrue(response.read(size).startswith(b"\xff\xd8"))
                self.assertEqual(response.read(2), b"\r\n")
            self.web.close()
            response.read()
        self.assertTrue(all(not thread.is_alive() for thread in self.web._threads))
        with self.assertRaises(RuntimeError):
            self.web.snapshot()

    def test_stale_image_is_not_served(self):
        self.web.snapshot()
        # 预览已结束，旧 JPEG 虽仍在内存中也不能用作实时画面。
        self.web.closed.set()
        self.web._threads[0].join(timeout=1)
        self.web.closed.clear()
        with self.web._condition:
            self.web._stamp = time.time() - 1
        with self.assertRaises(TimeoutError):
            self.web.snapshot(timeout=0.02)
