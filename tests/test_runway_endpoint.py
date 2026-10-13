"""中心列灰→黄边缘、多帧防跳变和真实 MJPEG 接口；不打开硬件。"""
from contextlib import redirect_stderr
import io
import json
import socket
import time
import unittest
from unittest.mock import patch
from urllib.request import ProxyHandler, build_opener

import cv2
import numpy as np

from config import VISION
from vision.runway_endpoint import EndpointCandidate, EndpointDetector, MjpegSource, endpoint_config, first_endpoint, main
from vision.runway_endpoint_web import EndpointWeb

urlopen = build_opener(ProxyHandler({})).open
GRAY = (143, 140, 141)
YELLOW = (143, 159, 155)


def scene(y=80, *, width=100, height=160):
    frame = np.full((height, width, 3), GRAY, dtype=np.uint8)
    if y is not None:
        frame[:y] = YELLOW
    return frame


def reference_endpoint(frame, overrides=None):
    """独立按窗口直接取均值，验证批量运算仍返回最靠下的合格端点。"""
    config = endpoint_config(overrides)
    height, width = frame.shape[:2]
    window = config["window"]
    column = cv2.cvtColor(frame[:, width // 2:width // 2 + 1], cv2.COLOR_BGR2LAB)[:, 0].astype(np.float32)
    for y in range(height - window, window - 1, -1):
        upper = column[y - window:y].mean(axis=0)
        lower = column[y:y + window].mean(axis=0)
        yellow_delta = float(upper[2] - lower[2])
        color_delta = float(np.linalg.norm(upper[1:] - lower[1:]))
        if yellow_delta < config["yellow_threshold"] or color_delta < config["color_threshold"]:
            continue
        support = sum(float(value - lower[2]) >= config["yellow_threshold"]
                      for value in column[y - window:y, 2]) / window
        if support >= config["yellow_support"]:
            return EndpointCandidate(width // 2, y, color_delta, yellow_delta)
    return None


class FirstEndpointTests(unittest.TestCase):
    def test_batched_windows_match_direct_means_for_noise_edges_and_minimum_height(self):
        rng = np.random.default_rng(731)
        for window in (1, 3, 8, 17):
            for height in (2 * window - 1, 2 * window, 2 * window + 1, 160, 720):
                noise = rng.integers(0, 256, (height, 3, 3), dtype=np.uint8)
                for frame in (noise, scene(height // 2, width=3, height=height),
                              scene(None, width=3, height=height)):
                    with self.subTest(window=window, height=height, pixel=frame[0, 0]):
                        original = frame.copy()
                        config = {"window": window, "yellow_support": 0.3,
                                  "yellow_threshold": 1.7, "color_threshold": 3.3}
                        self.assertEqual(first_endpoint(frame, config), reference_endpoint(frame, config))
                        np.testing.assert_array_equal(frame, original)

    def test_thresholds_keep_double_precision_at_qualifying_pixel_boundary(self):
        frame = scene()
        initial = reference_endpoint(frame)
        for key, value in (("yellow_threshold", initial.yellow_delta),
                           ("color_threshold", initial.color_delta)):
            for threshold in (value, np.nextafter(value, float("inf")),
                              np.nextafter(value, float("-inf"))):
                config = {key: float(threshold)}
                with self.subTest(config=config):
                    self.assertEqual(first_endpoint(frame, config), reference_endpoint(frame, config))

    def test_low_contrast_gray_to_yellow_matches_observed_camera_colors(self):
        candidate = first_endpoint(scene())
        self.assertEqual(candidate.x, 50)
        self.assertLessEqual(abs(candidate.y - 80), 8)
        self.assertGreaterEqual(candidate.color_delta, 6)
        self.assertGreaterEqual(candidate.yellow_delta, 4)

    def test_only_lowest_qualifying_transition_is_returned(self):
        frame = scene()
        frame[25:50] = GRAY
        self.assertEqual(first_endpoint(frame), first_endpoint(scene()))

    def test_off_center_edges_luminance_shadows_and_isolated_yellow_noise_are_rejected(self):
        frame = scene(None)
        frame[:80, :49] = YELLOW
        frame[:80, 51:] = YELLOW
        self.assertIsNone(first_endpoint(frame))
        shadow = np.full((160, 100, 3), 140, dtype=np.uint8)
        shadow[100:140] = 70
        self.assertIsNone(first_endpoint(shadow))
        frame = scene()
        frame[130, 50] = (0, 220, 255)
        self.assertEqual(first_endpoint(frame), first_endpoint(scene()))

    def test_scan_does_not_use_reversed_yellow_to_gray_transition(self):
        self.assertIsNone(first_endpoint(scene()[::-1].copy()))

    def test_small_or_invalid_frames_and_parameter_validation(self):
        self.assertIsNone(first_endpoint(scene(height=10, y=5)))
        for frame in (None, np.zeros((10, 10), np.uint8), np.zeros((0, 10, 3), np.uint8),
                      np.zeros((10, 10, 3), np.float32)):
            with self.subTest(frame_type=type(frame)), self.assertRaises(ValueError):
                first_endpoint(frame)
        for override in ({"window": 0}, {"stable_frames": True}, {"color_threshold": float("nan")},
                         {"max_jump_px": -1}, {"yellow_support": 1.1}, {"smoothing": 0}, {"unknown": 1}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                endpoint_config(override)


class EndpointTrackingTests(unittest.TestCase):
    def setUp(self):
        # 跟踪用例明确使用各自的确认条件，不依赖现场 config.py 的调参值。
        self.tracking_config = {"smoothing": 1.0, "stable_frames": 3, "jump_frames": 5,
                                "stable_tolerance_px": 6.0, "max_jump_px": 40.0}
        self.detector = EndpointDetector(self.tracking_config)
        self.index = 0

    def sample(self, y=80, *, detector=None, **frame_options):
        self.index += 1
        stamp = 10 + self.index * 0.03
        return (detector or self.detector).process(scene(y, **frame_options), frame_index=self.index,
                                                 capture_stamp=stamp, now=stamp)

    def acquire(self, y=80):
        self.assertIsNone(self.sample(y)["point"])
        self.assertIsNone(self.sample(y)["point"])
        result = self.sample(y)
        self.assertTrue(result["valid"])
        return result["point"]

    def test_initial_confirmation_and_small_movement(self):
        initial = self.acquire()
        result = self.sample(85)
        self.assertEqual(result["state"], "tracking")
        self.assertEqual(result["point"][1] - initial[1], 5)

    def test_large_jump_is_held_until_five_consistent_new_frames(self):
        initial = self.acquire(120)
        for count in range(1, 5):
            result = self.sample(40)
            self.assertEqual(result["point"], initial)
            self.assertEqual(result["pending_count"], count)
            self.assertEqual(result["required_count"], 5)
            self.assertFalse(result["valid"])
        result = self.sample(40)
        self.assertTrue(result["valid"])
        self.assertLess(result["point"][1], initial[1] - 40)

    def test_single_jump_is_discarded_when_old_position_returns(self):
        initial = self.acquire(120)
        self.assertEqual(self.sample(40)["state"], "jump-pending")
        result = self.sample(120)
        self.assertEqual(result["point"], initial)
        self.assertEqual(result["pending_count"], 0)
        self.assertEqual(result["state"], "tracking")

    def test_camera_timeout_interrupts_jump_confirmation(self):
        initial = self.acquire(120)
        self.sample(40)
        self.sample(40)
        self.detector.reset_confirmation()
        result = self.sample(40)
        self.assertEqual(result["point"], initial)
        self.assertEqual(result["pending_count"], 1)

    def test_pending_cluster_is_relative_to_first_candidate_not_last_candidate(self):
        self.acquire(120)
        self.assertEqual(self.sample(40)["pending_count"], 1)
        self.assertEqual(self.sample(45)["pending_count"], 2)
        self.assertEqual(self.sample(50)["pending_count"], 1)

    def test_missing_frames_interrupt_confirmation_and_eventually_clear_old_point(self):
        initial = self.acquire(120)
        self.sample(40)
        result = self.sample(None)
        self.assertEqual(result["point"], initial)
        self.assertEqual(result["pending_count"], 0)
        self.assertEqual(self.sample(40)["pending_count"], 1)
        for _ in range(10):
            result = self.sample(None)
        self.assertIsNone(result["point"])
        self.assertEqual(self.sample(40)["pending_count"], 1)

    def test_repeated_out_of_order_and_stale_frames_cannot_count_as_confirmation(self):
        for index, stamp, now in ((1, 10.03, 10.03), (0, 10.03, 10.03),
                                  (2, 10.01, 10.03), (2, 9.0, 10.03), (2, 11.0, 10.03)):
            with self.subTest(index=index, stamp=stamp, now=now):
                detector = EndpointDetector()
                detector.process(scene(), frame_index=1, capture_stamp=10.03, now=10.03)
                invalid = detector.process(scene(), frame_index=index, capture_stamp=stamp, now=now)
                self.assertFalse(invalid["valid"])
                self.assertIsNone(invalid["point"])
                fresh = detector.process(scene(), frame_index=3, capture_stamp=10.06, now=10.06)
                self.assertEqual(fresh["pending_count"], 1)

    def test_old_point_expires_by_time_and_new_resolution_starts_confirmation(self):
        self.acquire()
        result = self.detector.process(scene(), frame_index=4, capture_stamp=11, now=11)
        self.assertIsNone(result["point"])
        self.assertEqual(result["pending_count"], 1)
        self.detector = EndpointDetector(self.tracking_config)
        self.index = 0
        self.acquire()
        result = self.sample(width=120)
        self.assertIsNone(result["point"])
        self.assertEqual(result["pending_count"], 1)

    def test_small_motion_is_smoothed(self):
        self.detector = EndpointDetector({"stable_frames": 1, "smoothing": 0.35})
        initial = self.sample()["point"]
        result = self.sample(100)
        self.assertGreater(result["point"][1], initial[1])
        self.assertLess(result["point"][1], first_endpoint(scene(100)).y)


class EndpointWebTests(unittest.TestCase):
    def setUp(self):
        self.web = EndpointWeb("127.0.0.1", 0)
        self.addCleanup(self.web.close)
        self.frame = scene()
        self.result = EndpointDetector({"stable_frames": 1}).process(
            self.frame, frame_index=1, capture_stamp=time.time())
        self.web.publish(self.frame, self.result, index=1, stamp=time.time())
        self.web.start()
        self.url = f"http://127.0.0.1:{self.web.server.server_port}"

    def test_page_status_and_annotated_jpeg_share_the_processed_frame(self):
        with urlopen(self.url, timeout=2) as response:
            self.assertIn("跑道端点监视", response.read().decode())
        with urlopen(self.url + "/api/status", timeout=2) as response:
            result = json.load(response)
            self.assertEqual(result["point"], list(self.result["point"]))
            self.assertEqual(result["frame_index"], 1)
            self.assertEqual(result["size"], [100, 160])
        with urlopen(self.url + "/frame.jpg", timeout=2) as response:
            self.assertEqual(response.headers["Content-Type"], "image/jpeg")
            image = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, self.frame.shape)
        self.assertFalse(np.array_equal(image, self.frame))
        np.testing.assert_array_equal(self.frame, scene())

    def test_mjpeg_source_receives_distinct_frames_from_monitor(self):
        source = MjpegSource(self.url + "/stream.mjpg")
        self.addCleanup(source.close)
        first, index, _ = source.next_frame(timeout=2)
        self.assertEqual(first.shape, self.frame.shape)
        self.web.publish(scene(100), self.result, index=2, stamp=time.time())
        second, new_index, _ = source.next_frame(after=index, timeout=2)
        self.assertGreater(new_index, index)
        self.assertFalse(np.array_equal(first, second))

    def test_stale_monitor_clears_status_point_and_never_serves_cached_stale_frame(self):
        with self.web.condition:
            self.web._stamp = time.time() - VISION["frame_stale"] - 0.1
        status = self.web.status()
        self.assertEqual(status["state"], "stale")
        self.assertIsNone(status["point"])
        with self.assertRaises(TimeoutError):
            self.web.snapshot(timeout=0.01)


class EndpointEntrypointTests(unittest.TestCase):
    def test_camera_startup_failure_releases_the_monitor_port(self):
        with socket.socket() as connection:
            connection.bind(("127.0.0.1", 0))
            port = connection.getsockname()[1]
        with patch("vision.camera.CameraStream", side_effect=RuntimeError("摄像头不可用")), \
                redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--host", "127.0.0.1", "--port", str(port)]), 2)
        with socket.socket() as connection:
            connection.bind(("127.0.0.1", port))

    def test_malformed_stream_url_is_reported_before_opening_camera_or_port(self):
        with patch("vision.camera.CameraStream") as camera:
            for url in ("http://localhost:invalid/stream.mjpg", "file:///tmp/test.jpg"):
                with self.subTest(url=url), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    main(["--stream-url", url])
                self.assertEqual(error.exception.code, 2)
            camera.assert_not_called()


if __name__ == "__main__":
    unittest.main()
