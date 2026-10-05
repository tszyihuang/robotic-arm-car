import json
import threading
import time
import unittest
from types import SimpleNamespace
from car_nodes.common.safety import SafetyGate
from car_nodes.base.vision_source import RosVisionSource
from car_nodes.base.vision_straight import BoundarySample, VisionCfg, track_view

class RosSafetyGateTests(unittest.TestCase):
    def test_only_one_concurrent_goal_is_admitted(self):
        gate, results = SafetyGate(), []
        barrier = threading.Barrier(10)

        def admit():
            barrier.wait()
            results.append(gate.reserve())

        workers = [threading.Thread(target=admit) for _ in range(10)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        self.assertEqual(sum(results), 1)
        gate.release()
        self.assertTrue(gate.reserve())

    def test_stop_latches_after_worker_exits_and_reset_requires_idle(self):
        gate = SafetyGate()
        self.assertTrue(gate.reserve())
        gate.trip()
        self.assertTrue(gate.stop_event.is_set())
        self.assertFalse(gate.reset())
        gate.release()
        self.assertFalse(gate.reserve())
        self.assertTrue(gate.reset())
        self.assertTrue(gate.reserve())

    def test_cancel_does_not_latch_stop_for_the_next_goal(self):
        gate = SafetyGate()
        self.assertTrue(gate.reserve())
        gate.stop_event.set()
        gate.release()
        self.assertTrue(gate.reserve())
        self.assertFalse(gate.stop_event.is_set())

def boundary(index=1, stamp=100.0, *, info=None, state="ready", error="", inference_ms=40.0):
    if info is None:
        info = {"size": [1280, 720], "left": {"a": -0.3, "b": 496.0},
                "right": {"a": 0.3, "b": 784.0}}
    nanos = round(stamp * 1e9)
    sec, nanosec = divmod(nanos, 1_000_000_000)
    return SimpleNamespace(frame_index=index, header=SimpleNamespace(stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
                           state=state, error=error, inference_ms=inference_ms,
                           geometry_json=json.dumps(info))

class RosVisionSourceTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.1
        self.source = RosVisionSource(clock=lambda: self.now)

    def test_capture_age_includes_transport_and_inference(self):
        self.source.ingest(boundary())
        sample = self.source.sample()
        self.assertIsInstance(sample, BoundarySample)
        self.assertEqual((sample.frames, sample.seq), (1, 1))
        self.assertAlmostEqual(sample.t_valid, 100.0)
        self.assertAlmostEqual(sample.frame_dt, 0.1)
        self.assertAlmostEqual(self.source.age(), 0.1)
        self.assertIsNotNone(track_view(VisionCfg(), sample.info))
        self.assertTrue(self.source.wait_ready(0, log=lambda _: None))

    def test_heartbeat_cannot_keep_a_frozen_camera_alive(self):
        self.source.ingest(boundary())
        self.now = 100.9
        self.source.ingest(boundary())
        sample = self.source.sample()
        self.assertEqual(sample.seq, 1)
        self.assertEqual(sample.t, self.now)
        self.assertEqual(sample.t_valid, 100.0)
        self.assertAlmostEqual(self.source.age(), 0.9)
        self.assertFalse(self.source.wait_ready(0, log=lambda _: None))

    def test_duplicate_index_with_rewritten_timestamp_is_not_a_new_frame(self):
        self.source.ingest(boundary())
        self.now = 101.0
        self.source.ingest(boundary(stamp=101.0))
        self.assertEqual(self.source.sample().seq, 1)
        self.assertAlmostEqual(self.source.age(), 1.0)

    def test_new_frame_renews_sample_but_no_boundary_retains_old_age(self):
        self.source.ingest(boundary())
        self.now = 100.3
        self.source.ingest(boundary(2, 100.2))
        self.assertEqual(self.source.sample().seq, 2)
        self.now = 100.5
        self.source.ingest(boundary(3, 100.4, info={"left": None, "right": None}))
        self.assertEqual(self.source.sample().seq, 2)
        self.assertAlmostEqual(self.source.age(), 0.3)

    def test_queued_stale_frames_and_out_of_order_frames_are_not_valid(self):
        self.source.ingest(boundary(5, 100.0))
        self.source.ingest(boundary(4, 99.9))
        self.assertEqual(self.source.sample().frames, 5)
        self.now = 102.0
        self.source.ingest(boundary(6, 100.2))
        self.assertEqual(self.source.sample().seq, 1)
        self.assertAlmostEqual(self.source.age(), 2.0)

    def test_camera_restart_accepts_zero_counter_with_new_capture_stamp(self):
        self.source.ingest(boundary(20, 100.0))
        self.now = 100.4
        self.source.ingest(boundary(0, 100.3))
        self.assertEqual((self.source.sample().frames, self.source.sample().seq), (0, 2))

    def test_inference_duration_is_a_conservative_delay_floor(self):
        self.source.ingest(boundary(stamp=100.1, inference_ms=200.0))
        self.assertAlmostEqual(self.source.sample().frame_dt, 0.2)
        self.assertAlmostEqual(self.source.age(), 0.2)

    def test_invalid_geometries_do_not_refresh_validity(self):
        self.source.ingest(boundary())
        bad_infos = [{}, {"left": "bad"}, {"left": {"a": float("nan"), "b": 0}},
                     {"left": {"a": 0, "b": 4, "near_y": "bad"}},
                     {"size": [0, 720], "left": {"a": 0, "b": 4}}]
        for index, info in enumerate(bad_infos, 2):
            self.source.ingest(boundary(index, 100.1, info=info))
            self.assertEqual(self.source.sample().seq, 1)
        self.assertEqual(self.source.sample().t_valid, 100.0)

    def test_missing_or_future_timestamps_do_not_allow_movement(self):
        for message in (boundary(stamp=0.0), boundary(index=2, stamp=105.0)):
            self.source.ingest(message)
            self.assertEqual(self.source.sample().seq, 0)
            self.assertGreater(self.source.age(), 1e6)
        # A bad publisher timestamp must not prevent subsequent sane frames.
        self.source.ingest(boundary(index=3, stamp=100.0))
        self.assertEqual(self.source.sample().seq, 1)

    def test_error_status_is_reported_and_wait_can_be_cancelled(self):
        self.source.ingest(boundary(state="error", error="camera disconnected"))
        self.assertFalse(self.source.wait_ready(5, log=lambda _: None))
        source, cancelled = RosVisionSource(), threading.Event()
        cancelled.set()
        self.assertFalse(source.wait_ready(5, log=lambda _: None, stop_event=cancelled))

    def test_close_discards_data_and_rejects_further_callbacks(self):
        self.source.ingest(boundary())
        self.source.close()
        self.source.ingest(boundary(2, 100.1))
        self.assertEqual(self.source.sample().state, "off")
        self.assertIsNone(self.source.sample().info)
        self.assertGreater(self.source.age(), 1e6)
