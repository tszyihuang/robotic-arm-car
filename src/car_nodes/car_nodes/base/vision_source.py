"""Adapt ROS boundary messages to the existing visual motion controller.

This module imports ROS only when a node is supplied.  ``ingest`` accepts a
message with the Boundary fields, so freshness rules can run without ROS.
"""
from __future__ import annotations

import json
import math
import threading
import time

from .vision_straight import BoundarySample


def _finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_geometry(info):
    if not isinstance(info, dict):
        return False
    size = info.get("size")
    if size is not None and (not isinstance(size, (list, tuple)) or len(size) != 2
                             or any(not _finite_number(v) or v < 1 or int(v) != v for v in size)):
        return False
    found = False
    for name in ("left", "right"):
        side = info.get(name)
        if side is None:
            continue
        if not isinstance(side, dict) or not all(_finite_number(side.get(k)) for k in ("a", "b")):
            return False
        for key in ("near_y", "far_y", "stop_row", "angle_deg", "confidence"):
            if side.get(key) is not None and not _finite_number(side[key]):
                return False
        found = True
    return found


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


class RosVisionSource:
    """Receive ``vision/boundary`` and preserve capture-time freshness.

    ``sample()`` returns the project's BoundarySample.  Only a new, ready,
    geometrically valid and timely frame advances its ``seq`` and ``t_valid``.
    Repeated status messages retain the last good frame without extending its
    lifetime.  The node's executor must run independently of motion calls.
    """

    def __init__(self, node=None, topic="vision/boundary", max_frame_age=0.6,
                 clock=time.time, max_future_skew=0.1):
        if not math.isfinite(max_frame_age) or max_frame_age <= 0:
            raise ValueError("max_frame_age must be finite and positive")
        if not math.isfinite(max_future_skew) or max_future_skew < 0:
            raise ValueError("max_future_skew must be finite and nonnegative")
        self.max_frame_age = float(max_frame_age)
        self.max_future_skew = float(max_future_skew)
        self._clock = clock
        self._condition = threading.Condition()
        self._sample = BoundarySample(None, None, 0, "off", "", 0.0, 0.0)
        self._last_index = None
        self._latest_stamp = 0.0
        self._closed = False
        self._node, self._subscription = node, None
        if node is not None:
            from car_interfaces.msg import Boundary
            from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

            qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                             reliability=ReliabilityPolicy.RELIABLE)
            self._subscription = node.create_subscription(Boundary, topic, self.ingest, qos)

    def start(self):
        """Compatibility hook; subscriptions are active upon construction."""
        if self._closed:
            raise RuntimeError("vision source is closed")

    def ingest(self, message, now=None):
        """Accept one Boundary message; bad input never renews a good frame."""
        now = self._clock() if now is None else now
        state, error = str(getattr(message, "state", "?")), str(getattr(message, "error", ""))
        info, stamp, index, inference_seconds = None, 0.0, None, 0.0
        try:
            index = int(message.frame_index)
            if index < 0:
                raise ValueError("negative frame index")
            sec, nanosec = int(message.header.stamp.sec), int(message.header.stamp.nanosec)
            if sec < 0 or not 0 <= nanosec < 1_000_000_000:
                raise ValueError("invalid capture timestamp")
            stamp = sec + nanosec / 1e9
            inference_seconds = float(message.inference_ms) / 1000.0
            if not math.isfinite(inference_seconds) or inference_seconds < 0:
                raise ValueError("invalid inference duration")
            info = json.loads(message.geometry_json or "null", parse_constant=_reject_constant)
        except (AttributeError, TypeError, ValueError, OverflowError) as exc:
            state, error = "error", f"Invalid boundary message: {exc}"
        with self._condition:
            if self._closed:
                return
            old = self._sample
            frame_dt = max(0.0, now - stamp, inference_seconds)
            is_new = index is not None and index != self._last_index
            # A newer capture stamp also allows a restarted camera's counter to
            # start at zero. Older captures cannot replace the current frame.
            ordered = stamp >= self._latest_stamp
            timestamp_valid = stamp > 0 and stamp <= now + self.max_future_skew
            if is_new and ordered and timestamp_valid:
                self._last_index, self._latest_stamp = index, stamp
            fresh = timestamp_valid and frame_dt <= self.max_frame_age
            if is_new and ordered and fresh and state == "ready" and not error and _valid_geometry(info):
                self._sample = BoundarySample(info, index, old.seq + 1, state, error,
                                              now, now - frame_dt, frame_dt)
            else:
                self._sample = BoundarySample(old.info, old.frames, old.seq, state, error,
                                              now, old.t_valid, old.frame_dt)
            self._condition.notify_all()

    _ingest = ingest

    def sample(self):
        with self._condition:
            return self._sample

    def age(self, now=None):
        sample = self.sample()
        if not sample.t_valid:
            return 1e9
        return max(0.0, (self._clock() if now is None else now) - sample.t_valid)

    def wait_ready(self, seconds=60.0, log=print, stop_event=None):
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("seconds must be finite and nonnegative")
        deadline, reported = time.monotonic() + seconds, None
        while True:
            if stop_event is not None and stop_event.is_set():
                return False
            sample = self.sample()
            if self._closed:
                return False
            if sample.state == "error":
                log(f"视觉节点报错：{sample.error or sample.state}")
                return False
            if sample.state == "ready" and not sample.error and self.age() <= self.max_frame_age:
                return True
            if sample.state != reported:
                reported = sample.state
                log(f"等待视觉节点与新边界：{sample.state}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                log(f"等待视觉节点超时（{seconds:g}s，状态 {sample.state}）")
                return False
            with self._condition:
                self._condition.wait(min(remaining, 0.05))

    def close(self):
        with self._condition:
            self._closed = True
            self._sample = BoundarySample(None, None, self._sample.seq, "off", "", 0.0, 0.0)
            self._condition.notify_all()
        if self._subscription is not None:
            self._node.destroy_subscription(self._subscription)
            self._subscription = None
