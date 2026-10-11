"""独立车辆/针孔相机模型验证车身位移闭环，不连接硬件。

假设无侧滑、轮速响应时间 80ms；相机高 180mm、俯角 25°、跑道宽 300mm。
这些是假设的仿真参数，不是实车标定值。编码器按项目比例取整，视觉 10Hz。
"""
import math
import unittest
from contextlib import redirect_stdout
from collections import deque
from io import StringIO
from unittest.mock import patch

from base import vision_straight as vs
from base.api import Base
from config import BASE, VISION_CONTROL
from vision.boundary import BoundarySample


class LaneWorld:
    def __init__(self, offset_mm, track_mm=190, delay=0.15, lookahead=0.5):
        self.now = 100.0
        self.x = offset_mm
        self.yaw = 0.0
        self.track = track_mm
        self.delay = delay
        self.cfg = vs.VisionCfg(speed=BASE['speed'], lookahead=lookahead)
        self.velocity = [0.0, 0.0]
        self.command = [0.0, 0.0]
        self.travel = [0.0, 0.0]
        self.last_counts = [0, 0]
        self.poses = deque([(self.now - i * 0.01, self.x, self.yaw)
                            for i in range(math.ceil(delay / 0.01) + 1, -1, -1)], maxlen=1000)
        self.ticks = 0
        self.frames = 0
        self.max_yaw = 0.0
        self.max_diff = 0.0
        self.publish()

    def geometry(self, offset, yaw):
        f = self.cfg.focal(1280)
        pitch = math.radians(25)
        def edge(side):
            points = []
            for z in (300.0, 1500.0):
                dx = side - offset
                cx = dx * math.cos(yaw) + z * math.sin(yaw)
                cz = -dx * math.sin(yaw) + z * math.cos(yaw)
                cy = 180.0 * math.cos(pitch) - cz * math.sin(pitch)
                depth = 180.0 * math.sin(pitch) + cz * math.cos(pitch)
                points.append((640 + f * cx / depth, 360 + f * cy / depth))
            (xn, yn), (xf, yf) = points
            a = (xn - xf) / (yn - yf)
            return {'a': a, 'b': xf - a * yf, 'near_y': yn, 'far_y': yf, 'confidence': 1}
        return {'size': [1280, 720], 'left': edge(-150), 'right': edge(150)}

    def publish(self):
        stamp, offset, yaw = self.poses[0]
        for pose in self.poses:
            if pose[0] <= self.now - self.delay + 1e-8:
                stamp, offset, yaw = pose
        self.frames += 1
        self.latest = BoundarySample(self.geometry(offset, yaw), self.frames, self.frames,
                                     'ready', '', self.now, stamp, self.now - stamp)

    def sample(self):
        return self.latest

    def age(self, now=None):
        return (self.now if now is None else now) - self.latest.t_valid

    def feedback(self, *args):
        dt = 0.01
        self.now += dt
        self.ticks += 1
        if self.ticks > 10000:
            raise RuntimeError('simulation runaway')
        for i in range(2):
            self.velocity[i] += dt / 0.08 * (self.command[i] - self.velocity[i])
            self.travel[i] += self.velocity[i] * dt
        v = sum(self.velocity) / 2
        w = (self.velocity[1] - self.velocity[0]) / self.track
        self.yaw += w * dt
        self.x -= v * math.sin(self.yaw) * dt
        self.max_yaw = max(self.max_yaw, abs(math.degrees(self.yaw)))
        self.poses.append((self.now, self.x, self.yaw))
        if self.ticks % 10 == 0:
            self.publish()
        scale = BASE['meters_per_count'] * 1000
        totals = [round(value / scale) for value in self.travel]
        steps = [a - b for a, b in zip(totals, self.last_counts)]
        self.last_counts = totals
        return [totals[0], totals[0], totals[1], totals[1]], [steps[0], steps[0], steps[1], steps[1]]

    def spd(self, left, left_rear, right, right_rear):
        self.command = [round(max(-1000, min(1000, left))), round(max(-1000, min(1000, right)))]
        self.max_diff = max(self.max_diff, abs(self.command[1] - self.command[0]))


class LaneSimulationTests(unittest.TestCase):
    def drive(self, world, goal=1760):
        heading = vs.VisionHeading(world.cfg, world)
        with patch.object(vs.time, 'time', side_effect=lambda: world.now):
            measured, info = vs.vision_straight(world, heading, goal_mm=goal, log=lambda _: None)
        self.assertFalse(info['reason'])
        self.assertAlmostEqual(measured, goal, delta=30)
        self.assertEqual(world.command, [0, 0])

    def test_parallel_offset_converges_with_delayed_vision_and_different_track_widths(self):
        for offset in (-80, 80):
            for track in (190, 410):
                for delay in (0.15, 0.3):
                    with self.subTest(offset=offset, track=track, delay=delay):
                        world = LaneWorld(offset, track_mm=track, delay=delay)
                        self.drive(world)
                        self.assertLess(abs(world.x), 5)
                        self.assertLess(abs(math.degrees(world.yaw)), 2)
                        self.assertGreater(world.max_diff, 20)

    def test_vanishing_heading_leaves_parallel_offset_uncorrected(self):
        world = LaneWorld(80, lookahead='vanishing')
        self.drive(world)
        self.assertAlmostEqual(world.x, 80)
        self.assertAlmostEqual(world.yaw, 0)
        self.assertEqual(world.max_diff, 0)

    def test_short_distance_can_end_before_body_is_centered(self):
        world = LaneWorld(80)
        self.drive(world, goal=350)
        self.assertLess(abs(world.x), 60)
        self.assertGreater(abs(world.x), 5)

    def test_calibrated_straight_converges_to_requested_left_and_right_positions(self):
        for offset in (-50, 50):
            for initial in (-80, 0, 80):
                with self.subTest(offset=offset, initial=initial):
                    world = LaneWorld(initial)
                    world.cfg.lateral_offset_mm = offset
                    self.drive(world)
                    self.assertAlmostEqual(world.x, -offset, delta=5)
                    self.assertLess(abs(math.degrees(world.yaw)), 2)

    def test_task_api_applies_configured_offset_and_zero_restores_centering(self):
        for offset in (50, 0, -50):
            with self.subTest(offset=offset):
                world = LaneWorld(0)
                base = Base()
                base.motor = world
                heading = vs.VisionHeading(world.cfg, world)
                with patch.dict(VISION_CONTROL, LOG_DIR=None, LATERAL_OFFSET_MM=offset), \
                        patch.object(base, '_heading', return_value=heading), \
                        patch.object(vs.time, 'time', side_effect=lambda: world.now), \
                        redirect_stdout(StringIO()):
                    result = base.vision_straight(1.76, world)
                self.assertTrue(result['ok'])
                self.assertEqual(world.cfg.lateral_offset_mm, offset)
                self.assertAlmostEqual(world.x, -offset, delta=5)
                self.assertLess(abs(math.degrees(world.yaw)), 2)
                self.assertEqual(world.command, [0, 0])

    def test_visual_outage_uses_gap_and_recovery_resumes_lateral_correction(self):
        for recover in (False, True):
            with self.subTest(recover=recover):
                world = LaneWorld(80 if recover else 0)
                publish = world.publish

                def interrupted_publish():
                    elapsed = world.now - 100
                    if elapsed < 2 or (recover and elapsed >= 4):
                        publish()

                world.publish = interrupted_publish
                heading = vs.VisionHeading(world.cfg, world)
                with patch.object(vs.time, 'time', side_effect=lambda: world.now):
                    measured, info = vs.vision_straight(world, heading, goal_mm=1760,
                                                       log=lambda _: None)
                self.assertFalse(info['reason'])
                self.assertAlmostEqual(measured, 1760, delta=30)
                self.assertEqual(info['blind_switches'], 1)
                self.assertEqual(info['vision_recoveries'], int(recover))
                self.assertEqual(info['heading_mode'], 'vision' if recover else 'gap-blind')
                self.assertLess(abs(world.x), 5)
                self.assertLess(abs(math.degrees(world.yaw)), 2)
                self.assertEqual(world.command, [0, 0])
