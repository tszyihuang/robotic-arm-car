import unittest
from base import arc_turn as arc

class PlannerTests(unittest.TestCase):
    def test_low_speed_arc_keeps_requested_cruise_and_inner_wheel_below_old_floor(self):
        for sign in (-1, 1):
            p = arc.make_plan(.24, sign * 45, .03, None, .41)
            self.assertEqual(p.cruise, .03)
            travel = yaw = 0.0
            for _ in range(100):
                left, right = p.step(travel, yaw, .01, p.w)
                travel += (left + right) * .005
                yaw += (right - left) / p.track * .01
            self.assertFalse(p.braking)
            self.assertAlmostEqual((left + right) / 2, .03)
            self.assertGreater(min(left, right), 0)
            self.assertLess(min(left, right), .07)

    def test_in_place_turn_slows_below_old_floor_near_target(self):
        for sign in (-1, 1):
            p = arc.make_plan(0, sign * 45, .25, None, .41)
            yaw = p.angle - sign * arc.math.radians(2)
            for _ in range(100):
                left, right = p.step(0, yaw, .01, p.w)
            self.assertFalse(p.braking)
            self.assertAlmostEqual(left, -right)
            self.assertGreater(sign * right, 0)
            self.assertLess(abs(right), .07)

    def test_mirrored_arcs_reach_goal_without_reversing_inner_wheel(self):
        for radius, degrees in ((0.38, 87), (0.54, 45), (0.3, 180)):
            results = []
            for sign in (-1, 1):
                p = arc.make_plan(radius, sign * degrees, 0.25, None, 0.41)
                travel = yaw = 0.0
                for _ in range(2000):
                    left, right = p.step(travel, yaw, 0.01, p.w)
                    self.assertGreaterEqual(min(left, right), -1e-10)
                    self.assertLessEqual(max(left, right), p.wheel_limit + 1e-10)
                    if p.braking:
                        break
                    self.assertGreater(left + right, 0)
                    travel += (left + right) * 0.005
                    yaw += (right - left) / p.track * 0.01
                else:
                    self.fail('圆弧未结束')
                self.assertLessEqual(abs(p.angle - yaw), p.tolerance)
                self.assertLess(abs(travel / yaw / (sign * radius) - 1), 0.05)
                results.append((travel, yaw))
            self.assertAlmostEqual(results[0][0], results[1][0])
            self.assertAlmostEqual(results[0][1], -results[1][1])

    def test_brake_never_restarts_or_spins_to_correct_angle(self):
        p = arc.make_plan(0.38, 87, 0.25, None, 0.41)
        for _ in range(100):
            p.step(0.2, 0.5, 0.01)
        self.assertEqual(p.step(p.length, p.angle, 0.01), (0.0, 0.0))
        for travel in (p.length, p.length - 0.05, 0):
            self.assertEqual(p.step(travel, 0, 0.01), (0.0, 0.0))

    def test_distance_reached_with_heading_lag_does_not_stop(self):
        p = arc.make_plan(.35, 87, .15, None, .41)
        left, right = p.step(p.length, arc.math.radians(55.31), .01)
        self.assertFalse(p.braking)
        self.assertGreater(right, left)
        self.assertGreater(left + right, 0)

    def test_heading_ahead_of_distance_keeps_radius_correction_active_until_angle_crosses(self):
        p = arc.make_plan(.35, 30, .15, None, .41)
        yaw = p.angle - arc.math.radians(.5)
        left, right = p.step(.14, yaw, .01, .4)
        self.assertFalse(p.braking)
        self.assertGreater(left + right, 0)
        # 即使半径仍有偏差，跨过目标转角也立即刹车，防止继续过转。
        self.assertEqual(p.step(.14, p.angle, .01, .4), (0.0, 0.0))

    def test_measured_turn_rate_advances_braking_for_both_directions(self):
        for sign in (-1, 1):
            p = arc.make_plan(.35, sign * 87, .15, None, .41)
            yaw = p.angle - sign * arc.math.radians(2)
            self.assertEqual(p.step(p.length, yaw, .01, sign * 1.5), (0.0, 0.0))
            self.assertTrue(p.braking)

    def test_repeated_distance_frame_does_not_cause_speed_jump(self):
        p = arc.make_plan(0.54, -45, 0.25, None, 0.41)
        travel = 0.1
        for _ in range(100):
            p.step(travel, -travel / p.radius, 0.01)
        before = p.step(travel, -travel / p.radius, 0.01)
        after = p.step(travel, -travel / p.radius - 0.004, 0.01)
        self.assertLess(max(abs(a-b) for a, b in zip(before, after)), 0.001)

    def test_heading_correction_and_deceleration_respect_wheel_limits(self):
        for sign in (-1, 1):
            p = arc.make_plan(0.38, sign * 87, 0.25, 0.15, 0.41)
            for travel, lag in ((0.1, 0.15), (0.2, -0.15), (p.length-0.01, 0.15)):
                for _ in range(100):
                    left, right = p.step(travel, sign*travel/p.radius+lag, 0.01)
                    self.assertGreaterEqual(min(left, right), -1e-10)
                    self.assertLessEqual(max(left, right), 0.15 + 1e-10)

    def test_in_place_turn_checks_angle_and_mirrors_wheel_commands(self):
        p = arc.make_plan(0, -42, 0.25, None, 0.41)
        for _ in range(20):
            left, right = p.step(0, 0, 0.01)
        self.assertGreater(left, 0)
        self.assertAlmostEqual(left, -right)
        self.assertEqual(p.step(0, p.angle, 0.01), (0, 0))

class WheelFeedbackTests(unittest.TestCase):
    def test_fit_at_constant_speed_attenuates_duplicate_frame(self):
        ctrl = arc.WheelFeedback()
        errors = []
        previous = 0
        for n in range(200):
            count = round(n * .01 * .25 * arc.COUNTS_PER_METER)
            if n % 20 == 0:
                count = previous
            speeds = ctrl.measure(n*.01, (count,) * 4)
            previous = count
            if n > 20:
                errors.append(abs(speeds[0] - .25))
        # 单帧求导会误报零速度（误差 250mm/s）；窗口拟合误差应低于 10%。
        self.assertLess(max(errors), .025)
