import contextlib
import csv
import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
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

    def test_in_place_turn_keeps_movable_speed_near_target(self):
        for sign in (-1, 1):
            p = arc.make_plan(0, sign * 45, .25, None, .41)
            yaw = p.angle - sign * arc.math.radians(2)
            for _ in range(100):
                left, right = p.step(0, yaw, .01, p.w)
            self.assertFalse(p.braking)
            self.assertAlmostEqual(left, -right)
            self.assertGreater(sign * right, 0)
            self.assertGreaterEqual(abs(right), arc.SPIN_MIN_WHEEL_SPEED)
            self.assertLess(abs(right), .07)

    def test_spin_floor_respects_lower_speed_limit_and_brakes_in_both_directions(self):
        for sign in (-1, 1):
            p = arc.make_plan(0, sign * 44, .25, .03, .41)
            for _ in range(100):
                left, right = p.step(0, p.angle - sign * arc.math.radians(2), .01)
            self.assertAlmostEqual(sign * right, .03)
            self.assertAlmostEqual(left, -right)
            self.assertEqual(p.step(0, p.angle - sign * arc.math.radians(.5), .01), (0, 0))
            self.assertEqual(p.step(0, 0, .01), (0, 0))

    def test_spin_brakes_early_using_measured_rate(self):
        for sign in (-1, 1):
            p = arc.make_plan(0, sign * 44, .25, .15, .41)
            yaw = p.angle - sign * arc.math.radians(1.5)
            self.assertEqual(p.step(0, yaw, .01, sign * .3), (0, 0))

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
                self.assertLessEqual(abs(p.angle - yaw), max(arc.math.radians(1), 0.05 * abs(p.angle)))
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

class TurnCompletionTests(unittest.TestCase):
    def simulate_spin(self, degrees, *, blocked=False, use_imu=True, log_path=None):
        """独立轮速响应模型：35 mm/s 以下指令不动，模拟编码器与 IMU 新帧。"""
        now = previous = 100.0
        counts = [0.0] * 4
        speeds = [0.0] * 4
        commands = [0.0] * 4
        yaw = 0.0
        board, imu = Mock(), Mock()
        board.battery = 12.0
        imu.has_data.return_value = True
        imu.age.return_value = 0.0

        def poll():
            nonlocal previous, yaw
            dt = now - previous
            previous = now
            for i, command in enumerate(commands):
                target = command if not blocked and abs(command) >= .035 else 0.0
                speeds[i] += (target - speeds[i]) * (1 - math.exp(-dt / .035))
                counts[i] += speeds[i] * dt * arc.COUNTS_PER_METER
            rate = (sum(speeds[2:]) - sum(speeds[:2])) / (2 * .41)
            yaw += rate * dt
            board.counts = tuple(round(v) for v in counts)
            board.stamp = now
            imu.yaw, imu.rate = math.degrees(yaw), math.degrees(rate)

        def speed(values):
            commands[:] = [round(v * 1000) / 1000 for v in values]

        def brake(values):
            commands[:] = [0.0] * 4
            return (0,) * 4

        def wait(seconds, stop_event=None):
            nonlocal now
            now += seconds

        board.poll.side_effect = poll
        board.speed.side_effect = speed
        board.brake.side_effect = brake
        with patch.object(arc.time, 'monotonic', side_effect=lambda: now), \
                patch.object(arc, 'wait_cancelable', side_effect=wait):
            result = arc.turn_with_radius(0, degrees, board=board, imu=imu if use_imu else None,
                                          max_speed=.15, verbose=False, log_path=log_path)
        board.stop.assert_called_once()
        board.release.assert_called_once()
        return result, board

    def test_spin_finishes_with_motor_dead_zone_for_both_directions_and_heading_sources(self):
        for sign in (-1, 1):
            for use_imu in (False, True):
                with self.subTest(sign=sign, use_imu=use_imu):
                    # 同一个电机模型在缺少最低轮速时会在最后几度触发保护。
                    with patch.object(arc, 'SPIN_MIN_WHEEL_SPEED', 0.0):
                        stalled, _ = self.simulate_spin(sign * 44, use_imu=use_imu)
                    self.assertFalse(stalled['ok'])
                    self.assertIn('编码器持续无变化', stalled['fault'])
                    self.assertLess(abs(stalled['angle_error_deg']), 5)

                    result, board = self.simulate_spin(sign * 44, use_imu=use_imu)
                    self.assertTrue(result['ok'], result['summary'])
                    self.assertLess(abs(result['angle_error_deg']), 2)
                    self.assertLess(result['elapsed_s'], 6)
                    board.brake.assert_called()
                    for call in board.speed.call_args_list:
                        self.assertGreaterEqual(sign * call.args[0][2], 0)
                        self.assertLessEqual(max(abs(v) for v in call.args[0]), .15)

    def test_blocked_spin_still_faults_and_reports_angle_and_commands(self):
        result, board = self.simulate_spin(44, blocked=True)
        self.assertFalse(result['ok'])
        self.assertIn('已转 +0.00°', result['fault'])
        self.assertIn('剩余 +44.00°', result['fault'])
        self.assertIn('四轮指令 -150,-150,150,150mm/s', result['fault'])
        self.assertLess(result['elapsed_s'], 2)
        board.brake.assert_not_called()

    def test_spin_writes_commands_and_braking_to_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'spin.csv'
            result, _ = self.simulate_spin(44, log_path=path)
            with path.open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertTrue(result['ok'])
        self.assertTrue(any(float(row['cmd1']) < 0 < float(row['cmd3']) for row in rows))
        self.assertEqual(rows[-1]['phase'], 'brake')
        self.assertAlmostEqual(float(rows[-1]['yaw']), result['turned_deg'])
        self.assertTrue(all(None not in row for row in rows))

    def test_latest_imu_angle_can_start_braking_at_stall_deadline(self):
        now = 100.0
        board, imu = Mock(), Mock()
        board.battery = 12.0
        board.counts = (0,) * 4
        board.brake.return_value = (0,) * 4
        imu.has_data.return_value = True
        imu.age.return_value = 0.0
        imu.rate = 0.0

        def poll():
            board.stamp = now
            imu.yaw = 44.0 if now >= 101.52 else 0.0

        def wait(seconds, stop_event=None):
            nonlocal now
            now = round(now + seconds, 9)

        board.poll.side_effect = poll
        with patch.object(arc.time, 'monotonic', side_effect=lambda: now), \
                patch.object(arc, 'wait_cancelable', side_effect=wait):
            result = arc.turn_with_radius(0, 44, board=board, imu=imu, max_speed=.15, verbose=False)
        self.assertTrue(result['ok'], result['summary'])
        board.brake.assert_called()

    def test_spin_plan_prints_actual_wheel_limit(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            arc.turn_with_radius(0, 44, max_speed=.15, dry_run=True)
        self.assertIn('轮速上限 150mm/s', output.getvalue())
        self.assertIn('收尾轮速 50mm/s', output.getvalue())
        self.assertIn('编码器 修正', output.getvalue())

    def test_final_angle_and_radius_errors_do_not_reject_a_completed_turn(self):
        now = 100.0
        polls = 0
        board, imu = Mock(), Mock()
        board.battery = 12.0
        imu.has_data.return_value = True
        imu.age.return_value = 0.0
        imu.rate = 0.0

        def poll():
            nonlocal polls
            polls += 1
            board.counts = (0,) * 4 if polls == 1 else (round(0.67 * arc.COUNTS_PER_METER),) * 4
            board.stamp = now
            imu.yaw = 0.0 if polls == 1 else 92.0

        def wait(seconds, stop_event=None):
            nonlocal now
            now += seconds

        board.poll.side_effect = poll
        with patch.object(arc.time, 'monotonic', side_effect=lambda: now), \
                patch.object(arc, 'wait_cancelable', side_effect=wait):
            result = arc.turn_with_radius(0.38, 87, board=board, imu=imu, verbose=False)
        self.assertTrue(result['ok'])
        self.assertIsNone(result['fault'])
        self.assertAlmostEqual(result['angle_error_deg'], -5.0)
        self.assertGreater(abs(result['radius_error_pct']), 5.0)
        board.brake.assert_called()
        board.stop.assert_called_once()
        board.release.assert_called_once()


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
