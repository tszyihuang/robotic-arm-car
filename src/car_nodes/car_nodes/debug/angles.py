"""在控制台打印 arm 节点回传的四电机与双舵机角度。"""
import argparse
import json
import math
import signal
import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from .ros_client import DebugClient


def format_angles(data):
    rows = []
    for addr in ('1', '2', '3', '4'):
        row = data.get('motors', {}).get(addr, {})
        text = row.get('error', '无读数')
        if 'encoder_deg' in row:
            text = f"编码器 {row['encoder_deg']:8.2f}°"
            text += f" / 关节 {row['joint_deg']:8.2f}°" if 'joint_deg' in row else ' / 未校准'
        rows.append(f'电机{addr}: {text}')
    for addr in ('1', '2'):
        row = data.get('servos', {}).get(addr, {})
        text = f"{row['angle_deg']:8.2f}°" if 'angle_deg' in row else row.get('error', '无读数')
        rows.append(f'舵机{addr}: {text}')
    return ' | '.join(rows)


def main(args=None):
    parser = argparse.ArgumentParser(description='实时读取四个机械臂电机与舵机 ID1、ID2 的角度')
    parser.add_argument('--namespace', default='/car')
    parser.add_argument('--hz', type=float, default=5.0)
    parser.add_argument('--duration', type=float, default=0.0, help='打印秒数；0 为持续运行')
    parser.add_argument('--json', action='store_true', help='每行输出原始 JSON 反馈')
    opts = parser.parse_args(args)
    if not math.isfinite(opts.hz) or not 0.2 <= opts.hz <= 20:
        parser.error('--hz 必须为 0.2..20')
    if not math.isfinite(opts.duration) or opts.duration < 0:
        parser.error('--duration 必须是有限非负数')
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = DebugClient('angles_debug', opts.namespace, 'arm')
    last = [time.monotonic()]
    def feedback(message):
        try:
            data = json.loads(message.data)
            print(message.data if opts.json else time.strftime('%H:%M:%S') + ' | ' + format_angles(data), flush=True)
            last[0] = time.monotonic()
        except (ValueError, TypeError, KeyError) as exc:
            node.get_logger().error(f'无效角度反馈：{exc}')
    node.create_subscription(String, 'arm/angles', feedback, 1)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    worker = threading.Thread(target=executor.spin, daemon=True)
    worker.start()
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    started = False
    try:
        node.command(f'start-telemetry hz={opts.hz:g}')
        started = True
        start = time.monotonic()
        while node.context.ok() and (not opts.duration or time.monotonic() - start < opts.duration):
            time.sleep(0.1)
            if time.monotonic() - last[0] > max(3, 2 / opts.hz):
                print('角度反馈已超时，正在等待节点…', flush=True)
                last[0] = time.monotonic()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, TimeoutError) as exc:
        print(f'角度调试失败：{exc}', flush=True)
        return 2
    finally:
        node.cancel_active()
        if started:
            try:
                node.command('stop-telemetry', timeout=5)
            except (RuntimeError, TimeoutError) as exc:
                print(f'停止角度回传失败：{exc}', flush=True)
        executor.shutdown()
        worker.join(timeout=3)
        node.destroy_node()
        rclpy.shutdown()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0
