"""启动时重建软件零点，按 q1 q2 q3 q4 servo1 servo2 输出角度。"""
import argparse
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import time

if __name__ == '__main__' and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arm.api import Arm
from base.control import cleanup


def format_angles(data):
    values = []
    for addr in ('1', '2', '3', '4'):
        row = data.get('motors', {}).get(addr, {})
        values.append(row.get('joint_deg', float('nan')))
    for addr in ('1', '2'):
        row = data.get('servos', {}).get(addr, {})
        values.append(row.get('angle_deg', float('nan')))
    return ' '.join(f'{value:.2f}' for value in values)


def main(args=None):
    parser = argparse.ArgumentParser(description='直接读取四个机械臂电机与舵机 ID1、ID2 的角度')
    parser.add_argument('--hz', type=float, default=5.0)
    parser.add_argument('--duration', type=float, default=0.0, help='打印秒数；0 为持续运行')
    parser.add_argument('--json', action='store_true', help='每行输出原始 JSON 反馈')
    opts = parser.parse_args(args)
    if not math.isfinite(opts.hz) or not 0.2 <= opts.hz <= 20:
        parser.error('--hz 必须为 0.2..20')
    if not math.isfinite(opts.duration) or opts.duration < 0:
        parser.error('--duration 必须是有限非负数')
    arm = Arm()
    try:
        # 每次启动都以当前位置重建软件基准，沿用配置中的逻辑安装偏置。
        arm.config.encoder_zero_deg = None
        with contextlib.redirect_stdout(io.StringIO()):
            arm.calibrate()
        start = time.monotonic()
        while not opts.duration or time.monotonic() - start < opts.duration:
            tick = time.monotonic()
            data = arm.angles()
            print(json.dumps(data, ensure_ascii=False) if opts.json else format_angles(data), flush=True)
            time.sleep(max(0.0, 1 / opts.hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        # 软件归零后只读反馈；退出关闭会话。
        cleanup(('机械臂连接', arm.close), raise_errors=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
