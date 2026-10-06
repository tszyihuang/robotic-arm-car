"""在控制台直接读取四电机与双舵机角度。"""
import argparse
import json
import math
import time

from arm.api import Arm
from base.control import cleanup

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
        start = time.monotonic()
        while not opts.duration or time.monotonic() - start < opts.duration:
            tick = time.monotonic()
            data = arm.angles()
            print(json.dumps(data, ensure_ascii=False) if opts.json else
                  time.strftime('%H:%M:%S') + ' | ' + format_angles(data), flush=True)
            time.sleep(max(0.0, 1 / opts.hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        pass
    finally:
        # 只读调试不校准、不使能、不发送运动或夹爪松开命令。
        cleanup(('机械臂连接', arm.close), raise_errors=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
