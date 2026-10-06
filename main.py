"""比赛主线：到扫码点、夹球、继续跑图。"""
import argparse
import signal
import threading

from tasks import route, scan, ball, pause
from base.control import cleanup, MotionCancelled


def run(base, arm, vision):
    try:
        arm.calibrate()
        pause(arm)
        route.to_qrcode(base)
        mission = scan.run(arm, vision)
        route.to_ball(base, vision)
        ball.run(arm, vision, mission["ball"])
        route.after_ball(base, vision)
        # 打靶、取物和放物的路线、观察姿态与动作填写后，再接到这里。
    finally:
        cleanup(("底盘停车", base.stop), ("机械臂停止", arm.cancel),
                ("底盘关闭", base.close), ("机械臂关闭", arm.close),
                ("视觉关闭", vision.close), raise_errors=False)


def main(args=None):
    parser = argparse.ArgumentParser(description="比赛小车主线（直接 Python 运行）")
    parser.add_argument("--dry-run", action="store_true", help="只打印实际路线及条件分支，不连接设备")
    opts = parser.parse_args(args)
    stop_event = threading.Event()
    if opts.dry_run:
        from debug.dry_run import DryBase, DryArm, DryVision
        base, arm, vision = DryBase(), DryArm(), DryVision()
        print("干跑：不连接设备、不发送运动指令；二维码示例不代表实际比赛任务。")
    else:
        from base.api import Base
        from arm.api import Arm
        from vision.api import Vision
        # 构造只保存配置，连接都在 run() 的清理保护内按需建立。
        base = Base(stop_event=stop_event)
        arm = Arm(stop_event=stop_event)
        vision = Vision(stop_event=stop_event)

    def interrupt(signum, frame):
        stop_event.set()
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run(base, arm, vision)
    except (KeyboardInterrupt, MotionCancelled):
        stop_event.set()
        print("比赛已取消，已尝试停车并关闭设备。")
        return 130
    except Exception as exc:
        stop_event.set()
        print(f"比赛停止：{exc}")
        return 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
