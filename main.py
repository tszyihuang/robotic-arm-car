"""按 tasks.txt 的 [主线] 顺序执行比赛动作。"""
import argparse
import faulthandler
import signal
import threading
from pathlib import Path

from tasks.runner import load_main, execute
from base.control import check_cancel, cleanup, MotionCancelled

TASKS_FILE = Path(__file__).resolve().with_name("tasks.txt")


def run(base, arm, vision, *, tasks_path=None, stop_event=None):
    try:
        steps = load_main(TASKS_FILE if tasks_path is None else tasks_path)
        check_cancel(stop_event)
        vision.start()
        for index, step in enumerate(steps, 1):
            args = " ".join(f"{value:g}" for value in step.args)
            print(f"[主线 {index}/{len(steps)}] {step.command} {args}".rstrip(), flush=True)
            execute(step, base, arm, vision, stop_event=stop_event)
    finally:
        cleanup(("底盘停车", base.stop), ("机械臂停止", arm.cancel),
                ("底盘关闭", base.close), ("机械臂关闭", arm.close),
                ("视觉关闭", vision.close), raise_errors=False)


def main(args=None):
    parser = argparse.ArgumentParser(description="比赛小车主线（直接 Python 运行）")
    parser.add_argument("--dry-run", action="store_true", help="只打印 tasks.txt 的 [主线]，不连接设备")
    opts = parser.parse_args(args)
    stop_event = threading.Event()
    if opts.dry_run:
        from debug.dry_run import DryBase, DryArm, DryVision
        base, arm, vision = DryBase(), DryArm(), DryVision()
        print("干跑：按 tasks.txt 的 [主线] 打印动作，不连接设备、不发送运动指令。")
    else:
        from base.api import Base
        from arm.api import Arm
        from vision.api import Vision
        # 构造只保存配置；run() 在清理保护内启动常驻视觉后台。
        base = Base(stop_event=stop_event)
        arm = Arm(stop_event=stop_event)
        vision = Vision(stop_event=stop_event)

    def interrupt(signum, frame):
        stop_event.set()
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run(base, arm, vision, stop_event=stop_event)
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
    # 原生库崩溃无法进入 except/finally；输出所有线程的 Python 栈来定位。
    faulthandler.enable(all_threads=True)
    raise SystemExit(main())
