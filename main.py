"""按 tasks.txt 的选定段落顺序执行比赛动作。"""
import argparse
import faulthandler
import signal
import threading
from pathlib import Path

from tasks.runner import load_main, load_section, list_sections, execute
from base.control import check_cancel, cleanup, MotionCancelled

TASKS_FILE = Path(__file__).resolve().with_name("tasks.txt")


def run(base, arm, vision, *, tasks_path=None, stop_event=None, section="主线"):
    try:
        path = TASKS_FILE if tasks_path is None else tasks_path
        steps = load_main(path) if section == "主线" else load_section(path, section)
        check_cancel(stop_event)
        if section == "主线" or any(step.command in (
                "scan-qrcode", "detect-balls", "align", "vision-straight",
                "calibrate-ball-position", "calibrate-object-position") for step in steps):
            vision.start()
        for index, step in enumerate(steps, 1):
            args = " ".join(f"{value:g}" for value in step.args)
            print(f"[{section} {index}/{len(steps)}] {step.command} {args}".rstrip(), flush=True)
            execute(step, base, arm, vision, stop_event=stop_event)
    finally:
        # 底盘保持零速度闭环，机械臂保留最后目标；关闭连接时不释放电机。
        cleanup(("底盘停车", base.stop),
                ("底盘关闭", lambda: base.close(release_motors=False)), ("机械臂关闭", arm.close),
                ("视觉关闭", vision.close), raise_errors=False)


def main(args=None):
    parser = argparse.ArgumentParser(description="按任务表执行比赛小车动作", allow_abbrev=False)
    sections = parser.add_mutually_exclusive_group()
    try:
        names = list_sections(TASKS_FILE)
    except OSError as exc:
        parser.error(f"无法读取任务表：{exc}")
    for name in names:
        sections.add_argument(f"--{name}", dest="section", action="store_const",
                              const=name, help=f"执行 tasks.txt 的 [{name}] 段")
    parser.set_defaults(section="主线")
    parser.add_argument("--dry-run", action="store_true", help="只打印选定段落的动作，不连接设备")
    opts = parser.parse_args(args)
    stop_event = threading.Event()
    if opts.dry_run:
        from debug.dry_run import DryBase, DryArm, DryVision
        base, arm, vision = DryBase(), DryArm(), DryVision()
        print(f"干跑：按 tasks.txt 的 [{opts.section}] 打印动作，不连接设备、不发送运动指令。")
    else:
        from base.api import Base
        from arm.api import Arm
        from vision.api import Vision
        # 构造只保存配置；run() 在清理保护内按选定任务启动视觉后台。
        base = Base(stop_event=stop_event)
        arm = Arm(stop_event=stop_event)
        vision = Vision(stop_event=stop_event)

    def interrupt(signum, frame):
        stop_event.set()
        raise KeyboardInterrupt

    previous = {sig: signal.signal(sig, interrupt) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        run(base, arm, vision, stop_event=stop_event, section=opts.section)
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
