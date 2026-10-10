"""按任务表顺序行驶，在任务点结合扫码及现场排列调用对应动作段。"""
import argparse
import faulthandler
import signal
import threading
from pathlib import Path

from tasks.runner import load_main, load_section, load_branches, list_sections, execute
from tasks.mission import MissionState, TASK_SECTIONS
from base.control import check_cancel, cleanup, MotionCancelled

TASKS_FILE = Path(__file__).resolve().with_name("tasks.txt")


def run(base, arm, vision, *, tasks_path=None, stop_event=None, section="主线"):
    try:
        path = TASKS_FILE if tasks_path is None else tasks_path
        steps = load_main(path) if section == "主线" else load_section(path, section)
        branches = load_branches(path, steps, section=section)
        mission = MissionState()
        check_cancel(stop_event)
        if section == "主线" or any(step.command in (
                "scan-qrcode", "detect-balls", "detect-targets", "align", "vision-straight",
                "calibrate-ball-position", "calibrate-object-position", *TASK_SECTIONS)
                for plan in (steps, *branches.values()) for step in plan):
            vision.start()

        def run_steps(current_steps, current_section, *, called=False, target_color=None):
            for index, step in enumerate(current_steps, 1):
                check_cancel(stop_event)
                args = " ".join(f"{value:g}" for value in step.args)
                print(f"[{current_section} {index}/{len(current_steps)}] {step.command} {args}".rstrip(), flush=True)
                if called and step.command == "arm-calibrate" and getattr(arm, "calibrated", False) is True:
                    print("  机械臂沿用主线已建立的软件基准。", flush=True)
                    continue
                if step.command in TASK_SECTIONS:
                    selected = mission.select_section(step.command, vision, stop_event=stop_event)
                    color = mission.goals["target"] if step.command == "打靶任务" else target_color
                    if selected is None:
                        for name in TASK_SECTIONS[step.command].values():
                            print(f"  [条件分支预览：{name}；实机按现场排列选择一段]", flush=True)
                            run_steps(branches[name], name, called=True, target_color=color)
                    else:
                        run_steps(branches[selected], selected, called=True, target_color=color)
                else:
                    result = execute(step, base, arm, vision, stop_event=stop_event, target_color=target_color)
                    mission.record(step.command, result)

        run_steps(steps, section)
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
