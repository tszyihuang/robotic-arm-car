"""任务表的本地校验和 ROS 客户端；中断后等待子动作停车。"""
import argparse
import math
import signal
from pathlib import Path
import sys
import time

from .tasks import HELP, default_task_file, prepare_tasks, select_task_section, task_sections


def wait(node, future, timeout):
    import rclpy
    deadline = time.monotonic() + timeout
    while rclpy.ok() and not future.done() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not future.done():
        raise TimeoutError('等待服务/任务超时')
    return future.result()


def main(args=None):
    argv = list(sys.argv[1:] if args is None else args)
    # ROS 参数只从 --ros-args 开始，避免将写错的分段名交给 ROS。
    ros_start = argv.index('--ros-args') if '--ros-args' in argv else len(argv)
    cli_args, ros_args = argv[:ros_start], argv[ros_start:]
    parser = argparse.ArgumentParser(description="校验任务表，并交给 plan 节点执行",
                                     epilog='按 [分段名] 选择任务，例如 --主线、--抓左边的小球；默认执行主线。\n\n' + HELP,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     add_help=False, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group()
    source.add_argument('-f', '--file', type=Path,
                        help='任务清单路径（相对当前目录），默认使用根目录 tasks.txt')
    source.add_argument('--command', help='任务文本，支持多行')
    source.add_argument('--stop', action='store_true', help='锁存全局停止')
    source.add_argument('--reset-stop', action='store_true', help='复位底盘、机械臂、视觉和调度节点的停止锁存')
    parser.add_argument('--list', action='store_true', help='显示并校验清单，不连接 ROS')
    parser.add_argument('--dry-run', action='store_true', help='本地校验清单，不连接 ROS 或硬件')
    parser.add_argument('--ros-dry-run', action='store_true', help='通过 ROS 校验清单，不执行硬件动作')
    parser.add_argument('--keep-going', action='store_true')
    parser.add_argument('--gap', type=float, default=0.5)
    parser.add_argument('--timeout', type=float, default=600.0)
    parser.add_argument('--namespace', default='/car')
    # 先确定任务表，再将其中的标题注册为互斥的命令行选项。
    opts, _ = parser.parse_known_args(cli_args)
    text = None
    sections = {}
    source_error = None
    if not opts.stop and not opts.reset_stop:
        try:
            text = (opts.command if opts.command is not None else
                    (opts.file or default_task_file()).read_text(encoding='utf-8'))
            if opts.command is None:
                sections = task_sections(text)
        except (OSError, ValueError) as exc:
            source_error = exc
    selection = parser.add_mutually_exclusive_group()
    for name in dict.fromkeys(('主线', *sections)):
        try:
            selection.add_argument('--' + name, dest='section', action='store_const', const=name,
                                   help=f'只执行 [{name}] 分段' + ('（默认）' if name == '主线' else ''))
        except argparse.ArgumentError:
            parser.error(f'任务分段 [{name}] 与已有命令行选项冲突，请修改分段名称')
    parser.add_argument('-h', '--help', action='help', help='显示帮助并退出')
    opts = parser.parse_args(cli_args)
    if opts.section is not None and (opts.command is not None or opts.stop or opts.reset_stop):
        parser.error('分段选择不能与 --command、--stop 或 --reset-stop 同时使用')
    if source_error is not None:
        parser.error(str(source_error))
    if not math.isfinite(opts.gap) or opts.gap < 0:
        parser.error('--gap 必须是有限非负数')
    if not math.isfinite(opts.timeout) or opts.timeout <= 0:
        parser.error('--timeout 必须是有限正数')
    if text is not None:
        try:
            if opts.command is None:
                text = select_task_section(text, opts.section)
            steps = prepare_tasks(text)
        except ValueError as exc:
            parser.error(str(exc))
    if text is not None and (opts.list or opts.dry_run):
        if ros_args:
            parser.error('本地校验不接受额外参数：' + ' '.join(ros_args))
        for index, step in enumerate(steps, 1):
            print(f"[{index}/{len(steps)}] {step.target}: {step.text}")
        print(f"清单有效，共 {len(steps)} 步")
        return 0
    try:
        import rclpy
        from rclpy.action import ActionClient
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from std_msgs.msg import Bool
        from std_srvs.srv import Trigger
        from car_interfaces.action import RunTasks
    except ImportError as exc:
        parser.error(f"ROS 环境未加载或尚未构建，请先运行 bash scripts/build.sh 并 source install/setup.bash：{exc}")
    # Keep SIGINT under the client's control so it can issue ROS cancellation.
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=ros_args, signal_handler_options=SignalHandlerOptions.NO)
    previous_int = signal.getsignal(signal.SIGINT)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, terminate)
    previous_term = signal.signal(signal.SIGTERM, terminate)
    node = Node('task_client', namespace=opts.namespace)
    child = None
    accepted = None
    exit_code = 0
    try:
        if opts.stop:
            publisher = node.create_publisher(Bool, 'emergency_stop', QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            for _ in range(10):
                publisher.publish(Bool(data=True))
                rclpy.spin_once(node, timeout_sec=0.1)
            print('全局停止已发送；复位前请等待活动动作清理完成')
        elif opts.reset_stop:
            for target in ('base', 'arm', 'vision', 'tasks'):
                service = node.create_client(Trigger, target + '/reset_stop')
                if not service.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError(f'{target}/reset_stop 未启动')
                result = wait(node, service.call_async(Trigger.Request()), 5.0)
                print(f'{target}: {result.message}')
                if not result.success:
                    exit_code = 2
        else:
            client = ActionClient(node, RunTasks, 'tasks/run')
            if not client.wait_for_server(timeout_sec=10.0):
                raise RuntimeError('tasks/run 未启动，请先 ros2 launch')
            goal = RunTasks.Goal(tasks=text, dry_run=opts.ros_dry_run,
                                keep_going=opts.keep_going, gap_s=opts.gap)

            def feedback(message):
                f = message.feedback
                print(f'[{f.step_index}/{f.total_steps}] {f.phase}: {f.command}')

            accepted = client.send_goal_async(goal, feedback_callback=feedback)
            try:
                child = wait(node, accepted, 10.0)
            except KeyboardInterrupt:
                # Goal may already be accepted remotely; consume and cancel it.
                child = wait(node, accepted, 10.0)
                raise
            if not child.accepted:
                raise RuntimeError('任务被拒绝：检查清单、活动任务和停止状态')
            result = wait(node, child.get_result_async(), opts.timeout).result
            print(result.message)
            print(result.details_json)
            exit_code = 0 if result.success else 2
    except (KeyboardInterrupt, TimeoutError) as exc:
        if child is None and accepted is not None:
            try:
                child = wait(node, accepted, 10.0)
            except TimeoutError:
                pass
        if child is not None and child.accepted:
            child.cancel_goal_async()
            try:
                result = wait(node, child.get_result_async(), 15.0).result
                print(result.message)
            except TimeoutError:
                publisher = node.create_publisher(Bool, 'emergency_stop', QoSProfile(
                    depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
                for _ in range(5):
                    publisher.publish(Bool(data=True))
                    rclpy.spin_once(node, timeout_sec=0.1)
                print('取消确认超时，已发送全局停止')
        elif accepted is not None and child is None:
            publisher = node.create_publisher(Bool, 'emergency_stop', QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
            for _ in range(5):
                publisher.publish(Bool(data=True))
                rclpy.spin_once(node, timeout_sec=0.1)
            print('任务接收状态未知，已发送全局停止')
        print(str(exc) or '任务已取消')
        exit_code = 1
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'失败：{exc}')
        exit_code = 2
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTERM, previous_term)
    return exit_code
