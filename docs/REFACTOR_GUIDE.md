# 比赛小车重构指导

把现在的 ROS 项目改成普通 Python 文件。项目只用于这次比赛，优先让人看懂、现场好改、能直接运行。

不做 Python 安装包，不要 `pyproject.toml`、`setup.py`、包注册、插件、RPC、服务容器和通用调度框架。直接在项目目录运行 `python3 main.py`。文件夹按功能分，普通 `import` 调用，必要时放空的 `__init__.py`。

## 1. 文件怎么放

```text
car/
├── main.py                 比赛主线：到哪个点，做什么任务
├── config.py               端口、速度、PID、机械臂参数、模型路径
├── requirements.txt        简单依赖列表
├── README.md               怎么运行、调参、调试
├── tasks/
│   ├── route.py            跑图，每段路线一个函数
│   ├── scan.py             抬头扫码、解析二维码、回位
│   ├── ball.py             夹球，左中右动作直接写在这里
│   ├── target.py           打靶
│   └── delivery.py         取物、放物
├── base/
│   ├── api.py              底盘接口，调用以下已有算法
│   ├── motor.py            电机串口和编码器读取
│   ├── straight_pid.py     普通直走
│   ├── arc_turn.py         圆弧和原地转弯
│   ├── calibrate_position.py  倒车靠坎
│   ├── vision_align.py     视觉对正
│   └── vision_straight.py  视觉直走
├── arm/
│   ├── api.py              保留机械臂会话：校准、移动、回位、夹爪
│   ├── controller.py      关节运动、等待到位
│   ├── motor.py            四轴电机协议
│   └── servo.py            舵机协议
├── vision/
│   ├── api.py              管理摄像头、模型，提供扫码和目标观察
│   ├── camera.py           采集画面
│   ├── qrcode.py           二维码识别
│   ├── yolo_boundary.py    跑道边界
│   ├── yolo_objects.py     球、物体识别
│   ├── targets.py          判断目标在左、中、右
│   └── models/             保留两套权重及各自 args.yaml
├── sensor/
│   └── imu.py              IMU 串口、航向角、加速度
├── debug/                  摄像头网页、机械臂角度调试
└── tests/                  复用有价值的现有测试
```

这不是要求凑齐文件。已有的限位校验、舵机端口识别等代码可以继续放在所属文件夹；短小的辅助文件可以合并。不要把同一功能拆成很多层包装。统一配置放 `config.py`，可以按功能用几个字典，不必声明一堆配置类。

## 2. 主线和任务怎么写

`main.py` 直接调用任务函数。下面是当前已有流程的组织示意，具体距离和动作从当前代码、任务表搬过来：

```python
from tasks import route, scan, ball
from base.api import Base
from arm.api import Arm
from vision.api import Vision


def main():
    base = Base()
    arm = Arm()
    vision = Vision()
    try:
        arm.calibrate()
        route.to_qrcode(base)
        mission = scan.run(arm, vision)
        route.to_ball(base, vision)
        ball.run(arm, vision, mission["ball"])
        route.after_ball(base, vision)
        # 打靶、运物体的路线和动作补齐后，再接到这里。
    finally:
        # cleanup 要逐个尝试停止、关闭，不能前一个出错就跳过后面。
        cleanup(base, arm, vision)


if __name__ == "__main__":
    main()
```

以上是示意，`cleanup` 需要实现。设备构造只保存配置，连接放在受清理保护的流程内按需建立。主线可以顺序读懂，不需要另做状态机、任务注册器或调用链框架。

`tasks/route.py` 直接写路线：

```python
def to_qrcode(base):
    base.straight(0.3)


def to_ball(base, vision):
    base.turn(87, 0.38)
    base.turn(-45, 0.52)
    base.turn(-42, 0)
    # 按现有主线顺序继续搬，保留全部参数。
```

`scan.run()` 包含原来的抬头关节动作、扫码、回初始姿态，返回简单字典，例如 `{"ball": "green", "target": "red", "object": "cylinder"}`。

`ball.run()` 先移动到现有观察姿态，调用视觉得到左、中、右，再用普通 `if/elif` 调对应函数。每个位置的关节角和夹爪动作写在该任务文件里，方便现场修改。`target.py`、`delivery.py` 用同样方式。

把 `tasks.txt` 中的动作逐条搬到这些 Python 函数，顺序和数值不变。它只作为迁移时的对照，确认一致后不必继续维护任务表解析器，也不必兼容原来的全部 CLI。保留一个主入口和必要的调试入口即可。

## 3. 模块之间怎么调用

任务调用 `base`、`arm`、`vision` 的接口，接口再调用已有算法和驱动。底盘、机械臂和视觉不要反过来导入 `main.py` 或任务文件。

- 底盘提供 `straight()`、`turn()`、`align()`、`vision_straight()`、`calibrate_position()`。
- 机械臂提供 `calibrate()`、`move_joints()`、`home()`、夹爪开合、角度读取。
- 视觉提供 `scan_qrcode()`、`observe_target()` 和跑道边界反馈。
- 设备各建一次，任务通过参数共用，不在每个任务里重复开串口、开摄像头。
- 用现成的类保存设备连接和零点就够了，任务用普通函数；返回值用数字、字符串、元组或字典。

主流程按顺序执行。IMU、编码器、摄像头继续用后台线程采集；视觉行驶需要时后台推理边界。反馈直接保存在对象里，加必要的锁，算法读取最新数据。删除 ROS 消息转发，不再让编码器绕一圈返回底盘。

## 4. 现有行为要保留

优先搬代码，不重写控制算法和协议。

- 当前主线 25 步，中间小球 7 步，动作和单位原样保留：距离 m、速度 mm/s、角度度。
- 编码器比例 `0.00016029 m/count`；普通直走和视觉控制轮距 `190 mm`；圆弧轮距 `0.41 m`。先保留不同值，不重新标定。
- 机械臂校准读取当前位置作为软件基准，不产生归零运动；当前姿态对应 `0 0 160 24`，`home` 回本次初始姿态。
- 夹爪 ID2，张开 291°、闭合 243°；保留夹持时不等待闭合角到位的行为，退出不自动松开物体。
- 二维码三位分别指定球颜色、靶颜色、物体形状；复用当前颜色和形状映射。
- 球和物体用已有模型，靶用现有 HSV 识别。目标必须明确、稳定才选择位置，不猜。
- 旧反馈不能当新反馈，编码器增量不能重复累计；丢失反馈或视觉时保留现有停车行为。
- 保留 Ctrl+C 取消、读写超时、异常停车和资源关闭。清理失败要打印，不覆盖原始错误。现有停止代码能用就继续用。

当前只有中间球动作已填写，另外 8 个位置子任务为空。缺失的动作直接抛出 `NotImplementedError` 并说明缺什么，不编造关节角，也不假装完成。打靶和运物体先有清楚的函数位置，等用户补动作后再启用。

## 5. 调试与迁移顺序

摄像头网页和角度显示保留，但直接使用设备对象。独立调试时不要同时运行比赛主线，避免抢串口和摄像头。调试脚本可以通过 `python3 -m debug.camera_web` 这类命令从项目根目录运行，不做额外的入口包装层。

按这个顺序重构：

1. 把算法、协议、模型和机械臂会话搬到对应文件夹，去掉 ROS 导入。
2. 接好直接读取的反馈、设备接口及关闭逻辑。
3. 把现有任务表搬成路线函数和任务函数，写出直观的 `main.py`。
4. 迁移调试工具，验证后清掉旧 ROS 节点、接口、launch 和构建配置。
5. README 简单写清运行命令、改路线的位置、改参数的位置、未完成的动作。

用 `Path(__file__)` 等方式定位资源，不把旧目录路径写死。保留模型、舵机绑定信息、许可证。依赖列到 `requirements.txt`，沿用本机可用的 OpenCV、PyTorch 和 CUDA 环境。

## 6. 怎么确认改好了

提供 `python3 main.py --dry-run`，打印计划动作和参数，不开设备、不发运动命令。最简单可以用几个只打印动作的模拟设备，共用实际路线函数，不另抄一份主线。

复用当前有意义的算法、反馈、扫码、机械臂测试；把 ROS 通信测试换成直接调用测试。重点确认主线和分支顺序、参数一致，缺失动作会报错，Ctrl+C 和设备异常能结束动作并清理。重构验证不自动驱动实车。

完成后项目直接 `python3 main.py` 运行，不需要 ROS、colcon、安装自己的包或加载环境脚本。最终说明改了哪些文件、怎么启动、哪些动作还没填写即可。

判断标准：想改路线打开 `tasks/route.py`，想改夹球打开 `tasks/ball.py`，想调参数打开 `config.py`，看 `main.py` 就知道整趟比赛怎么跑。
