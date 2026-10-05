# car

按任务表运行的小车项目，使用 ROS 2 Jazzy。`plan.py` 是任务表入口，五个节点各放在一个文件夹中：

```text
car/
├── plan.py                       任务表命令行入口
├── tasks.txt                     主线与位置子任务
├── debug/
│   ├── camera_web.py             局域网画面、截图、模型开关和扫码
│   └── angles.py                 四个机械臂电机和两个舵机角度
├── src/
│   ├── car_nodes/
│   │   ├── car_nodes/
│   │   │   ├── planner/          plan.py：整表校验、状态、调度与结果
│   │   │   ├── base/             底盘 PID、直走、转弯、视觉对正
│   │   │   ├── arm/              四轴与夹爪驱动、连接和软件零点
│   │   │   ├── vision/           摄像头、二维码、边界和物体模型
│   │   │   ├── sensor/           IMU 采集、编码器反馈发布
│   │   │   ├── debug/            两个调试客户端的实现
│   │   │   └── common/           Action 生命周期、取消与停止
│   │   ├── config/car.yaml       实机端口与运行参数
│   │   └── launch/car.launch.py  启动五个节点
│   └── car_interfaces/           节点使用的消息和 Action
├── scripts/                      构建、验证
└── tests/                        任务与节点通信回归
```

`car_nodes/tasks.txt` 是指向根目录任务表的相对符号链接，打包时安装同一份任务表。源码运行和 symlink 安装每次读取根目录任务表；普通安装读取安装包中的任务表。

## 运行

先校验清单，不需要 ROS，不打开设备：

```bash
cd /home/tszyi/Desktop/car
python3 plan.py --list
python3 plan.py --dry-run
python3 plan.py --主线 --dry-run
# 也可以用 --分段名 单独调试一个已填写的分段。
```

构建并启动节点；默认干跑：

```bash
bash scripts/build.sh
source install/setup.bash
ros2 launch car_nodes car.launch.py
```

另一个终端加载环境后提交任务：

```bash
cd /home/tszyi/Desktop/car
source install/setup.bash
python3 plan.py --ros-dry-run
# 或：ros2 run car_nodes plan --ros-dry-run
```

实机运行前，在 `src/car_nodes/config/car.yaml` 填写 `base_node.motor_port`、`sensor_node.imu_port`、`arm_node.arm_port`、`arm_node.servo_port` 和 `vision_node.device`。`sensor_node.motor_port` 与底盘端口一致，用于自动识别时排除驱动板。端口留空沿用自动识别；机械臂留空沿用驱动中的稳定设备路径。固定端口可避免多个节点在自动识别时探测同一个设备。

```bash
source install/setup.bash
ros2 launch car_nodes car.launch.py dry_run:=false
# 另一个已加载环境的终端：
python3 plan.py
python3 plan.py --主线
```

启动节点不会自行执行任务表。底盘、机械臂和摄像头按需连接；传感器节点在实机模式下持续读取 IMU。默认视觉推理使用 CPU，可在适配本机 CUDA 的环境中将 `infer_device` 改为 `cuda:0`。扫码无需加载 YOLO；清单需要视觉移动时，调度节点先请求开启边界/物体推理并等待有效边界，两份模型共用摄像头。

Python 依赖见 `requirements.txt`。OpenCV、PyTorch 和 torchvision 使用本机 JetPack/CUDA 对应版本；ROS 节点使用系统 Python。查询全部语法：`python3 plan.py --help`。

## 任务与节点接口

每行一条任务，允许空行、`#` 注释和带引号的参数。整份清单通过语法及关节限位校验后才发送任何目标。

任务表以 `[主线]`、`[抓左边的小球]` 等标题分段，每段到下一个标题为止。默认或 `--主线` 按主线顺序执行；遇到下表的任务标记，使用本趟任务之前的扫码数据请求视觉观察，执行所选子任务，再回到主线下一条指令。子任务定义不会因写在文件下方而顺序执行。运行前校验主线及可能执行的子任务；子任务继承主线已建立的机械臂校准。无标题的旧清单仍按行执行。

| 主线任务标记（同义名称） | 二维码位 | 值的含义 | 选择的子任务 |
| --- | --- | --- | --- |
| `抓球任务`（`排爆任务`） | 第 1 位 | 1 红、2 绿、3 蓝 | `[抓左边的小球]` / `[抓中间的小球]` / `[抓右边的小球]` |
| `打靶任务`（`反恐任务`） | 第 2 位 | 1 红、2 绿、3 蓝 | `[打左边的靶]` / `[打中间的靶]` / `[打右边的靶]` |
| `抓物体任务`（`救援任务`） | 第 3 位 | 1 圆柱、2 圆锥、3 腰鼓 | `[抓左边的物体]` / `[抓中间的物体]` / `[抓右边的物体]` |

二维码必须为三位 `1..3` 的组合。例如 `211` 选择绿色小球、红色靶、圆柱形救援物体。画面从左到右直接对应左边、中间、右边。小球及形状使用现有 YOLO；现有权重没有靶类别，靶以 HSV 色块轮廓识别，实际照明和背景应通过网页观察验证，可调 `target_min_area_ratio`。观察从任务标记后的新画面开始，必须看到三个候选、目标匹配唯一，并连续 `observe_stable_frames` 帧位置与类别顺序稳定；默认 3 帧、10 秒超时。无法确认或选中空子任务时默认停止并返回原因；`--keep-going` 才继续主线。目前任务表只填写了中间小球子任务，其余分段仍需填写实际动作。干跑只校验可执行分支，不从虚拟画面猜选子任务。

标题仍自动成为 `--分段名` 参数，用于单独调试一个分段。单独执行子任务时不继承主线校准，需在该分段自行包含 `arm-calibrate`；使用 `--help` 查看分段列表。`-f` 支持其他任务表，`--command` 可提交多行指令或完整分段表。

| 任务 | 目标节点 |
| --- | --- |
| `straight 0.3 150`、`turn 90 0.3` | 底盘 |
| `calibrate-position`、`align`、`vision-straight 0.5` | 底盘 |
| `arm-calibrate`、`arm-move 6 0 160 1`、`arm-home` / `home` | 机械臂 |
| `gripper-open`、`gripper-close`、`arm-disable` | 机械臂 |
| `scan-qrcode`、`scan-qrcode timeout=30` | 视觉 |

距离单位 m、速度 mm/s、角度度。普通直走支持负距离，视觉直走仅接受正距离。机械臂 `arm-move Q1 Q2 Q3 Q4 [G]` 按 ID1–4 排列，`G` 可选 `open` / `close` / 0..1。四轴移动前必须有同一份清单里的 `arm-calibrate`；校准读取当前编码器作为软件零点，不产生归零运动。夹爪使用 ID2，张开 291°，闭合 243°。

扫码使用中央一半宽高的画面，放大两倍识别。默认等待 30 秒，可用任务参数 `timeout=` 调整。文本由视觉 Action 返回，存入调度节点 `qr_data`，并随整份任务结果的 `details_json` 返回。扫码超时或故障默认停止清单；`--keep-going` 可继续。任务表默认执行间隔 0.5 秒，使用 `--gap` 调整。

| ROS 接口（默认命名空间 `/car`） | 用途 |
| --- | --- |
| `tasks/run`：`RunTasks` Action | 整表目标、步骤进度及整表结果 |
| `base/execute`、`arm/execute`、`vision/execute`：`ExecuteCommand` Action | 单步目标、进度、结果或失败原因 |
| `vision/boundary`、`vision/objects` | 边界几何与物体检测 |
| `vision/image/compressed`、`vision/debug_status` | 网页调试启用后的 JPEG 画面和模型状态 |
| `arm/angles`：`std_msgs/String` | 调试启用后的角度 JSON，含逐设备错误信息 |
| `sensors/imu`、`sensors/encoders` | 传感器反馈 |
| `base/raw_encoders` | 电机串口返回的编码器数据，交给传感器节点发布 |
| `emergency_stop`、各控制节点 `stop` / `reset_stop` | 停止锁存与复位 |

编码器与电机命令共用驱动板串口，因此底盘独占该串口，读取原始编码器后交给传感器节点转发；底盘控制器只从传感器消息取反馈。转发保留采集时间，重复消息不会延长数据有效期。IMU 串口归传感器节点，摄像头归视觉节点，机械臂与夹爪串口归机械臂节点。

取消任务会先取消正在运行的子动作，等待停车及清理后再返回。可用 `Ctrl+C` 取消当前任务，或：

```bash
python3 plan.py --stop
python3 plan.py --reset-stop
```

普通直走及视觉控制的有效轮距沿用 190 mm，圆弧转弯沿用 0.41 m，编码器比例沿用 0.00016029 m/count。本次重构没有重新标定运动参数。

## 两个调试程序

先构建、加载 `install/setup.bash` 并启动需要的节点。若五个节点已经启动，直接运行调试程序即可；只调试摄像头时可单独启动视觉节点：

```bash
cd /home/tszyi/Desktop/car
bash scripts/build.sh
source install/setup.bash
ros2 run car_nodes vision_node --ros-args -r __ns:=/car -p dry_run:=false
```

另一个已加载环境的终端运行：

```bash
python3 debug/camera_web.py --port 8080
# 或 ros2 run car_nodes camera_debug --port 8080
```

打开程序打印的 `http://小车局域网IP:8080`。网页实时显示画面和已启用模型的标注，支持截图下载、边界 YOLO / 物体 YOLO 独立开关及扫码。关掉两个模型后摄像头仍持续采集；首次启用模型需要等待加载。扫码结果在页面显示，不替代任务表自己的 `scan-qrcode`。`--host` 默认 `0.0.0.0`，`--namespace` 默认 `/car`；网页关闭后节点保持当前模型设置。干跑节点不会生成真实画面。

角度调试依赖机械臂节点；若未启动，可在另一个终端单独启动：

```bash
ros2 run car_nodes arm_node --ros-args -r __ns:=/car -p dry_run:=false
```

然后打印反馈：

```bash
python3 debug/angles.py --hz 5
# 只打印 10 秒或以 JSON 输出：
python3 debug/angles.py --hz 5 --duration 10 --json
# 或 ros2 run car_nodes angles_debug --hz 5
```

反馈读取机械臂电机 ID1–4 的编码器角度，以及同一 TTL 总线上的舵机 ID1、ID2 实测角度，全部以度显示；机械臂完成校准后还显示任务使用的关节角度。读取本身不建立零点、不使能、不发送运动指令。通信失败按设备打印原因，不以 0° 代替。连接归机械臂节点，任务与调试共享每条总线的通信锁；退出程序后停止角度轮询。默认 5 Hz，实际反馈速率受串口应答耗时影响；节点干跑时明确打印“无硬件读数”。

## 验证与迁移范围

```bash
bash scripts/test.sh -q
```

测试覆盖任务路由、整表拒绝、机械臂仿真、实际二维码解码、反馈过期、碰撞锁存、直走/视觉控制和真实 ROS Action 的取消与清理。测试不驱动实机；实机到位精度和目标 GPU 推理仍需硬件验证。

保留任务表需要的串口协议、关节运动、底盘控制、二维码和运行模型。两个调试客户端按需启用；原 `car_ylq` 保留在原目录。



# 运动控制所需的第三方串口库。
pyserial>=3.5

# 视觉处理的通用依赖。
numpy>=1.26,<3
PyYAML>=6,<7
ultralytics>=8.4.164,<9

# OpenCV、PyTorch、torchvision 请使用与本机 JetPack/CUDA 匹配的安装版本。
# 保留已可用的系统 OpenCV；这些库不是其他项目中的自定义程序。
