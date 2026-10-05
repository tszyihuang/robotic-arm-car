# car

按任务表运行的小车项目，使用 ROS 2 Jazzy。`plan.py` 是任务表入口，五个节点各放在一个文件夹中：

```text
car/
├── plan.py                       任务表命令行入口
├── tasks.txt                     默认：校准、直走、抬头扫码、返回
├── task1.txt                     机械臂抓取与投放
├── tasks.track.txt               备用跑图路线
├── src/
│   ├── car_nodes/
│   │   ├── car_nodes/
│   │   │   ├── planner/          plan.py：整表校验、状态、调度与结果
│   │   │   ├── base/             底盘 PID、直走、转弯、视觉对正
│   │   │   ├── arm/              四轴与夹爪驱动、连接和软件零点
│   │   │   ├── vision/           摄像头、二维码、边界和物体模型
│   │   │   ├── sensor/           IMU 采集、编码器反馈发布
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
python3 plan.py --抓左边的小球 --dry-run
python3 plan.py -f tasks.track.txt --dry-run
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
python3 plan.py --抓左边的小球
python3 plan.py -f task1.txt
```

启动节点不会自行执行任务表。底盘、机械臂和摄像头按需连接；传感器节点在实机模式下持续读取 IMU。默认视觉推理使用 CPU，可在适配本机 CUDA 的环境中将 `infer_device` 改为 `cuda:0`。扫码无需加载 YOLO；清单需要视觉移动时，调度节点先请求开启边界/物体推理并等待有效边界，两份模型共用摄像头。

Python 依赖见 `requirements.txt`。OpenCV、PyTorch 和 torchvision 使用本机 JetPack/CUDA 对应版本；ROS 节点使用系统 Python。查询全部语法：`python3 plan.py --help`。

## 任务与节点接口

每行一条任务，允许空行、`#` 注释和带引号的参数。整份清单通过语法及关节限位校验后才发送任何目标。

任务表可使用 `[主线]`、`[抓左边的小球]` 等标题分段，每段到下一个标题为止。`python3 plan.py` 默认只执行 `[主线]`，`python3 plan.py --抓左边的小球` 只执行对应分段；标题会自动成为同名参数，使用 `--help` 查看当前任务表中的全部分段。一次只能选择一个分段，空分段或不存在的分段会报错。`-f` 指定的任务表也支持分段；没有分段标题的旧任务表仍执行整表，`--command` 仍直接执行传入的指令。只校验、提交所选分段，因此需要机械臂校准的分段须自行包含 `arm-calibrate`。

| 任务 | 目标节点 |
| --- | --- |
| `straight 0.3 150`、`turn 90 0.3` | 底盘 |
| `calibrate-position`、`align`、`vision-straight 0.5` | 底盘 |
| `arm-calibrate`、`arm-move 6 0 160 1`、`arm-home` / `home` | 机械臂 |
| `gripper-open`、`gripper-close`、`arm-disable` | 机械臂 |
| `scan-qrcode`、`scan-qrcode timeout=30` | 视觉 |

距离单位 m、速度 mm/s、角度度。普通直走支持负距离，视觉直走仅接受正距离。机械臂 `arm-move Q1 Q2 Q3 Q4 [G]` 按 ID1–4 排列，`G` 可选 `open` / `close` / 0..1。四轴移动前必须有同一份清单里的 `arm-calibrate`；校准读取当前编码器作为软件零点，不产生归零运动。夹爪使用 ID2，张开 291°，闭合 243°。

扫码使用中央一半宽高的画面，放大两倍识别。默认等待 30 秒，可用任务参数 `timeout=` 调整。文本由视觉 Action 返回，存入调度节点 `qr_data`，并随整份任务结果的 `details_json` 返回。扫码超时或故障默认停止清单；`--keep-going` 可继续。任务表默认执行间隔 0.5 秒，使用 `--gap` 调整。备用跑图表补上了原表缺少的机械臂校准步骤。

| ROS 接口（默认命名空间 `/car`） | 用途 |
| --- | --- |
| `tasks/run`：`RunTasks` Action | 整表目标、步骤进度及整表结果 |
| `base/execute`、`arm/execute`、`vision/execute`：`ExecuteCommand` Action | 单步目标、进度、结果或失败原因 |
| `vision/boundary`、`vision/objects` | 边界几何与物体检测 |
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

## 验证与迁移范围

```bash
bash scripts/test.sh -q
```

测试覆盖任务路由、整表拒绝、机械臂仿真、实际二维码解码、反馈过期、碰撞锁存、直走/视觉控制和真实 ROS Action 的取消与清理。测试不驱动实机；实机到位精度和目标 GPU 推理仍需硬件验证。

保留任务表需要的串口协议、关节运动、底盘控制、二维码和运行模型；移除了旧 Python 直控入口与兼容 API、网页预览、录制、本机 HTTP/视觉子进程、键盘与调试工具、机械臂笛卡尔控制、模型训练产物、历史文档和 Git 元数据。原 `car_ylq` 保留在原目录。
