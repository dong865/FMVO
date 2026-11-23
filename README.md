# 3V1 学习基线：训练与 Crazyswarm2 实机部署

IQLD、IRL 和博弈论基线的训练及部署命令见 `ADDITIONAL_BASELINES.md`。
虚拟障碍物坐标、训练和实机配置见 `VIRTUAL_OBSTACLES.md`。
三维仿真界面及快捷键见 `PYBULLET_GUI.md`。
TensorBoard 训练与评估监控见 `TENSORBOARD.md`。
含动捕、延迟、PID 执行器和机间气流扰动的环境说明见 `ROBUST_SIMULATION.md`。
论文对齐后的 5 m 场景和公平性约束见 `PAPER_ALIGNED_SCENARIO.md`。


该目录把 `gym-pybullet-drones-main/project_3v1/mappo_open_baseline.py` 的核心结构迁移为独立模块：三架追捕机共享 Actor，集中式 Critic 用于训练，LSTM EPN 预测逃逸机轨迹。策略重新训练后，由 `real_flight.py` 通过与 `pid_swarm_test.py` 相同的 `/tf` 动捕输入和 `cmdVelocityWorld` 接口部署。

## 关键约定

- 固定 3 架追捕机 + 1 架逃逸机；`--roles` 前三项是追捕机，第四项是逃逸机。
- 训练和验证场地固定为 5 m × 5 m；追捕机上限 `0.45 m/s`，低于逃逸机 `0.50 m/s`。
- 训练和实机决策均为 50 Hz（`dt=0.02 s`）。策略输出归一化水平加速度，运行器积分、限幅后发送世界坐标速度。
- 高度由独立 PD 环保持；学习策略只处理平面 3V1 博弈。
- Crazyflie 参数取自 `crazyflie_sim/crazyflie_sim/backend/np.py`：质量 `0.034 kg`、惯量 `[16.571710, 16.655602, 29.261652]e-6 kg m²`、臂长 `0.046 m`、推力/力矩比 `0.006`、重力 `9.81 m/s²`。参数随检查点保存。
- 没有修改 `pid_swarm_test.py`、`motion_capture.yaml` 或总控机动捕配置。`mocap_bridge.py` 使用同一 `TFSubscriber`、刚体名映射和状态布局，只增加启动完整性及超时安全检查。

## 1. 独立开发电脑重新训练

```bash
cd /media/yudong/data/ros2_ws
export PYTHONPATH=$PWD/src/crazyswarm2/crazyflie_examples:$PYTHONPATH
python3 -m crazyflie_examples.three_v_one_baseline.smoke_test
python3 -m crazyflie_examples.three_v_one_baseline.train \
  --iterations 300 --rollout-steps 2048 \
  --output checkpoints/three_v_one
```

依赖 Python 3、NumPy 和 PyTorch；训练不依赖 ROS、无线电或动捕。检查点为 `checkpoints/three_v_one/mappo_epn_3v1_latest.pt`。默认规模是可运行基线，不代表一定收敛，应根据捕获率增加轮数并使用多个随机种子评估。

## 2. 复制到总控机并构建

复制整个 `three_v_one_baseline` 文件夹和 `.pt` 文件到总控机相同位置，然后执行：

```bash
cd /media/yudong/data/ros2_ws
colcon build --symlink-install --packages-select crazyflie_examples
source install/setup.bash
python3 -m crazyflie_examples.three_v_one_baseline.smoke_test
```

总控机需要安装与训练机兼容的 PyTorch。

## 3. 实机运行

先按原流程启动 Crazyswarm2 和 Nokov。确认 `crazyflies.yaml` 启用四架飞机，且 `/tf` 的 `child_frame_id` 与机名一致，再执行：

```bash
source /media/yudong/data/ros2_ws/install/setup.bash
python3 -m crazyflie_examples.three_v_one_baseline.real_flight \
  /absolute/path/to/mappo_epn_3v1_latest.pt \
  --roles cf1,cf2,cf3,cf4 \
  --duration 20 --max-speed 0.15 --max-accel 0.30
```

首次测试应拆桨或使用系留架，并采用低速度、短时长。程序会等待四个动捕刚体，随后以 50 Hz 发速度指令；动捕超时、Ctrl+C 或结束时会发零速度、退出低层流式模式、降落并解锁。若第四架只作静止目标，添加 `--evader-mode stationary`。

## 文件说明

- `config.py`：任务及 ROS2 Crazyflie 参数。
- `features.py`：训练与实机共用的 32 维障碍物感知观测。
- `environment.py`：与速度接口一致的二维训练环境。
- `model.py`：MAPPO、EPN 和版本化检查点。
- `train.py`：PPO、GAE、EPN 训练循环。
- `mocap_bridge.py`：Nokov `/tf` 状态读取。
- `real_flight.py`：50 Hz 实机推理和安全保护。
- `smoke_test.py`：无需 ROS 的端到端离线检查。
- `pybullet_gui.py`：不改变环境动力学的 PyBullet 三维可视化器。

