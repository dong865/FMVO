# 新增迁移基线使用说明

本目录现在支持四种统一 50 Hz 实机运行策略：`mappo_open`、`no_graph_iqld_maddpg`、`no_graph_irl_maddpg` 和 `game_theory`。后三种是本次新增迁移。

虚拟障碍物配置及 `--obstacles-file` 用法见 `VIRTUAL_OBSTACLES.md`。

四种基线的 TensorBoard 启动方式、日志目录和指标说明见 `TENSORBOARD.md`。

所有学习策略都必须在本目录的 50 Hz 环境中重新训练。原项目的检查点采用不同的观测维度、时间步长和文件布局，不能直接交给实机运行器。

## No-Graph IQLD-MADDPG

保留的算法结构包括：

- LSTM EPN 逃逸者轨迹预测；
- Transformer Situation Encoder；
- 三包围角色的 3×3 Sinkhorn 可微匹配；
- Path Control Network；
- 由 Game Theory 生成的专家状态—动作经验、专家回放预填充和行为克隆；
- 专家硬角色排列不进入训练，Sinkhorn 仍由 IQLD 端到端学习；
- 专家数据与自身交互回放共同进行双 Q、expectile V 和优势加权策略提取；
- 双 Q 分歧抑制的不确定性感知优势加权。

分配对象不是唯一逃逸机编号，而是逃逸机周围三个相隔 120 度的动态包围角色，因此分配矩阵由 IQLD 内部的可学习网络产生，并通过 Sinkhorn 归一化保持近似双随机。IQLD 在训练阶段读取 Game Theory 的状态—动作专家轨迹，但不读取其硬角色排列；训练完成后的推理和实机部署不再调用 Game Theory。IQLD 检查点已升级为 schema 4，旧模型必须重训。

```bash
cd /media/yudong/data/ros2_ws
export PYTHONPATH=$PWD/src/crazyswarm2/crazyflie_examples:$PYTHONPATH
python3 -m crazyflie_examples.three_v_one_baseline.train_iqld \
  --episodes 1000 --max-steps 1750 \
  --expert-episodes 20 --bc-steps 2000 \
  --output checkpoints/no_graph_iqld
```

输出检查点：

```text
checkpoints/no_graph_iqld/no_graph_iqld_maddpg_latest.pt
```

## No-Graph IRL-MADDPG

保留的算法结构包括：

- 参数共享的确定性 Actor；
- 三个集中式 Critic 和软更新目标网络；
- LSTM EPN；
- IBR 专家轨迹及行为克隆预训练；
- 全局状态—动作判别器、梯度惩罚和 GAIL 形式奖励；
- 环境奖励与 IRL 奖励混合。

```bash
python3 -m crazyflie_examples.three_v_one_baseline.train_irl \
  --episodes 1000 --max-steps 1750 \
  --expert-episodes 20 --bc-steps 2000 \
  --output checkpoints/no_graph_irl
```

输出检查点：

```text
checkpoints/no_graph_irl/no_graph_irl_maddpg_latest.pt
```

## Game Theory / Online Planning

该方法不需要训练，保留了逃逸者恒速预测、三机角色分配、120 度包围势函数和三轮顺序最佳响应。动作候选在 50 Hz 控制周期下重新缩放。

离线评估：

```bash
python3 -m crazyflie_examples.three_v_one_baseline.evaluate_game_theory \
  --episodes 100 --max-steps 1750
```

虚拟障碍物不依赖动捕刚体；同一 JSON 布局同时用于训练观测、博弈势函数、逃逸机控制和实机安全层。

## 实机运行

IQLD：

```bash
python3 -m crazyflie_examples.three_v_one_baseline.real_flight \
  checkpoints/no_graph_iqld/no_graph_iqld_maddpg_latest.pt \
  --baseline no_graph_iqld_maddpg \
  --roles cf1,cf2,cf3,cf4 --duration 20 \
  --max-speed 0.15 --max-accel 0.30
```

IRL：

```bash
python3 -m crazyflie_examples.three_v_one_baseline.real_flight \
  checkpoints/no_graph_irl/no_graph_irl_maddpg_latest.pt \
  --baseline no_graph_irl_maddpg \
  --roles cf1,cf2,cf3,cf4 --duration 20 \
  --max-speed 0.15 --max-accel 0.30
```

博弈论方法不需要检查点：

```bash
python3 -m crazyflie_examples.three_v_one_baseline.real_flight \
  --baseline game_theory \
  --roles cf1,cf2,cf3,cf4 --duration 20 \
  --max-speed 0.15 --max-accel 0.30
```

三种方法共用原 `pid_swarm_test.py` 的 `/tf` 动捕命名及状态布局、50 Hz `cmdVelocityWorld` 发送、高度 PD、速度/加速度限幅、地理围栏、紧急间距检查和安全降落流程。没有修改动捕系统配置。

