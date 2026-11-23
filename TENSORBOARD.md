# TensorBoard 训练与评估监控

四种基线默认写入 `runs/three_v_one/`。未指定 `--run-name` 时，目录名由基线、启动时间和随机种子自动组成，因此不同实验不会互相覆盖。

## 启动训练或评估

```bash
cd /media/yudong/data/ros2_ws
export PYTHONPATH=$PWD/src/crazyswarm2/crazyflie_examples:$PYTHONPATH
PYTHON=/home/yudong/anaconda3/envs/drones/bin/python
TB_DIR=$PWD/runs/three_v_one

$PYTHON -m crazyflie_examples.three_v_one_baseline.train \
  --iterations 300 --rollout-steps 2048 \
  --tensorboard-dir "$TB_DIR" --run-name mappo_seed7

$PYTHON -m crazyflie_examples.three_v_one_baseline.train_iqld \
  --episodes 1000 --max-steps 1750 \
  --expert-episodes 20 --bc-steps 2000 \
  --tensorboard-dir "$TB_DIR" --run-name iqld_seed11

$PYTHON -m crazyflie_examples.three_v_one_baseline.train_irl \
  --episodes 1000 --max-steps 1750 \
  --expert-episodes 20 --bc-steps 2000 \
  --tensorboard-dir "$TB_DIR" --run-name irl_seed13

$PYTHON -m crazyflie_examples.three_v_one_baseline.evaluate_game_theory \
  --episodes 100 --max-steps 1750 \
  --tensorboard-dir "$TB_DIR" --run-name game_theory_seed17
```

上面四条命令是相互独立的实验，不需要同时执行。正式训练时建议为不同参数和随机种子设置不同的 `--run-name`。

## 打开 TensorBoard

另开一个终端：

```bash
cd /media/yudong/data/ros2_ws
/home/yudong/anaconda3/envs/drones/bin/tensorboard \
  --logdir "$PWD/runs/three_v_one" --port 6006
```

浏览器打开 `http://localhost:6006`。如果不想记录某次调试运行，可添加 `--no-tensorboard`。

## 主要曲线

- MAPPO：`loss/*`、`rollout/*`、`prediction/mean_confidence`、`episode/*`。
- IQLD-MADDPG：`expert/*`、`loss/behavior_clone`、`loss/q`、`loss/value`、`loss/policy`、`loss/epn`、`assignment/geometric_cost`、`assignment/entropy`、`episode/*`。
- IRL-MADDPG：`loss/actor`、`loss/critic`、`loss/discriminator`，以及环境、判别器和混合回报。
- Game Theory：没有可训练参数；记录 `episode/*`、累计捕获率、累计平均回报和动作强度。
- 四种方法均记录 `progress/environment_steps`；涉及训练缓存的方法还记录 `buffer/*`。
- 鲁棒环境额外记录 `tracking/*`、`mocap/*`、`latency/*` 和 `domain/*`，详见 `ROBUST_SIMULATION.md`。

障碍物碰撞指标统计每个控制周期内发生碰撞的追捕机数量，因此一次持续接触可能贡献多个计数。TensorBoard 仅旁路读取现有状态，不改变环境、奖励、控制频率或检查点格式。
