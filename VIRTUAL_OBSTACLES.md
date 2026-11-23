# 虚拟障碍物配置

虚拟障碍物不需要在 Nokov 中建立刚体。程序把它们视为贯穿飞行高度的竖直圆柱，并在训练、策略观测、博弈论规划、逃逸机控制和实机安全层中使用同一布局。

## 坐标定义

障碍物坐标系为 `initial_swarm_centroid`：

- 实机起飞前读取四架无人机的 Nokov 世界坐标；
- 四架无人机初始 XY 坐标的平均值作为局部原点；
- JSON 中的 `x`、`y` 是相对该原点的偏移，方向与 Nokov 世界坐标轴一致；
- `radius` 是虚拟圆柱半径；
- `safety_margin` 会额外加在半径外，构成无人机禁止进入的安全区。

示例配置位于：

```text
crazyflie_examples/config/virtual_obstacles_3v1.json
```

```json
{
  "coordinate_frame": "initial_swarm_centroid",
  "safety_margin": 0.15,
  "obstacles": [
    {"name": "virtual_cylinder_1", "x": 0.0, "y": 0.55, "radius": 0.16},
    {"name": "virtual_cylinder_2", "x": 0.55, "y": -0.35, "radius": 0.16}
  ]
}
```

当前固定观测最多支持四个障碍物。少于四个时剩余槽位自动填充；超过四个会在启动时拒绝运行，避免训练和实机观测维度不一致。

## 训练

训练和实机应使用同一份 JSON：

```bash
OBSTACLES=/media/yudong/data/ros2_ws/src/crazyswarm2/crazyflie_examples/config/virtual_obstacles_3v1.json

python3 -m crazyflie_examples.three_v_one_baseline.train \
  --obstacles-file "$OBSTACLES" --output checkpoints/three_v_one

python3 -m crazyflie_examples.three_v_one_baseline.train_iqld \
  --obstacles-file "$OBSTACLES" --output checkpoints/no_graph_iqld

python3 -m crazyflie_examples.three_v_one_baseline.train_irl \
  --obstacles-file "$OBSTACLES" --output checkpoints/no_graph_irl
```

若省略 `--obstacles-file`，程序使用与示例 JSON 相同的两个内置障碍物。建议实际实验始终显式传入文件，避免复制到总控机后不清楚当前布局。

## 实机

构建后配置文件位于：

```text
/media/yudong/data/ros2_ws/install/crazyflie_examples/share/crazyflie_examples/config/virtual_obstacles_3v1.json
```

实机命令示例：

```bash
python3 -m crazyflie_examples.three_v_one_baseline.real_flight \
  checkpoints/no_graph_iqld/no_graph_iqld_maddpg_latest.pt \
  --baseline no_graph_iqld_maddpg \
  --obstacles-file /media/yudong/data/ros2_ws/install/crazyflie_examples/share/crazyflie_examples/config/virtual_obstacles_3v1.json \
  --roles cf1,cf2,cf3,cf4
```

实机安全层在进入障碍物安全区前删除朝向障碍物的径向速度；若无人机已经位于安全区内，则发送远离圆柱中心的恢复速度。该安全层同时作用于三架追捕机和逃逸机，不依赖学习策略是否已经收敛。

加入障碍物后共享观测由 20 维增加到 32 维，因此之前生成的迁移检查点不兼容，MAPPO、IQLD 和 IRL 都必须重新训练。博弈论方法不需要检查点。

