# 3V1 PyBullet 风格三维界面

`pybullet_gui.py` 是现有 `ThreeVOneEnv` 的独立可视化器。默认使用兼容的 Matplotlib 3D 窗口仿照 PyBullet GUI；位置更新、速度积分、奖励、捕获判定和障碍物碰撞仍全部由原环境执行，因此开启 GUI 不会改变仿真结果。

## 显示内容

- 三架不同颜色的追捕机和一架红色逃逸机；
- 固定高度的二维围捕运动及三维 Crazyflie 简化模型；
- 追捕机围捕三角形和四机历史轨迹；
- 场地边界、网格、虚拟障碍物本体及安全半径；
- 当前回合、步数、包围状态、捕获状态和 50 Hz 控制状态。

## 安装依赖

默认界面需要 NumPy 和 Matplotlib；`drones` Conda 环境已经包含这些依赖。原生 PyBullet 后端为可选功能，需要 `pybullet`，但当前 NVIDIA 595.84 驱动与已安装 PyBullet 3.2.x 的 ExampleBrowser 存在 GLX/`glGenBuffers` 崩溃，因此当前机器应保持默认 `matplotlib` 后端。

## 无检查点运行

game-theory 基线不需要模型文件，最适合先检查界面：

```bash
cd /media/yudong/data/ros2_ws
export PYTHONPATH=$PWD/src/crazyswarm2/crazyflie_examples:$PYTHONPATH

python3 -m crazyflie_examples.three_v_one_baseline.pybullet_gui \
  --baseline game_theory \
  --obstacles-file src/crazyswarm2/crazyflie_examples/config/virtual_obstacles_3v1.json
```

构建并 `source install/setup.bash` 后也可以运行：

```bash
ros2 run crazyflie_examples three_v_one_gui --baseline game_theory
```

## 加载学习基线

检查点放在第一个位置参数：

```bash
python3 -m crazyflie_examples.three_v_one_baseline.pybullet_gui \
  /absolute/path/to/no_graph_iqld_maddpg_latest.pt \
  --baseline no_graph_iqld_maddpg \
  --obstacles-file /absolute/path/to/virtual_obstacles_3v1.json
```

`mappo_open` 和 `no_graph_irl_maddpg` 使用相同命令格式，只需替换 `--baseline` 和检查点。GUI 使用检查点中的任务配置；显式传入 `--obstacles-file` 时，仅以该文件覆盖障碍物布局。

## 快捷键

- `Space`：暂停或继续；
- `N`：暂停时前进一步；
- `R`：重置当前回合；
- `T`：显示或隐藏轨迹；
- `C`：恢复默认相机；
- `Q`：退出。

鼠标可以旋转、平移和缩放三维视角，操作方式与 PyBullet GUI 相近。

`--renderer pybullet` 可启动隔离的原生 PyBullet 窗口，供将来升级驱动或 PyBullet 后测试；当前机器不要添加该参数。

## 常用参数

- `--episodes 0`：持续运行，默认值；正数表示运行指定回合数；
- `--max-steps N`：限制每个回合的步数；
- `--speed 2`：以两倍墙钟速度播放，控制步长仍为 0.02 秒；
- `--seed N`：设置环境随机种子；
- `--trail-length N`：设置每架无人机保留的轨迹长度；
- `--renderer matplotlib`：默认且已验证的三维窗口后端；
- `--renderer pybullet`：可选原生 PyBullet 后端，当前驱动组合不可用；
- `--modern-opengl`：仅作用于原生 PyBullet 后端；
- `--headless --no-realtime`：无窗口快速回归测试。

## gym-pybullet-drones 素材场景(默认开启)

`--renderer pybullet` 与 `--headless` 现在默认加载 gym-pybullet-drones 素材
构建场景(`--no-assets` 可回退到旧的几何调试线风格)：

- 地面：pybullet_data 的 `plane.urdf`(纹理网格地面)；
- 围栏：与 6V2 SDGCR GUI 一致的四面低矮蓝色半透明围墙；
- 障碍物(柱子)：与 6V2 SDGCR GUI 一致的灰色半透明圆柱，并保留核心
  半径圈与安全半径圈便于判读；
- 无人机：`cf2x.urdf`(+ `cf2.dae` 网格)的 Crazyflie 2X 模型，按角色着色
  (三架追捕 + 一架红色逃逸)。

素材目录默认是 `/media/yudong/data/gym-pybullet-drones-main`，可用
`MTV2_PYBULLET_ASSETS` 环境变量或 `--assets-path` 覆盖，避免硬编码路径。
该场景需要 numpy<2 的解释器(与 PyBullet 二进制兼容)，例如：

```bash
cd /media/yudong/data/ros2_ws
export PYTHONPATH=$PWD/src/crazyswarm2/crazyflie_examples:$PYTHONPATH
/home/yudong/anaconda3/envs/uav/bin/python -m \
  crazyflie_examples.three_v_one_baseline.pybullet_gui \
  --baseline game_theory --renderer pybullet --episodes 1
```

legacy GUI 默认采用 `--scene-preset gui_scattered`，MtV2 GUI 默认采用
`--spawn scattered`。两者都使用同一个全地图独立采样规则：三架追捕机零散
分布、彼此保持间距且不会开局包围逃逸机，必须先完成接近与封口。旧协议仍可
分别用 `--scene-preset legacy`、`--spawn far1` 或 `--spawn rand_open` 选择。

MtV2(分裂覆盖)场景使用同一套素材渲染，并额外绘制两个出口 MOUTH、诱饵
DECOY、真实出口金色内环与隐藏侧/feint 信息，见
`crazyflie_examples/mt_v2_pybullet_gui.py`。

PyBullet 默认使用与 6V2 SDGCR GUI 相同的 OpenGL3+ 渲染路径，以正常显示
棋盘纹理和网格素材；只有 OpenGL3+ 无法启动时才显式添加 `--opengl2` 回退。
# Successful episode replay

Successful MtV2 and hard episodes use the same versioned NPZ/JSON format. They
can be replayed without loading a policy:

```bash
python replay_successful_episode.py \
  evaluations/successful_episodes/scene1_mtv2_success.npz
python replay_successful_episode.py \
  evaluations/successful_episodes/scene2_hard_success.npz
```

Use `--headless --speed 1000` for a fast DIRECT integrity check. Generate the
paper figures with `plot_successful_episode.py EPISODE.npz`.

For real flight, `real_flight --replay-data ... --acknowledge-open-loop`
transmits the recorded world-frame velocity commands to all four aircraft. It
does not run a policy and does not add position feedback, so it reproduces the
command sequence but cannot guarantee identical physical positions. Mocap
staleness, arena bounds and emergency separation can abort the replay.
