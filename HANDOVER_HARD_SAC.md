# 交接文档:HARD 场景 IQLD × SAC-先验自举

_分支:_ `debug/deployment-readiness`(worktree `iqld-gate`),工作目录
`src/crazyswarm2/crazyflie_examples`(注意包路径双写:
`crazyflie_examples/crazyflie_examples/three_v_one_baseline/`)。

## 1. 当前结论(截至本文件)

在 **hard 场景**(scene preset `hard` = far1:出生带 1.3–2.1m)上,从离线 IQLD
checkpoint(`offline_hard best_safe`,正式 nominal≈0.51)接续的 **SAC-先验自举**
(sacprior_hard150,150ep)首次把 IQLD 家族拉到教师水平:

| 方法 | nominal | robust | colfree | 备注 |
|---|---|---|---|---|
| **sacprior_hard150 best_safe**(ep60) | 0.814 | 0.764 | 0.998/0.998 | 5×100,robust 超教师(边际 z=1.80,p=0.07) |
| **sacprior_hard150 latest**(ep150) | 0.796 | 0.784 | 0.998/0.99 | 5×100,**robust 显著超教师(z=2.55,p=0.011)** |
| **anneal300 best_safe**(ep40) | 0.790 | 0.772 | 1.0/0.994 | 5×100,robust 超教师(p=0.036) |
| **anneal300 latest**(ep300,β 已放开) | 0.808 | **0.842** | 1.0/0.99 | 5×100,**robust 强烈超教师(z=4.87,p<1e-5)** |
| game-theory 教师(hard,正式 5×100) | **0.834** | 0.714 | 1.0/0.998 | nominal 高于我们但 NS(z −0.8/−1.6) |
| 接续点 offline_hard best_safe | 0.51 | 0.36 | — | lite |
| 旧 Cal-QL/朴素在线(hard) | 0.16–0.55 | — | — | 全部退化 |

固定 β 版 robust 显著高于教师 0.714(latest 0.784,z=2.55,p=0.011);**β 退火 300ep 版进一步把 robust 推到 0.842(z=4.87,p<1e-5)**,nominal 0.79–0.81 略低于教师 0.834 但不显著。**注意**:seed11 25ep 校验对该族严重低估(anneal300 末段校验 ~0.28 而正式 robust 0.842)——判定一律以多种子 5×100 为准。**关键机制**:冻结 IQLD 先验做 trust-region + SAC critic
target 自举 + β 退火(详见下节),解决了此前"样本内无法超越数据"
与"样本外 max 幻影崩溃"两个问题。

## 2. 机制一句话

IQLD 离线蒸馏定分布 → 冻结确定性 IQLD actor 为 `π_prior`(安全带)→ SAC 式
critic TD target(min Q̄(s′,π(s′))−αlogπ)让 Q 在**策略自己的动作**上变准 →
策略项 `β(t)·‖π_θ−π_prior‖² + α·logπ − minQ(s,π)` 在分布内自举 → β 3.0→0.6
退火逐步放开。critic 为 SAC target(非样本内),value 仅作辅助;50/50 在线/离线
对称采样;eval/部署用确定性均值。

## 3. 代码改动(均默认关闭/兼容,iwqld 路径不变)

- `role_iqld.py`:
  - `SharedRoleActor` 增加 `stochastic/log_std` + `sample()`(tanh-Gaussian,
    重参数化 + log prob),trunk 参数与旧版逐键一致 → 旧 checkpoint 可载入;
  - `RoleIQLDAgent.__init__(actor_mode='iqld'|'iac', log_std_init=-1.0)`;iac 时
    建 `q1_target/q2_target`(polyak `update_q_targets`);`policy()` 恒走确定性
    均值;checkpoint extra 记录 `actor_mode`,iac 时另存 target-Q;
  - `RoleIQLDRuntime` 按 ckpt 的 actor_mode 构造(iac checkpoint 可部署)。
- `train_iqld_expert.py`:
  - flags:`--actor-mode/--log-std-init/--entropy-coef/--sac-weight/
    --anchor-weight/--prior-weight/--q-target-tau`;
  - `_load_init_checkpoint`:iac 时 actor `strict=False` 载入 + deepcopy 冻结
    `agent.iqld_prior`;
  - `update_iqld`:iac 分支改用 SAC critic target;策略 loss =
    `anchor(优势加权BC) + prior·‖π_θ−π_prior‖² + sac_weight·(αlogπ−minQ)`;
  - 每 episode 按进度把 `agent.prior_weight` 退火到 `0.2×`。

**未提交/需跟进**:prior-anchor、SAC-critic-target、prior 退火这三处改动是在
`0c66c66`(IAC 骨架)之后加的,**当前仍是 working tree 修改**
(`role_iqld.py`, `train_iqld_expert.py`)——继续前先提交(附结果 JSON 与本文档)。

## 4. 已提交序列(本会话相关)

```
0c66c66 feat(exp): IAC-mode actor(SAC 项,默认关)
72bc92d result: hard IQLD vs expert(IQLD 离线+在线超不过教师)→ 引出 SAC 路线
cafe23b feat: scene-preset 全链路(scene_options 注入 env/采集/训练/评测)
f495493 feat+audit: SceneSamplingOptions + presets + analytic 可解性 + hard 预设
```

## 5. 关键产物路径(worktree 内,均在 `src/crazyswarm2/crazyflie_examples/`)

- 专家数据:`checkpoints/contract_v2/game_theory_expert_hard_{300,600}.pt`
  (hard 场景 300/600 合格 episode);
- 离线蒸馏:`runs/hard_bench/offline_hard_ckpt/no_graph_iqld_maddpg_best_safe.pt`
  (接续点,~ep800,val 0.52);`latest` ~ep1390;
- SAC-先验自举:`runs/hard_bench/sacprior_hard_ckpt/{best_safe,best_capture,latest}.pt`
  (150ep,β 固定 3.0);正式评测 `runs/hard_bench/liteeval_sacprior_{best,latest}/`;
- **退火版(运行中)**:`runs/hard_bench/sacprior_anneal_hard300`(300ep,β
  3.0→0.6),ckpt 输出 `runs/hard_bench/sacprior_anneal_hard_ckpt/`;
- 教师对照:`runs/hard_bench/formal_teacher_hard`(5×100,运行中)与
  `lite_teacher`(125-lite 0.840/0.728);
- 失败记录对照:`runs/hard_bench/{probe_online_v11_60_ckpt, online_hard_ckpt,
  lite_v11_on_hard, lite_offline_best, lite_offline_latest, BC...}`;
- TB 曲线:`runs/hard_bench/sacprior_hard150` 与 `sacprior_anneal_hard300`
  (capture/colfree/evader_success)。

## 6. 复现命令(镜像已跑通版本)

```bash
# SAC-先验自举(150ep,固定 β) —— 用后一律加 --actor-mode iac
python -m crazyflie_examples.three_v_one_baseline.train_iqld_expert \
  --expert-dataset checkpoints/contract_v2/game_theory_expert_hard_600.pt \
  --role-artifact checkpoints/scan_role_kstar_1_0.pt \
  --init-checkpoint runs/hard_bench/offline_hard_ckpt/no_graph_iqld_maddpg_best_safe.pt \
  --actor-mode iac --log-std-init -1.6 --entropy-coef 0.05 --sac-weight 0.3 \
  --anchor-weight 1.0 --prior-weight 3.0 \
  --episodes 150 --seed 11 --device cuda --updates-per-step 10 \
  --expert-fraction-start 0.5 --expert-fraction-end 0.5 \
  --exploration-noise 0.02 --minimum-exploration-noise 0.02 \
  --rollout-safety-guard --scene-preset hard \
  --validation-interval 15 --validation-episodes 25 \
  --tensorboard-dir runs/hard_bench --run-name <tag> --output runs/hard_bench/<tag>_ckpt

# 评测(hard 5×100)
python -m crazyflie_examples.three_v_one_baseline.deployment_evaluation \
  --baseline no_graph_iqld_maddpg --checkpoint <ckpt.pt> --device cuda \
  --seeds 101 211 307 401 503 --episodes-per-seed 100 \
  --scene-preset hard --output <dir>
```

## 7. 待办/下一步(建议顺序)

1. 等 `sacprior_anneal_hard300` 跑完(300ep,剩约 2h),5×100 评测 best/latest,
   与教师正式 5×100(`formal_teacher_hard`,CPU 在跑)对比 → 判断退火版是否把
   nominal 推到 ≥0.84;
2. 提交 pending 代码改动(prior anchor/SAC target/退火)+ 结果 JSON + 本文档;
3. 若退火版 ≥ 固定 β 版:多 seed(21/31)复验 + 更长(500ep);若 ≤:β 退火到
   0.3 或延长固定-β 时长重试;
4. 关注点:best@ep60 在固定 β 下即达 0.814,后期 latest 反而 nominal 略降
   (0.796)→ 检查是否 β 释放后微退,或 early-stop 更优;
5. 延伸问题(可选):legacy 场景用同法从 v11 续训是否也能接近 0.944 教师
   (此前 Cal-QL 只能守住 ~0.91)。

## 8. 环境/工具备忘

- python:`/home/yudong/anaconda3/envs/uav/bin/python`(conda `uav`);
- TB 读取:`tensorboard.backend.event_processing.event_accumulator`;
- runs/ 均 gitignored,需要留存的结果已复制进
  `three_v_one_baseline/*.json` 提交;
- 场景难度事实(供引用):hard 教师 clean 0.796(250)/0.840(125)、robust
  0.728;oracle×1.5≈1.0(场景全部可解);教师失败几乎全为逃逸者达目标。
