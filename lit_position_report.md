# 文献定位:序列感知可解释子任务发现 + 条件蒸馏(3V1 Pursuit-Evasion)

日期:2026-09-02。调研基于 web_search + 两个子代理对原文/专利/ar5iv 全文的核查。

## 一句话定位
你的方法在"pursuit-evasion 场景 × 专家演示离线 × 序列感知(时序)聚类出离散可解释相位 × 条件蒸馏进策略"这一组合上是**现有文献中的空缺**。但"发现→预测→条件化"这条完整链路的**近邻先例是存在的**(IPOD 专利、LOKI),审稿人最可能用它质询新颖性,须正面处理。

## 最接近的 2 篇近邻(必须正面处理)
1. **IPOD(Interpretable imitation learning via prototypical option discovery)**
   - NEC 专利:US12380360B2 / JP7466702B2(申请 US20210374612A1),发明人 Yu/Chen/Cheng;未见公开论文。
   - 机制:BC+soft-attention 找瓶颈状态 → DBSCAN 切变长段;LSTM+soft-attention 段嵌入;K 个原型=真实专家子轨迹(可解释);option 标准四元组 ⟨I,π,β,g⟩;高层 π_h(o|s) 选 option,低层以 option 为条件的模仿策略 λ(a|s,o) 用 imitation loss 蒸馏。
   - 与你的差异:段级归属、非 gated multi-head attention、无 per-step 激活剖面解码器、载体是专利(无公开论文/代码/实验)、目标数据是决策/医疗时序。
   - URL: https://patents.google.com/patent/US12380360 ; https://www.justia.com/patents/12380360
2. **LOKI(Offline Discovery of Interpretable Skills from Multi-Task Trajectories)**
   - Zhu, Vanniasinghe, Chen, Lee; arXiv:2602.01018, 2026。
   - 机制:EVQ-VAE(弱任务标签)+entropy 宏切分 → 自监督序列模型+聚类微切分 → 终止 β(s)(93.7%) + 单一统一条件 BC 策略 π(a|s,k)。
   - 差异:弱监督(依赖任务标签)、VQ/熵边界而非逐步激活聚类、高层是 canonical 技能序列而非独立逐步因果预测器。
   - URL: https://ar5iv.labs.arxiv.org/html/2602.01018

## 方法学近邻(机制相近,场景/范式不同)
- **RODE**(Wang et al., ICLR 2021, arXiv:2010.01523):先离线式 k-means 按"动作效果"聚类出 K 个离散角色动作空间(固定),顶层 role selector(GRU 历史编码,每 c 步重选角色)按 Q=z_τᵀz_ρ 打分,底层 role policy 受限动作空间执行。→ "离散发现 + 角色条件化 + 显式时间粒度"的 MARL 锚点,与你的"分工"维度最近,但分解对象是动作空间(非时序相位)、数据来自 RL 自举(非专家演示)、角色是分工(非时序阶段)。
- **ROMA**(Wang, Dong, Lesser, Zhang; ICML 2020, arXiv:2003.08039):角色 = 连续随机嵌入(多元高斯,维3),role encoder 在线采样,role decoder 超网络生成策略参数(条件化);正则 ℒ_I(GRU 历史→角色可识别/时序稳定)、ℒ_D(不相似度→子任务专化);角色在线涌现,与离线/专家/离散/因果无关。→ 你引言里要区分的"隐式、无监督、在线角色涌现"一类。
- **Multi-Head DDPG for Pursuit-Evasion**(Lehis et al., ISI Vol.30 No.12, 2025;DOI 10.18280/isi.301204):10 追捕者 vs 2 逃跑者(20×20 网格),actor 三头输出追赶/聚合/分离三力 + softmax 权重在线加权;全在线 DDPG,无离线/聚类/专家数据,Success 85.2%。→ 同场景(追逃)不同机制(力场/控制分解,非时序子任务),作背景对照。注:非 2024、非 Elsevier Information Sciences、非 3v1(官方元数据 2025-09-21 收到,12-31 online)。
- **SPiRL**(Pertsch et al., CoRL 2020, arXiv:2010.11944):单智能体离线专家数据 + 时序一致性正则 → 连续技能嵌入 → 技能条件化策略;常用作技能-先验类比。→ 背景引用。

## 其它相关(按是否"可解释/预测器/是否控制/源自示范"区分)
- PODNet(arXiv:1911.00171):categorical VAE + option inference + option-conditioned policy。
- OPAL(arXiv:2010.13611):轨迹级 VAE primitive,skill 连续。
- VQ-BeT(arXiv:2403.03181)/ BeT(arXiv:2206.11251):动作模态 k-means/VQ 离散,**非时间相位**。
- DIAYN(arXiv:1802.06070)/ DADS(arXiv:1907.01657):在线 RL 互信息技能,无示范、不可解释。
- Behaviour Discovery & Attribution(arXiv:2503.14973, Rishav/Nath/Michalski/Kahou):纯解释、无策略。
- Cockcroft(arXiv:2001.06793):贝叶斯切分 + IRL + SVM 学 options。
- CPD-HRL(arXiv:2510.24988):Transformer CPD + Option-Critic(在线 RL)。
- VQ-HIL(arXiv:2301.12962):RL 可预测性奖励找子目标 + VQ 规划 + 条件低层。
- HIGL(arXiv:2303.11166):图上子目标 + subgoal-conditioned 策略蒸馏(RL)。
- SkillDiffuser(CVPR 2024):扩散任务执行中的技能抽象可解释层次规划。
- Segment-Conditioned Latent-Intent Framework for Cooperative Multi-UAV Search:每 agent 一个慢速隐技能,按固定 segment 支配行为——与你"分段条件化"同构,值得引用。

## 五条差异化落点(写创新点时用)
1. **per-step 软激活剖面解码器**(step_gate[K,T]):显式建模"每个相位在哪些时间步激活",能区分顺序不同但均值相同的序列(approach-then-hold vs hold-then-approach)——旧常量偏移/纯 FiLM 做不到,序列聚类独有。
2. **完全无监督**:不依赖任务标签/弱任务标签(对比 LOKI 的 EVQ-VAE 弱监督)。
3. **独立的逐步状态→相位因果预测器**(state→mode,准确率 0.840):不是"canonical 宏序列",是逐时刻可在线调用的因果映射(这里既是你的创新点,也是你发现"条件化在强基座冗余"的分析工具)。
4. **gated MHA 序列编码器**(序列感知聚类)而不是段级 DBSCAN/熵边界。
5. **pursuit-evasion(MARL)+ 专家演示离线 + 公开 benchmark 上的闭环验证**:与上述单智能体/纯 RL 工作形成场景空缺。

## 诚实提示
- IPOD 是专利(无公开论文/代码/实验),必须在定位里正面处理,否则会被审稿人当作最强 prior art。
- 你已确认的负结果(条件化在强基座无正效果)与这些近邻的"提性能"主张不一致——这是你的**真实差异点**之一,可作为"我们进一步证明其收益边界(在强基座/已收敛策略上冗余)"的诚实论述,而非隐瞒。

## 来源 URL(核查过的)
- IIETA: https://www.iieta.org/journals/isi/paper/10.18280/isi.301204 ; PDF https://www.iieta.org/download/file/fid/192147
- ROMA: https://arxiv.org/abs/2003.08039 ; https://github.com/TonghanWang/ROMA
- RODE: https://arxiv.org/abs/2010.01523 ; https://github.com/TonghanWang/RODE ; https://iclr.cc/virtual/2021/poster/3033
- IPOD 专利: https://patents.google.com/patent/US12380360 ; https://www.justia.com/patents/12380360
- LOKI: https://ar5iv.labs.arxiv.org/html/2602.01018
- SPiRL: https://arxiv.org/abs/2010.11944
- PODNet: https://arxiv.org/abs/1911.00171
- OPAL: https://arxiv.org/abs/2010.13611
- VQ-BeT: https://arxiv.org/abs/2403.03181 ; BeT https://arxiv.org/abs/2206.11251
- DIAYN: https://arxiv.org/abs/1802.06070 ; DADS https://arxiv.org/abs/1907.01657
- Behaviour Discovery & Attribution: https://arxiv.org/abs/2503.14973
- Cockcroft: https://arxiv.org/abs/2001.06793
- CPD-HRL: https://arxiv.org/abs/2510.24988
- VQ-HIL: https://arxiv.org/abs/2301.12962
- HIGL: https://arxiv.org/abs/2303.11166
- SkillDiffuser: https://mlanthology.org/cvpr/2024/liang2024cvpr-skilldiffuser/
- Segment-Conditioned Latent-Intent for Cooperative Multi-UAV Search: https://www.sciencedirect.com/org/science/article/pii/S1546221826001840

---

## 附:重新设计主实验的"价值-预算边界"数据(2026-09-02,3 seeds×6 budgets)

弱基座(随机 actor = IQLD BC 预训练起点),条件化 vs 无条件化,holdout 动作拟合 MSE。

| 训练预算 | 无条件化 MSE | 条件化 MSE | dMSE(cond−uncond) |
|---|---|---|---|
| 100 | 0.1516±0.0018 | 0.1450±0.0011 | −4.34% |
| 300 | 0.1151±0.0011 | 0.1084±0.0008 | −5.81% |
| 600 | 0.0729±0.0031 | 0.0753±0.0030 | +3.23% |
| 1000 | 0.0504±0.0017 | 0.0597±0.0015 | +18.31% |
| 2000 | 0.0396±0.0022 | 0.0437±0.0005 | +10.34% |
| 3000 | 0.0385±0.0003 | 0.0402±0.0011 | +4.21% |

结论:条件化的边际价值集中于早期(预算≤300,−4.3%~−5.8%,std 小),随预算增加反超(+3%~+18%)。
→ 支撑"序列感知结构 = 结构化先验,预算受限有价值,基座自学习完成后冗余"的核心论点;也解释近邻
(RODE/LOKI/IPOD)在"弱基座 + 结构为必要条件 + 分层"设定下为何能改善——它们把结构信息聚焦在信息匮乏期。
