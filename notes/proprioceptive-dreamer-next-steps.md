# 下一步行动计划：从文档到可运行 MVP

> 目标：把 `proprioceptive-dreamer-v1` 从“架构说明”落地成可训练、可评估、可迭代的最小系统。

## 0. 先明确“本周可交付”

本周只追一个目标：**跑通端到端训练闭环**（哪怕性能一般）。

**完成标准（Definition of Done）**
1. 能读取序列数据并构造 `K` 步窗口样本。
2. 能前向通过：Encoder + GRU/RSSM + 多任务头。
3. `L_obs/L_reward/L_continue/L_stuck/L_slip/L_progress/L_kl` 可反向传播。
4. 训练 10k~50k steps 不崩溃，loss 有下降趋势。
5. 能导出基础可视化：`stuck_prob/slip_prob/progress_pred/interaction_map`。

---

## 1. 第一步：冻结范围，先做“GRU 版 MVP”

先不要直接上完整 DreamerV3 技巧，先做：

- 时序模块：`GRU`（后续可替换为 RSSM stochastic）
- 输出头：
  - next_proprio
  - reward
  - continue
  - stuck_prob
  - slip_prob
  - progress_delta
  - interaction_map (12x3)
- 策略头先可选：
  - A 方案：先不训 actor，只训 world model
  - B 方案：加一个行为克隆/监督 actor 头用于冒烟验证

**为什么这样做**：先验证“本体+历史序列”是否能学到稳定 latent 表示，避免过早引入 RL 不稳定性。

---

## 2. 第二步：把数据协议定死（最关键）

建议先定义统一样本结构（无论仿真/实车日志）：

```text
sample_t = {
  obs_t: {
    vx, vy, wz,
    ax, ay, az, gx, gy, gz,
    roll, pitch, yaw,
    wheel_speed[4],
    contact_flag
  },
  path_t: {
    e_lat, e_yaw, s_progress,
    target_distance, local_curvature,
    preview_points[M,2]
  },
  action_t: {
    target_speed, target_yaw_rate
  },
  reward_t,
  done_t,
  labels_t: {
    stuck, slip, mode(optional),
    interaction_map(optional privileged)
  }
}
```

再封装 `SequenceDataset`：
- 输入：`(o_{t-K+1:t}, a_{t-K:t-1}, path_{t-K+1:t})`
- 监督：`o_{t+1}, reward_t, continue_t, stuck/slip/progress, interaction(optional)`

---

## 3. 第三步：最小训练配置（建议直接抄）

- batch size: 64
- sequence length K: 20
- encoder hidden: 128
- gru hidden h: 256
- stochastic z: 先省略（GRU-only），或设 32
- optimizer: AdamW(lr=3e-4, weight_decay=1e-5)
- grad clip: 1.0

损失权重初值：
- `λ_obs=1.0`
- `λ_reward=0.5`
- `λ_continue=0.2`
- `λ_stuck=0.5`
- `λ_slip=0.5`
- `λ_progress=0.5`
- `λ_inter=0.3`（无标签则置0）
- `λ_kl=1e-3`（如果启用 z）

---

## 4. 第四步：先做 4 个“冒烟实验”

1. **Overfit-1-batch**：单 batch 能否快速降损。
2. **No-history ablation**：K=1 对比 K=20，验证历史序列价值。
3. **No-derived-feature ablation**：去掉 slip/stuck_rule，观察 stuck/slip 头退化。
4. **No-interaction-head ablation**：验证交互场是否提升 recover 相关指标。

如果这四个实验跑不通，不要进入 RL 阶段。

---

## 5. 第五步：指标与看板（必须先做）

每 N step 记录：
- total loss + 各子损失
- stuck/slip AUC 或 F1
- progress MAE
- continue accuracy
- interaction_map BCE/IoU（若有标签）

每个 eval episode 记录：
- 路径跟踪误差（e_lat/e_yaw）
- 平均前进率（progress per sec）
- 卡困次数与平均脱困时长
- 脱困后 rejoin 成功率

---

## 6. 第六步：进入策略训练的门槛

满足以下条件再训 actor-critic：
1. world model 的 next_proprio 误差稳定收敛。
2. stuck/slip/progress 头在验证集可用（非随机）。
3. latent 可视化能分离 track/recover 样本簇（粗分离即可）。

然后按顺序：
- 先冻结 world model 训 actor-critic
- 再小学习率联合微调

---

## 7. 七天执行清单（可直接照着排期）

- Day 1：数据字段统一 + SequenceDataset
- Day 2：Encoder+GRU+heads 前向跑通
- Day 3：loss 组合与训练循环
- Day 4：日志与可视化（tensorboard/wandb）
- Day 5：四个冒烟实验
- Day 6：修正损失权重与特征工程
- Day 7：出 MVP 报告（曲线、案例、失败样例）

---

## 8. 你现在立刻可以做的 3 件事

1. 把你现有数据按上面的统一字段对齐（这是第一阻塞项）。
2. 用 GRU-only MVP 跑一个 overfit-1-batch（最快发现工程错误）。
3. 把训练日志和 20 条失败样例导出，优先修 stuck/slip 标签质量。

> 结论：下一步不是“继续写大而全文档”，而是把数据协议、MVP 训练闭环和冒烟实验先打通。
