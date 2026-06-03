# 本体感知 Dreamer 风格小车策略网络结构（中文版 v1）

## 1. 任务目标

仅使用小车本体传感器（车速、轮速、IMU、历史动作、路径相对信息），不依赖相机、激光雷达等外部感知，实现：

1. 正常路径跟踪
2. 异常接触/卡困识别
3. 自主脱困
4. 脱困后回归参考路径
5. 局部不确定环境中的合理决策

核心思想：不直接重建外部几何地图，而是通过历史本体观测与动作学习“隐式环境交互表征（belief/interaction latent）”，并基于该隐变量完成控制与模式决策。

---

## 2. 总体网络结构

1. 输入编码器（Encoder）
2. 时序世界模型（RSSM/GRU/Transformer）
3. 隐式环境交互表征头（Interaction Head）
4. 任务预测头（Task Heads）
5. 策略与价值网络（Actor/Critic）
6. 模式管理头（Mode Head，推荐）

流程：

`o_t, a_{t-1}, path_info_t -> Encoder -> e_t -> RSSM -> (h_t, z_t) -> 多任务头 + Actor/Critic`

---

## 3. 输入定义

### 3.1 本体观测 `o_t`
- 车体速度：`vx, vy, wz`
- IMU：`ax, ay, az, gx, gy, gz, roll/pitch/yaw`
- 轮速：四轮转速
- 接触信号：碰撞开关
- 上一时刻动作 `a_{t-1}`：`target_speed/throttle`, `target_yaw_rate/steering`

### 3.2 路径相对信息 `path_info_t`
- 横向误差 `e_lat`
- 航向误差 `e_yaw`
- 累计进度 `s_progress`
- 局部目标距离 `target_distance`
- 局部曲率 `local_curvature`
- 预瞄点（车体坐标系）`preview_points`

### 3.3 历史窗口 `H_t`
- 采用最近 `K` 步序列：
  `H_t = {(o_{t-K+1}, a_{t-K}), ..., (o_t, a_{t-1})}`
- 建议 `K=10~40`（覆盖 1~3 秒）

---

## 4. 输入编码器 Encoder

输入：

`x_t = concat(proprio_raw_t, derived_features_t, path_info_t, a_{t-1})`

推荐派生特征：
- `slip_ratio`
- `progress_delta`
- `action_change`
- `yaw_error_rate`
- `vibration_level`
- `stuck_score_rule`

结构建议：2~3 层 MLP（Linear + LayerNorm + SiLU/ReLU），输出 `e_t ∈ R^128`。

---

## 5. 时序世界模型（核心）

采用 Dreamer 风格 RSSM：
- 确定性状态 `h_t`
- 随机状态 `z_t`
- 联合状态 `s_t = [h_t, z_t]`

更新：
- `h_t = GRU(h_{t-1}, concat(z_{t-1}, a_{t-1}, e_t))`
- `p(z_t | h_t)`（先验）
- `q(z_t | h_t, e_t)`（后验，训练使用）

部署/想象 rollout 使用 prior，训练使用 posterior。

该 latent 主要表达：外部约束位置倾向、打滑/硬碰/陷车类型、恢复动作有效性、不确定性。

---

## 6. 交互表征头 Interaction Head

推荐输出“极坐标交互场”：
- `interaction_map ∈ R^(N×C)`
- 方向扇区 `N=12/16/24`
- 通道 `C=3~5`

通道建议：
1. `passability`
2. `trap_risk`
3. `recovery_gain`
4. `contact_prob`（可选）
5. `surface_uncertainty`（可选）

说明：纯本体感知更适合学习“动作-结果相关交互语义”，不适合稳定重建完整几何 occupancy。

---

## 7. 任务预测头 Task Heads

输入默认来自 `s_t`（必要时加 `a_t`）：
- 下一时刻观测预测（next proprio）
- 奖励预测（reward）
- 继续概率（continue/done）
- `stuck_prob`
- `slip_prob`
- `progress_delta`
- `mode`（`track/recover/rejoin`）

---

## 8. Actor/Critic

策略输入建议：

`concat(h_t, z_t, interaction_map_flatten, stuck_prob, slip_prob, mode, path_info_summary)`

动作输出建议（MVP）：
- `target_speed`
- `target_yaw_rate` 或 `target_steering`

Critic 输出：`V(s_t)`（可后续升级 distributional）。

模式化策略初版建议：单 Actor + mode one-hot 条件输入。

---

## 9. 推荐尺寸（MVP）

- `encoder_dim=128`
- `h_dim=256`
- `z_dim=32`
- `interaction_bins=12`
- `interaction_channels=3`
- `actor_hidden=256`
- `critic_hidden=256`

---

## 10. 损失函数

`L_total = L_dyn + λ_r L_reward + λ_c L_continue + λ_obs L_obs + λ_stuck L_stuck + λ_slip L_slip + λ_prog L_progress + λ_inter L_interaction + λ_mode L_mode + λ_kl L_kl + L_actor_critic`

关键项：
- `L_obs`：下一时刻观测重建（MSE/Gaussian NLL）
- `L_reward`
- `L_continue`（BCE）
- `L_stuck / L_slip / L_mode`（BCE/CE）
- `L_progress`（MSE/Huber）
- `L_interaction`（有特权标签时监督）
- `L_kl = KL(q||p)`（Dreamer 风格）

---

## 11. 训练流程

1. **阶段 1**：先训世界模型（Encoder + RSSM + 多任务头）
2. **阶段 2**：冻结/半冻结世界模型，训练 Actor/Critic
3. **阶段 3**：联合微调（小学习率）
4. **阶段 4**：课程学习（从无障碍到复杂脱困）

---

## 12. 可解释中间量（强烈建议可视化）

- latent norm
- stuck/slip 概率
- mode logits
- progress 预测
- interaction_map（扇区图）
- actor 输出
- critic value
- continue probability

---

## 13. 最小可行版本（MVP）

输入：车速、轮速、IMU、上一动作、路径横向/航向误差、预瞄点。

时序模块：GRU 或 RSSM。

输出头：next proprio / reward / continue / stuck / slip / progress / 12方向 interaction_map。

策略输出：`target_speed + target_yaw_rate`。

模式：`track/recover/rejoin`。

---

## 14. 一句话总结

该方案本质不是“仅凭本体感知重建几何地图”，而是学习时序 latent 并解码为“可通行性/卡困风险/脱困收益”的交互场，再据此完成跟踪、异常识别与自主脱困。
