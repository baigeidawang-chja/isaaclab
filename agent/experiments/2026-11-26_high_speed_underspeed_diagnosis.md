# 1.2 m/s 高速欠速配对诊断

日期：2026-11-26（Asia/Shanghai）。本机 Isaac Lab 单环境仿真；本轮只诊断，不调模型、MPC、动作、限矩或地面参数。每组仅一次完整运行，不将逐拍样本当作独立重复。

## 实验问题与数据

检验：在原名义 MPC 接近输入上限时出现的 1.2 m/s 欠速，是 MPC 没有给足物理速度目标，还是持续给到底盘 1.5 m/s 后仍不能稳定保持目标？A 组保留名义 MPC 的速度和转向；B 组只将纵向物理目标替换为 1.5 m/s，转向仍逐拍由同一 MPC 给出。两组规划参考都是 1.2 m/s、世界航向 0 rad，seed 42，静止同初态，控制周期 0.05 s，连续 16 s。B 的 1.5 m/s **不是规划参考**，经原 ActionTerm 以 0.1 m/s/拍上升。

外置盘数据根目录 `D`：

```text
/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_high_speed_deficit/20261126_paired_1p2
```

| 组别 | 原始 CSV | 配置、预测及终止记录 |
| --- | --- | --- |
| A | `D/A_nominal_mpc/steps.csv` | 同目录 `metadata.json`、`runtime_environment.json`、`model_config.json`、`mpc_config.json`、`steering_config.json`、`predictions.jsonl` |
| B | `D/B_max_physical_input/steps.csv` | 同上；B 的预测轨迹是未执行其纵向输出的 MPC 反事实计划，不能当作 B 的实际预测误差 |

两组各 320 拍，均 `evaluation_horizon` 正常结束，320/320 次 `solved`、0 回退、0 提前终止或自动复位。A 前 8 s 对历史 F7 的参考、MPC 输出、执行目标、实际速度、航向和转向逐拍最大差均为 **0**；历史 F7/S1 原数据未改。B 的执行速度目标从 0 按 0.1 m/s/拍上升，在指令时刻 0.70 s 首次达到 1.5 m/s（执行后采样 0.75 s），随后保持 306 拍。MPC 的上一拍输入已核验为 ActionTerm 真正的上一拍执行动作。两组有效环境配置、USD、动作实现、模型、MPC 和转向配置逐项一致。

## 核心结果

速度为执行后车体纵向速度；偏差 = 1.2 减末 1 s 均速；区间为末 1 s 的最小至最大值；末尾带内时长为最后连续处于 [1.15, 1.25] m/s 的首末采样跨度。最后 4 s 趋势为对 80 个速度采样作线性拟合，仅作窗口内描述。

| 组别/时点 | 末 1 s 均速 | 有符号偏差 | 末 1 s 范围 (m/s) | 带内比例 | 末尾连续带内 | 最后 4 s 均速 | 最后 4 s 斜率 (m/s²) |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| A / 8 s | 1.147233 | +0.052767 | 1.113940–1.196197 | 35% | 0 s | 1.156588 | -0.003874 |
| A / 16 s | 1.163115 | +0.036885 | 1.132822–1.203507 | 80% | 0.40 s | 1.162974 | +0.002674 |
| B / 8 s | 1.178184 | +0.021816 | 1.137791–1.209194 | 90% | 0 s | 1.174913 | -0.002303 |
| B / 16 s | 1.140123 | +0.059877 | 1.103017–1.187786 | 45% | 0 s | 1.158214 | -0.008557 |

B 曾瞬时超过 1.2 m/s，8 s 末段均值也接近 1.2，但 16 s 末 1 s 均值为 **1.140123 m/s**，仅 45% 采样在容差带内，末拍不在带内；不能称为持续保持。A 的 16 s 末段均值进入容差带，但末尾只连续保持 0.40 s，同样不能称为稳定跟随。B 最后 4 s 均值 **1.158214 m/s**，A 为 1.162974 m/s，差 -0.004760 m/s；B 最后 4 s 仍有轻微负斜率，单次运行不足以认定稳态极限速度。

A 的 MPC 物理速度目标触及 1.5 m/s 上限的比例：前 8 s 76.25%，全 16 s 84.375%，末 1 s 85%。A 后 8 s 的实际执行目标均值 1.49924 m/s、最后 4 s 1.49910 m/s；B 后 8 s 一直为 1.5 m/s。最后 4 s 两组执行目标均值仅差 0.000898 m/s，明显小于各自的实际速度波动；因此这里没有形成“给 B 显著更多输入、B 显著更快”的证据。B 的未执行 MPC 速度提议不计作 B 的真实物理输入。

末 1 s 四轮数据（FL/FR/RL/RR 按真实 joint name 和 ID 9/8/4/5 取值）。单位：轮速 rad/s，扭矩 N·m；扭矩均为每个控制周期**最后一个物理子步**的数值，不是周期平均或接触力。裁剪比例是 20 个末段采样中 `|computed| > |applied| + 1e-4` 的占比。

| 组/轮 | 目标轮速 | 实测轮速 | 计算扭矩绝对均值 | 施加扭矩绝对均值 | 裁剪比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A/FL | 23.045 | 14.948 | 3.200 | 0.926 | 70% |
| A/FR | 23.062 | 16.644 | 4.089 | 0.984 | 85% |
| A/RL | 23.045 | 16.939 | 3.048 | 0.977 | 90% |
| A/RR | 23.062 | 15.551 | 3.088 | 0.981 | 90% |
| B/FL | 23.069 | 15.658 | 3.811 | 0.953 | 80% |
| B/FR | 23.085 | 15.417 | 2.947 | 0.968 | 80% |
| B/RL | 23.069 | 17.142 | 2.883 | 0.962 | 75% |
| B/RR | 23.084 | 16.532 | 3.177 | 0.988 | 95% |

两组目标轮速均约 23 rad/s，实测约 15–17 rad/s，末段大部分采样发生扭矩裁剪；执行层存在显著轮速跟随差距。此处只能定位线索，不能由轮级速度或扭矩推导低附着/滑移真值，也不能单独认定限矩是唯一原因。两组全程最大航向绝对误差 A 0.005202 rad、B 0.005515 rad；末 1 s 实际前轮转角绝对最大值 A 0.004506 rad、B 0.004489 rad，属于同一种近直行工况。

**结论：持续给 1.5 m/s 后，底盘在本次配置下未能稳定保持 1.2±0.05 m/s。** B 组 16 s 末 1 s 跑到均值 1.140123 m/s（范围 1.103017–1.187786），最后 4 s 均值 1.158214 m/s；瞬时达标不等于持续达标。由于 A 后半程也几乎给满物理目标，且 A/B 输出差远小于自身波动，本次结果**不支持现在调 MPC 权重或高速段模型来解释主欠速**。下一项最有区分度的检查是在固定 1.5 m/s 输入下记录每个 0.01 s 物理子步的四轮目标/实际轮速、computed/applied torque 与车体速度，检查限矩持续性和轮地响应；本轮不改限矩或地面。

## 复现与来源

在仓库根目录运行（`D` 为上文绝对路径）：

```bash
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --task Isaac-TractionProtect-CommandFollow-Car4WD-v0 --headless --profile fixed --speed 1.2 --heading 0 --duration 16 --segment_duration 16 --seed 42 --output_dir "$D/A_nominal_mpc"
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --task Isaac-TractionProtect-CommandFollow-Car4WD-v0 --headless --profile fixed --speed 1.2 --heading 0 --duration 16 --segment_duration 16 --seed 42 --physical-speed-override 1.5 --output_dir "$D/B_max_physical_input"
./isaaclab.sh -p scripts/user/analyze_nominal_mpc_high_speed.py --root "$D" --historical-f7 /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_mpc_longitudinal/20261124_chapter2/formal/F7
```

分析输出为 `D/core_metrics.csv`、`D/wheel_comparison.csv`、`D/paired_response.png`/`.pdf`、`D/diagnostic_checks.json`；原始控制台见 `D/A.console.log`、`D/B.console.log`。`D/source_snapshots_9efc7abd2582/` 与 `D/provenance.json` 保存实际源码快照、哈希及版本。主仓库运行时为 `main` / `3011944f8c809030c74f3ccbb7344e932ed059f7`；原有 WheeledLab、tdmpc2 子仓库未提交修改及 `.vscode` 状态保留。Isaac Lab 0.34.1，Isaac Sim `4.5.0-rc.36+release.19112.f59b3005.gl`，OSQP 0.6.5，RTX 4060 / 驱动 570.144。实际 USD `/home/chja/myproject/car_no_damper_04.usd` SHA-256 `28c56b286c78df43a6f9d06a9f9393be205f9b3279211b5c05b14b95ee7a5940`；模型 `78fe257a39f108a103c0a0edb3dcc1afb53aff3b861bcac9926cfca7a28da3f3`，MPC `96371980dd186468da813fe339f02838956ef49f9fa19ba8be043c6ef6f1aded`，转向 `16104c2c3febad5921d21d9d1989191cd2ee574c5d8c78325d319af778090ac0`，WheeledLab action `7603e541e56b564f351e85d76e312ddf7a177bf55d9c73190e812b3a8e4e3355`，均与前次基线相同。

仿真控制台仍报告资产中 6 个指向不存在 contact prim 的 FixedJoint 错误，以及 steering/suspension ImplicitActuator 双限值告警；两组均存在，本轮未处理。数据只证明本机这一次配置下的配对结果，跨机器与重复性待验证。
