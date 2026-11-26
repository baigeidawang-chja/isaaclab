# 高附着平地名义 MPC 纵向实验记录

日期：2026-11-24，Asia/Shanghai。范围：论文 2.4.1–2.4.3，纯 Isaac Lab 仿真。

## 目标、假设与结论

目标是测量冻结名义 MPC 在定速、升降速、停车及直接正反换向时的真实响应，
区分参考速度、物理目标、限速率执行目标和车体速度，而不是调参使结果符合预期。
待检验假设：当前固定控制器可以在全部指定速度/进入条件下，在 8 s 内进入并持续保持
目标 ±0.05 m/s。**该全范围假设不成立**：F7、S1 的升速进入 0.9 m/s 段、S4 首段
在 8 s 观察窗内不满足判据；其他部分阶段的带内保持仅发生在末尾几拍。

已实际完成 3 次短测、12 次正式运行和 2 次补充长保持，共 17 次、6,160 控制步。
正式矩阵为 29 段、232 s、4,640 步，全部 solved，零回退、零提前终止、零自动复位。
没有改模型、MPC 权重、动作限制、车辆、PPO、奖励、注册或检查点，没有生成替代实验数据。
本机已验证，跨机器待验证。单个工况只有一次正式完整运行，不报告置信区间或多随机试验结论。

## 实验起点

主仓库实际分支 `main`，HEAD 为 **`329c8033b8252f1186fc0e327f15e4bfd8fe137e`**，
以 `provenance/inventory.json` 为机器可核验来源。
参照实验提交为 `db77845751351fe789582db36f7beafec3aaf9a7`；329c8033b8 增加前次审计材料。
本次启动前保留 WheeledLab 和 tdmpc2/tdmpc2 两个脏子仓库；运行期间出现的
`.vscode/browse.vc.db-shm` 修改也保留，没有复位或清理工作区，没有提交/推送。

任务为 `Isaac-TractionProtect-CommandFollow-Car4WD-v0`；默认 NominalMPC/OSQP，绕开 PPO。
模型、OSQP 和转向文件位于 `traction_protective_control/mpc/config/`：

| 内容 | 文件 / SHA-256 |
| --- | --- |
| 双向模型 | `nominal_car4wd_bidirectional_model.json`；`78fe257a39f108a103c0a0edb3dcc1afb53aff3b861bcac9926cfca7a28da3f3` |
| OSQP | `nominal_car4wd_bidirectional_osqp_mpc.json`；`96371980dd186468da813fe339f02838956ef49f9fa19ba8be043c6ef6f1aded` |
| 转向 | `nominal_car4wd_steering.json`；`16104c2c3febad5921d21d9d1989191cd2ee574c5d8c78325d319af778090ac0` |
| 实际 USD | `/home/chja/myproject/car_no_damper_04.usd`；`28c56b286c78df43a6f9d06a9f9393be205f9b3279211b5c05b14b95ee7a5940` |
| WheeledLab 实际 action | `WheeledLab/source/wheeledlab/wheeledlab/envs/mdp/actions/ackermann_actions.py`；`7603e541e56b564f351e85d76e312ddf7a177bf55d9c73190e812b3a8e4e3355` |

上述冻结文件与 11 月 23 日记录一致，运行前后全部源文件/配置哈希核验无变化。
WheeledLab 的既有补丁含环境变量控制的轮速打印及计数，没有改 Ackermann 公式，
完整补丁保存在 `provenance/WheeledLab.patch`；其他子仓库差异也已存档。
辨识集与独立验证集仍为外置盘 `traction_protective_control_nominal_identification/20261120_173554/`，
本轮只重新哈希，没有拟合或选择参数：identification SHA-256
`323cac91fd9c98384cab59253db715c4d11c5acb46c8bdf67406f36fd36f9870`，validation
`3258542fde8b5a403a54ef9bf3998bfde813f2055470c0958418aa44fbf35529`。
**辨识采集当时的历史 USD 哈希缺失**，本轮当前 USD 哈希不能补作历史证据。

运行时：Python `/home/chja/miniconda3/envs/env_isaaclab/bin/python`，Isaac Lab 0.34.1，
Isaac Sim `4.5.0-rc.36+release.19112.f59b3005.gl`，OSQP Python 0.6.5 / native 0.6.3，
Torch 2.5.1+cu124、skrl 1.4.3、NumPy 1.26.0、SciPy 1.10.1；RTX 4060 / 驱动 570.144。
OSQP 实际来自 Isaac Sim 的 `omni.isaac.core_archive/pip_prebundle/osqp`，没有新装依赖。

## 冻结配置与采集检查

轮半径 0.065 m、轴距 0.411 m、轮距 0.290 m；理想 Ackermann 几何。
物理 dt=0.01 s、decimation=5、控制周期 0.05 s。静/动摩擦为 0.8/0.6，无障碍物、无低附着区。
参考速度 [-0.5,1.2] m/s；ActionTerm 物理速度 [-1.5,1.5] m/s，转向 scale=0.24 rad；
MPC 输入速度同上、中心转角 [-0.22,0.22] rad；两轴变化率均为 2 单位/s，单拍限幅 0.1。
OSQP horizon=20，Q=[12,5,0.2]、Qf=[24,10,0.4]、R=[0.05,0.1]、Rdu=[0.8,1.5]，未调参。

沿用 evaluator 的覆盖：单环境、headless、观测噪声关闭、世界期望航向 0、xy/yaw 复位为 0，
episode=duration+2 s，自动指令重采样间隔=duration+10 s，越界 ±2000 m。
所有运行初始根状态为位置约 `[0,0,0.38] m`、单位四元数、六维速度为 0，执行目标为 0。
**0.38 m 是资产初始 z 与原复位 z 偏移相加的实际高度，不是新改参数**；没有增加接触预沉降。
因此首段指标包含原起点的启动/接触建立瞬态，不能称为从已接触稳态起步。
初始 heading 数值约 -2.91e-13 rad，属于单位姿态计算舍入，核验容差 1e-9 rad。

本次只增加执行后 `actual_world_vx_m_s/actual_world_vy_m_s`、投影速度、世界位置、
执行前位置/航向/上一拍执行目标、初态、关节 ID、配置/版本快照及 reset 标记。
`actual_projected_speed_m_s = vx_world*cos(psi_ref)+vy_world*sin(psi_ref)`；全部 6,160 拍
psi_ref=0，逐拍投影与世界 vx 差为 0，倒车不取绝对值。旧 CSV 没有世界速度时投影指标不可用，
不从 body vx 近似补造。新主指标仍使用执行后车体 vx。

| 日志轮位置 | 实际 joint name | ID |
| --- | --- | ---: |
| FL | joint_front_left_wheel_link_wheel | 9 |
| FR | joint_front_right_wheel_link_wheel | 8 |
| RL | joint_back_left_wheel_link_wheel | 4 |
| RR | joint_back_right_wheel_link_wheel | 5 |

ActionTerm 输出顺序 RL/RR/FL/FR，对应 `[4,5,9,8]`；两个前轮转向 L/R 对应 `[7,6]`。
采集按关节名读取 target、actual、computed torque、applied torque，没有按 tensor 位置混配。
用现有理想公式逐拍重算，轮速目标最大差 5.99e-6 rad/s，转角目标最大差 1.34e-8 rad。
所有轮目标与执行速度符号一致。F1 最后 1 s 四轮实测均值为
[-8.0444,-7.3706,-7.2214,-7.2155] rad/s；F5 为 [5.9636,6.5005,9.5318,7.6293] rad/s，
整体方向正确，但瞬时轮速可反号。例如 F1 前左/前右最后 20 拍同向比例 85%/95%，
F5 为 75%/85%，不能声称每拍实测轮速都同向。

本拍 pre_body_speed 与上一拍 actual_body_vx 差为 0；previous_executed 与上一拍 executed 差为 0。
MPC 物理目标至 ActionTerm 目标最大误差 1.04e-7 m/s，限速率前后最大差 1.19e-7 m/s。
两个前轮的目标和实际角均已逐拍保存；每段实际转角末段均值在两张主表中。
computed/applied torque 是**最后一个物理子步的执行器量**，不是 0.05 s 平均扭矩，亦不是接触力测量。
执行后 reset 状态显式标无效；分析在首个终止/reset 前截止。本次没有此类样本。

## 运行目录与完成索引

本次新建外置盘根目录，以下称 `D`：

```text
/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_mpc_longitudinal/20261124_chapter2
```

每个 `D/<phase>/<ID>/` 都有 `steps.csv`、`metadata.json`、`predictions.jsonl`、
model/mpc/steering 三份 JSON 快照、`runtime_environment.json`、分析 JSON/CSV。
相邻 `<ID>.launch.json` 保存实际 argv 与 shell 命令，`<ID>.console.log` 保存完整日志，
`<ID>.result.json` 保存退出码、墙钟耗时、预期/实际步数及核验项。以下路径均相对 D：

| ID | 参考序列 m/s | 段数 | 时长 s | 实际步数 | 原始目录 |
| --- | --- | ---: | ---: | ---: | --- |
| F1 | -0.5 | 1 | 8 | 160 | formal/F1 |
| F2 | -0.25 | 1 | 8 | 160 | formal/F2 |
| F3 | 0 | 1 | 8 | 160 | formal/F3 |
| F4 | 0.3 | 1 | 8 | 160 | formal/F4 |
| F5 | 0.6 | 1 | 8 | 160 | formal/F5 |
| F6 | 0.9 | 1 | 8 | 160 | formal/F6 |
| F7 | 1.2 | 1 | 8 | 160 | formal/F7 |
| S1 | 0.3,0.6,0.9,1.2,0.9,0.6,0.3 | 7 | 56 | 1120 | formal/S1 |
| S2a | 0.6,0.3,1.2 | 3 | 24 | 480 | formal/S2a |
| S2b | 0.6,0.9,1.2 | 3 | 24 | 480 | formal/S2b |
| S3 | 0.5,-0.5,0.5,0,-0.5,0 | 6 | 48 | 960 | formal/S3 |
| S4 | 1.2,-0.5,1.2 | 3 | 24 | 480 | formal/S4 |

上表全部完成；各末段和 metadata 指令序列逐一一致，正式统一 seed=42，各段连续不复位。
所有结束原因为 `evaluation_horizon`，不是 timeout、flip 或 out_of_bounds。没有仿真失败重试。
完整索引和文件哈希：`analysis/run_index.csv/json`、`analysis/artifact_sha256.json`。

### 4 秒短测

B1/B2 seed42，B3 seed43，均 0.6 m/s、0 rad、80 步；路径 `smoke/B1`、`B2`、`B3`。
三次 MAE=0.062438741、RMSE=0.155170482、末 1 s 偏差=+0.006527165 m/s，
与 11 月 23 日短测一致，逐拍实测车速最大差为 0。
相互比较初态、参考、实测速度/航向、物理速度目标和执行动作的最大差均为 0，
三次只是技术重复，不把换 seed 的相同物理轨迹当成新的独立随机样本。
耗时单列：B1/B2/B3 全进程墙钟约 7.340/7.336/7.343 s；完整 solve 中位数
27.648/28.017/27.941 ms，P95 28.242/28.422/28.501 ms，最大 30.501/28.895/28.747 ms。

## 统一指标口径

指令及本拍输入在 `command_time_s=k*0.05` 生效，车辆响应采样于 `sample_time_s=(k+1)*0.05`。
用指令时间划分阶段，速度误差 `e=v_ref-vx_body_post`，航向误差 `wrap(psi_ref-psi_post)`。
MAE=mean(abs(e))，RMSE=sqrt(mean(e²))；末段偏差为最后 1 s 的有符号均值。
窗口为 `(T-1,T]`，时间按 1e-9 s 精度处理浮点端点，全部完整段为 20 点。
速度与投影误差的单位为 m/s，航向指标单位为 rad。

稳定时间为首个使余下全部采样都满足 abs(e)<=0.05 的**执行后**相对时刻，不平滑、不忽略反复越界。
末尾连续带内跨度为 `(带内末尾点数-1)*dt`；只有最后一点合格时跨度为 0，
例如稳定时间 8.00 s 不代表可靠的持续稳定。末点不合格时记未稳定/null。
超调=max(0,max(sign(v_new-v_old)*(v_actual-v_new)))，单位 m/s；首段无前序参考，记 N/A。
停车段另报末 1 s 世界 x 净位移及平面路径长度，单位 m，使用 T-1 的 pre 位置与后续20个 post 位置；
不把速度均值当成位置漂移，也不由 20 Hz 瞬时速度强行反推位置。

直接正负换向要求连续3拍沿新方向超过0.02 m/s，取这3拍首拍。
执行目标按 command_time 计时，实测速度按 sample_time 计时，二者相减即采样定义下的车辆附加延迟。
确认本身需等后两拍到达，但不把这0.1 s再加进首拍时刻；停车不用换向判据。
若把两个字段都放在 post 时间上，执行目标时刻会晚0.05 s，附加延迟相应小0.05 s，不混用。

物理速度目标触 ±1.5 m/s 的容差为1e-4，分别统计全段/末1s、上限/下限；
执行目标变化 abs(u_k-u_prev)>=0.1-1e-6 视为触变化率限幅。
完整 solve 指标直接使用 `solve_time_s`，不是只计 OSQP 内核的 `solver_run_time_s`。
两张主表包含所有这些指标以及逐轮统计：`analysis/fixed_results.csv`、`analysis/directed_step_results.csv`。
阶段编号使用0起始，首段仍保留，但不假装为有已知前序参考的阶跃。

## 2.4.1 定速结果

| 工况 | 参考 | 末1s实测均值 | MAE | RMSE | 末1s偏差 | 稳定时间 s | 尾部带内跨度 s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | -0.50 | -0.492215 | 0.034011 | 0.092574 | -0.007785 | 7.15 | 0.85 |
| F2 | -0.25 | -0.249865 | 0.018710 | 0.046389 | -0.000135 | 3.95 | 4.05 |
| F3 | 0 | 0.000860 | 0.001910 | 0.002753 | -0.000860 | 0.05 | 7.95 |
| F4 | 0.30 | 0.296975 | 0.019935 | 0.053056 | +0.003025 | 0.30 | 7.70 |
| F5 | 0.60 | 0.576407 | 0.044675 | 0.113214 | +0.023593 | 7.45 | 0.55 |
| F6 | 0.90 | 0.898558 | 0.067957 | 0.176928 | +0.001442 | 7.60 | 0.40 |
| F7 | 1.20 | 1.147233 | 0.111979 | 0.257774 | +0.052767 | 未稳定 | 0 |

除时间列外速度量均为 m/s。图1：`analysis/figures/01_fixed_speed.png/pdf`，含等值线。
在本次8s内，F1–F6满足有限窗口的末尾进入判据，但只有 -0.25、0、0.3 m/s 展现数秒持续带内区间；
-0.5、0.6、0.9 m/s 的末段均值接近参考，不能据此声称全程或长期稳定。
F7 有持续欠速、未稳定。七次定速航向 MAE 为0.000101–0.001490 rad，最大绝对航向误差不超过0.006658 rad。

## 2.4.2 升降速与进入条件

图2：`analysis/figures/02_S1_tracking.png/pdf`，同图记录参考、MPC物理目标、执行目标、实测车速。
MPC目标与执行目标几乎重合是因为 QP 已施加与 ActionTerm 一致的变化率限制，不是漏画曲线。

| S1 有向进入 | MAE m/s | 末1s偏差 m/s | 稳定时间 s | 尾部带内跨度 s | 超调 m/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.3→0.6 | 0.025024 | +0.000079 | 7.45 | 0.55 | 0.060717 |
| 0.6→0.9 | 0.030372 | +0.003453 | 未稳定 | 0 | 0.085969 |
| 0.9→1.2 | 0.060817 | +0.050942 | 7.85 | 0.15 | 0.011418 |
| 1.2→0.9 | 0.030164 | -0.017092 | 8.00 | 0 | 0.089996 |
| 0.9→0.6 | 0.021417 | +0.006660 | 7.85 | 0.15 | 0.074599 |
| 0.6→0.3 | 0.018586 | -0.000251 | 6.40 | 1.60 | 0.052425 |

从升速与降速进入0.6 m/s，MAE分别为0.025024/0.021417，但严格稳定时间为7.45/7.85 s；
进入0.9 m/s的MAE接近，但升速未稳定，降速只在末点入带。这表明瞬态均值和末尾稳定并非同一指标。
本轮不滤除原始车速波动，不把首次靠近目标的时间替代用户规定的稳定时间。

### S2a 与 S2b 第三段

图3：`analysis/figures/03_S2_entry_conditions.png/pdf`，相对切换时刻对齐至0，附输入饱和时段。

| 指标 | S2a: 0.3→1.2 | S2b: 0.9→1.2 |
| --- | ---: | ---: |
| 切换前实测车速 m/s | 0.298539 | 0.948292 |
| 上一拍执行目标 m/s | 0.324165 | 1.010984 |
| 切换前航向 rad | -0.000239 | +0.001282 |
| 全段 MAE / RMSE m/s | 0.080482 / 0.156237 | 0.054183 / 0.072877 |
| 末1s偏差 m/s | +0.035695 | +0.042437 |
| 稳定时间 / 尾部跨度 s | 7.35 / 0.65 | 7.90 / 0.10 |
| 超调 m/s | 0.012603 | 0.003184 |
| 上限触及比例，全段 / 末1s | 71.875% / 85% | 81.875% / 95% |

S2b 的全段误差较小，但不是“历史状态单因素使控制更好”的证据：前序速度和阶跃幅值同时改变。
结论仅为**两种进入条件的响应差异**；每种完整序列正式 n=1，同一序列各拍、各阶段不是独立重复。

### 单独归档的16秒保持

采集前已登记趋势判据：1.2 m/s 段末2s速度OLS斜率绝对值>0.01 m/s²，
或最后1s与之前1s均值差绝对值>0.02 m/s。S2a最后段斜率 -0.013146、S4最后段 +0.024082 m/s²，
触发延长；S4最后两个1s均值差还达到 +0.023419 m/s。精确值保存在 `analysis/all_stage_metrics.csv`。
其他1.2段未触发，不按希望的结论追加实验。
`supplement/L_S2a_2` 与 `supplement/L_S4_2` 各32s/640步，前两段仍各8s，最后一段16s。
它们的前480步与原正式运行逐拍物理轨迹完全一致，保留了相同进入条件。

| 最后一段 | 16s末1s偏差 m/s | 稳定时间 s | 尾部跨度 s | 全段/末1s上限触及 |
| --- | ---: | ---: | ---: | ---: |
| L_S2a_2 | +0.026199 | 15.05 | 0.95 | 79.6875% / 75% |
| L_S4_2 | +0.069417 | 未稳定 | 0 | 83.75% / 100% |

延长后仍有波动/趋势，不能宣称已收敛；补充运行不替换原8s主表，也不继续无限延长至获得好结果。
2次补充合计64s/1280步，包括重复的前序阶段；没有把它们并入正式矩阵计数。

## 2.4.3 换向与停车

图4：`analysis/figures/04_reversals_wheels_torques.png/pdf`，并列S3/S4车速、四轮轮速和计算/施加扭矩。

| 有向换向 | 执行目标过阈时间 s | 实测速度过阈时间 s | 车辆附加延迟 s |
| --- | ---: | ---: | ---: |
| S3 0.5→-0.5 | 0.25 | 0.35 | 0.10 |
| S3 -0.5→0.5 | 0.25 | 0.35 | 0.10 |
| S4 1.2→-0.5 | 0.75 | 0.90 | 0.15 |
| S4 -0.5→1.2 | 0.25 | 0.40 | 0.15 |

这四次执行目标到过阈之前均沿新方向以约0.1 m/s/拍变化；S4降速从上一拍目标1.5 m/s开始，
所以这段输入过零过程较长。该过程已发生在带变化率约束的MPC输出中，
不能误写成“ActionTerm又额外造成同样延迟”；两者差仅float32舍入量级。
总过零延迟由0.25–0.75s的受限输入变化过程和0.10–0.15s的采样车辆附加响应构成，
不是重新辨识出的纯物理时延常数。S4两次换向严格稳定时间分别为5.50s和7.50s，
后者尾部只有0.50s带内，过零快不等于长期稳定。

S3 0.5→0 的稳定时间为0.35s，-0.5→0 为1.00s；尾部带内跨度分别7.65/7.00s。
末1s世界x净位移分别 +4.768e-6 m 和 +1.490e-8 m；末1s平面累计路径约
0.000383 m 和0.000027 m。它们是离散位置记录的漂移，不是实车里程计精度或传感器指标。

## 1.2 m/s 执行现象分析

以F7最后1s为例，参考1.2、实测均值1.147233 m/s，MPC上限触及85%；全8s上限触及76.25%。
下限触及为0。F7速率限制触及比例为9.375%，而末段欠速仍伴随上限触及，
不能仅用起步的速率限制解释全部末段误差。

| 轮 | 目标均值 rad/s | 实测均值 rad/s | 计算扭矩绝对均值 N m | 施加扭矩绝对均值 N m | 裁剪比例 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FL | 23.0464 | 15.6390 | 3.6607 | 0.9590 | 85% |
| FR | 23.0474 | 16.4153 | 3.9328 | 0.9695 | 85% |
| RL | 23.0463 | 15.4577 | 3.1122 | 0.9809 | 90% |
| RR | 23.0473 | 15.3167 | 3.1626 | 0.9663 | 75% |

实际轮驱动仍为原DCMotor：effort_limit=1.0 N m、saturation_effort=1.05 N m、damping=0.5。
裁剪判据为 abs(computed)>abs(applied)+1e-4 N m，同轮同一日志拍对齐；完整数据含逐轮误差MAE。
这些证据支持“物理输入上限、执行器裁剪与轮级跟随误差同时出现”，
但没有通过改变单个物理因素做因果排除，不认定某一因素为唯一原因，更不据此生成滑移真值。
车体速度与轮速均为每控制周期末的瞬时样本，轮级高速波动及子步采样应作为解释限制。

## 求解、抽查与可复现性

正式4,640次完整solve：中位数27.739ms，P95=28.422ms，最大41.714ms，超过50ms为0次；
4640 solved、0 fallback。全部17次运行也无回退/提前终止。
这是本机本批次耗时统计，不是硬实时保证；包含建模/组装/求解等完整solve流程，不包含环境step。
逐运行耗时和状态在 `analysis/run_summary.csv/json`，全进程耗时在运行索引。

`analysis/raw_spot_checks.json` 独立用stdlib从原始CSV重算F7整段、S4换向第二段、S3停车第四段：
MAE、RMSE、最后20点偏差与主表差<1e-12，另核对F7饱和比例、S4三拍过零与附加延迟、S3位置漂移。
保留窗口边界行和过零前一拍/连续三拍。`analysis/collection_verification.json` 包含每次映射误差、
初态、告警、扩展前缀差异、未变源文件和辨识数据哈希。7个纯指标单元测试通过；测试夹具不是实验数据。

实际启动顺序如下（本次D已经存在且有结果，批量入口拒绝覆盖；复跑须换新目录）：

```bash
D=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_mpc_longitudinal/20261124_chapter2
./isaaclab.sh -p scripts/user/run_nominal_mpc_longitudinal.py --root "$D" --phase smoke
./isaaclab.sh -p scripts/user/run_nominal_mpc_longitudinal.py --root "$D" --phase formal
./isaaclab.sh -p scripts/user/analyze_nominal_mpc_longitudinal.py --root "$D"
./isaaclab.sh -p scripts/user/run_nominal_mpc_longitudinal.py --root "$D" --phase supplement --case-json "$D/analysis/suggested_supplements.json"
./isaaclab.sh -p scripts/user/analyze_nominal_mpc_longitudinal.py --root "$D"
./isaaclab.sh -p scripts/user/verify_nominal_mpc_longitudinal.py --root "$D"
./isaaclab.sh -p scripts/user/test_nominal_mpc_longitudinal.py
```

同一4s基准的单次复核方法，`NEW_RUN`必须尚不存在：

```bash
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py \
  --task Isaac-TractionProtect-CommandFollow-Car4WD-v0 --headless \
  --profile fixed --speed 0.6 --heading 0 --duration 4 --segment_duration 4 \
  --seed 42 --output_dir "$NEW_RUN"
./isaaclab.sh -p scripts/user/analyze_nominal_mpc.py "$NEW_RUN"
```

各正式命令均显式传入整次duration及segment_duration=8，完整argv在每次launch记录。
表图重算入口接受任意迁移后的 `--root`，优先验证运行目录内配置快照，不要求旧绝对配置路径仍存在。
离线重算不宣称加载了本机现存USD；当前USD身份另由采集元数据与本轮核验记录证明。
运行环境配置仍含历史绝对资产路径，跨机器执行需要真实提供同一资产/依赖，不只是迁移CSV。

## 告警、限制与文件变更

所有运行仍出现6条 contact FixedJoint 指向不存在prim的Error；此外有重复Unitree Gym注册、
ImplicitActuator双重限值、非NVIDIA GPU跳过及弃用提示。它们与基准日志一致，
仿真实际成功但不能写“无错误日志”。本轮遵守范围，没有修改接触结构、驱动限矩或USD。
WheeledLab缺失.gitmodules映射和本地补丁仍是新机器复现差异；没有联网fetch核验远端新状态。
初态检查曾因将 -2.91e-13 rad 与0严格比较而失败，查明单位四元数后仅调整核验数值容差；
另外一次在矩阵未结束时尝试分析，被完整性检查拒绝。这些不是仿真失败或替换运行。

修改：`scripts/user/eval_nominal_mpc.py`（仅日志），`scripts/user/analyze_nominal_mpc.py`
（可重定位快照读取、旧投影字段可用性）。新增批量入口、纵向分析器、采集核验器、7项指标测试及本文。
采集前源码/补丁在 `provenance/sources/`、`*.patch`；分析最终源码及哈希在 `provenance/final_sources/`。
没有修改任务或控制参数。后续可基于这些原始证据决定是否专门研究执行器波动/饱和及改进控制，
但当前结论不延伸到低附着、实车传感器、PPO对比或保护控制。
