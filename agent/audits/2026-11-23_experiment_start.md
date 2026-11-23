# 实验起点记录

日期：2026-11-23，Asia/Shanghai。论文第一节点：固定实验起点。

## 结论

本机主仓库 `main` 的 HEAD 与指定参照提交
`db77845751351fe789582db36f7beafec3aaf9a7` 完全一致。
当前双向模型、两份独立数据、四组历史闭环短测均可访问，关键哈希核验通过。
当前默认 MPC 完成一次新的单环境固定指令检查：80 步、全部 solved、零回退。

本实验起点应表述为 **指定提交 + 已登记的本地子仓库差异 + 当前 USD 哈希 + 冻结参数/数据哈希**，
不能仅写成“干净的 GitHub 提交”。WheeledLab 的实际执行文件存在本地修改。
本轮只增加审计材料，没有改变 PPO、MPC 模型/权重、动作、任务、USD 或任何历史结果。
没有重新辨识、训练、安装依赖、提交代码、切换分支、清理工作区或扩大到低附着研究。

## 版本清单

| 项目 | 本机核验结果 |
| --- | --- |
| 仓库路径 | `/home/chja/IsaacLab` |
| 分支 / HEAD | `main` / `db77845751351fe789582db36f7beafec3aaf9a7` |
| 本地远端跟踪引用 | `origin/main` 同一提交；未联网 fetch，不声称核验了远端此刻状态 |
| 提交时间 | 2026-11-22 17:02:06 +0800 |
| Isaac Lab | `0.34.1`，实际加载本仓库 `source/isaaclab/isaaclab/__init__.py` |
| Isaac Sim | `/home/chja/isaacsim`；VERSION 为 `4.5.0-rc.36+release.19112.f59b3005.gl` |
| Python | `/home/chja/miniconda3/envs/env_isaaclab/bin/python`，3.10.16 |
| OSQP Python 包 | `0.6.5` |
| OSQP 原生求解器 | `osqp.OSQP().version()` 返回 `0.6.3`，与 Python 包版本分开登记 |
| OSQP 实际加载位置 | `/home/chja/isaacsim/exts/omni.isaac.core_archive/pip_prebundle/osqp/__init__.py` |
| PyTorch / skrl | `2.5.1+cu124` / `1.4.3` |
| NumPy / SciPy | 1.26.0 / 1.10.1，启动环境发行包元数据 |
| GPU / 驱动 | NVIDIA RTX 4060 / 570.144，由本次控制台输出核对 |

Isaac Sim 没有可用的 `isaacsim` Python distribution 元数据，本记录使用安装目录 VERSION、
实际 Kit 日志路径和运行时路径，不能把缺少 pip 元数据误报成“未安装”。

### 工作区差异

开始时主仓库状态仅有 `.vscode/browse.vc.db-shm` 修改及两个脏子仓库。
`source/`、`scripts/`、`agent/` 的已跟踪文件相对参照提交无差异，无暂存修改。
本轮新增 `agent/audits/2026-11-23_experiment_start*`，其余已有状态保持。

| 子仓库 | gitlink 与实际 HEAD | 未提交改动 |
| --- | --- | --- |
| WheeledLab | `137809a93d21667e30f5a6e98b70ab685b33287f`，一致 | 6 个文件 |
| tdmpc2/tdmpc2 | `8bbc14ebabdb32ea7ada5c801dc525d0dc73bafe`，一致 | 11 个文件，非本次 MPC 控制后端 |

WheeledLab 的 6 个文件分别为 `ackermann_actions.py`、`rss_cfgs.py`、
`custom_video_recorder.py`、`modified_rsl_rl_runner.py`、`wheeledlab_tasks/__init__.py`、
`wheeledlab_tasks/common/actions.py`。完整相对路径、文件 SHA-256 及两份子仓库补丁已存入审计目录。

运行时确认 `RateLimitedCarVWAction.apply_actions` 来自本地
`WheeledLab/source/wheeledlab/wheeledlab/envs/mdp/actions/ackermann_actions.py`。
其差异增加了环境变量控制的轮速调试打印和计数，没有改变该文件的 Ackermann 目标计算公式；
不能据此忽略文件差异。当前文件 SHA-256：
`7603e541e56b564f351e85d76e312ddf7a177bf55d9c73190e812b3a8e4e3355`。

`git submodule status` 实际失败：`.gitmodules` 缺少 `WheeledLab` 映射。
本记录改用 `git ls-tree HEAD` 和各目录 `git rev-parse HEAD/status/diff` 核对，未修复 Git 配置。
因此全新机器仅凭递归 clone 尚不能保证复现本机依赖布局。

## 默认执行配置

默认任务：`Isaac-TractionProtect-CommandFollow-Car4WD-v0`。
Gym 实际入口为 `command_follow_env_cfg:CommandFollowEnvCfg`；PPO agent 入口仍是
`car4wd.agents:skrl_ppo_cfg.yaml`。环境本身不自动选择 MPC；本轮由既有
`scripts/user/eval_nominal_mpc.py` 绕开 policy，加载 `NominalMPC` 默认后端。

以下文件均位于 `source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/`：

| 内容 | 当前实际文件 |
| --- | --- |
| 任务配置 | `config/car4wd/command_follow_env_cfg.py` |
| 双向辨识模型 | `mpc/config/nominal_car4wd_bidirectional_model.json` |
| OSQP 配置 | `mpc/config/nominal_car4wd_bidirectional_osqp_mpc.json` |
| 转向约束 | `mpc/config/nominal_car4wd_steering.json` |
| 独立验证证据 | `mpc/config/nominal_car4wd_bidirectional_validation.json` |

| 参数 | 当前加载值 |
| --- | --- |
| 轮胎直径 / 半径 | 0.130 / 0.065 m |
| 轴距 / 轮距 | 0.411 / 0.290 m |
| 物理仿真 dt / decimation / 控制 dt | 0.01 s / 5 / 0.05 s，即 20 Hz |
| ActionTerm | scale=(1.5, 0.24)，offset=0，clip，no_reverse=False |
| 单步变化限制 | 速度 0.1 m/s，中心转角 0.1 rad，分别来自 2 m/s²、2 rad/s |
| 规划器 / MPC 参考速度 | [-0.5, 1.2] m/s |
| 训练任务期望航向 | [-0.4, 0.4] rad，世界坐标车体航向 |
| ActionTerm 速度 / 转角目标 | [-1.5, 1.5] m/s / [-0.24, 0.24] rad |
| 辨识支持及 MPC 输入范围 | [-1.5, 1.5] m/s / [-0.22, 0.22] rad |
| MPC 预测状态约束 | 航向误差 ±pi，车体速度 ±1.6 m/s，转角 ±0.22 rad |
| OSQP | horizon=20；eps_abs=eps_rel=1e-6；max_iter=4000；polish/adaptive_rho 开启 |
| 代价权重 | Q=[12,5,0.2]，Qf=[24,10,0.4]，R=[0.05,0.1]，Rdu=[0.8,1.5] |
| Policy observation 维度 | 25；本轮没有用其驱动 MPC |

这里的几何是用户确认的理想 Ackermann 配置，不是本轮重新测量 USD 网格所得。
目标范围不是实际车速的硬保证。负速度保持同一个车体航向参考，不加 pi。
运行时 float32 转向尺度为 `0.23999999463558197`，是 0.24 的数值表示，不是参数改变。

实际动作轮关节顺序为 rear-left/right、front-left/right，id `[4,5,9,8]`；
转向为 front-left/right，id `[7,6]`。轮速 observation 顺序是 front-left/right、rear-left/right，
与 action 顺序不同但分别有显式契约，不能按列位置混用。

车辆复用 `isaaclab_assets.CAR_CFG`，实际来自 `robots/car_edit.py`；加载 USD：
`/home/chja/myproject/car_no_damper_04.usd`。

### 冻结文件哈希

| 文件 | SHA-256 |
| --- | --- |
| 双向模型 | `78fe257a39f108a103c0a0edb3dcc1afb53aff3b861bcac9926cfca7a28da3f3` |
| OSQP 配置 | `96371980dd186468da813fe339f02838956ef49f9fa19ba8be043c6ef6f1aded` |
| 转向配置 | `16104c2c3febad5921d21d9d1989191cd2ee574c5d8c78325d319af778090ac0` |
| 实际车辆 USD | `28c56b286c78df43a6f9d06a9f9393be205f9b3279211b5c05b14b95ee7a5940` |

三份配置与四组历史短测快照逐一一致；USD 与四组 metadata 记载一致。
模型还与外置盘 `fit/frozen_parameters.json` 以及独立验证证书中的参数哈希一致。
模型 provenance 指向拟合当时的 `4cb68f...`，当前已归入后续 `db778457...` 提交，
这是来源时间差异，不代表参数被重新拟合。旧前进模型不作为当前默认。

## 数据索引

外置盘根目录记作 `D=/media/chja/CE54D158C95990271/IsaacLabTrainingData`。
逐项读文件核验，没有重新拟合或再次用 validation 选参数。

### 辨识和独立验证

实际目录：`D/traction_protective_control_nominal_identification/20261120_173554/`。

| 文件 | 数据行数，不含表头 | SHA-256 | 核验 |
| --- | ---: | --- | --- |
| identification.csv | 8000 | `323cac91fd9c98384cab59253db715c4d11c5acb46c8bdf67406f36fd36f9870` | 与模型 source_sha256 一致 |
| validation.csv | 5040 | `3258542fde8b5a403a54ef9bf3998bfde813f2055470c0958418aa44fbf35529` | 与独立验证记录一致 |

两份 CSV 路径及哈希不同，split 字段分离；抽检的速度、转角、偏航和执行输入字段无 NaN/Inf。
模型标记 `validation_used_for_parameter_selection=false`，验证记录标记
`parameters_refit_or_selected_on_validation=false`；源码保持 fit/validate 分开读取数据。
这是数据与流程证据核验，不是仅凭“文件不同”证明所有统计独立性。

`manifest.json`、`protocol.json`、`fit/frozen_parameters.json`、辨识指标、验证指标、
两组预测曲线和残差 PNG 均存在。仓库验证证书与外置盘验证报告逐字节哈希一致。
这些数据及本节记录路径未发生迁移，缺失项为零。

历史独立验证 RMSE 保留原口径：速度 0.048338 m/s、中心转角 0.000452 rad、
偏航角速度 0.052580 rad/s。本轮没有将短测跟踪误差冒充模型验证误差。

### 四组闭环短测

实际父目录：`D/traction_protective_control_nominal_mpc/bidirectional_20261120/`。

| 子目录 | steps.csv / predictions.jsonl 行数 | CSV 求解状态 | 回退 | steps.csv SHA-256 |
| --- | ---: | --- | ---: | --- |
| reverse_fixed | 160 / 160 | 160 solved | 0 | `2075aa24a7e3e0bbcfb5d655d3a452ff58715d8b119daada9a14c6afd96f1529` |
| reverse_heading | 400 / 400 | 400 solved | 0 | `810ba99c076826207596f939aa6df83537ef68b058adcda72a0f258a41896d06` |
| signed_joint | 500 / 500 | 500 solved | 0 | `5b5d489a6d046f692900a685e7376dc13bb815c74c3cf5aa88254dc679edd646` |
| signed_speed | 400 / 400 | 400 solved | 0 | `bb9845f561ba7c941e6bdfd9a2acac1db42739798b45c08ef5aed4f2f36370e2` |

共 1460 步；四组 metadata、analysis、stage_metrics、三个配置快照均存在，
配置快照哈希与各自 metadata 一致，结束原因均为 evaluation_horizon。
这些 steps.csv 哈希在本轮登记，不声称 11 月 20 日文本已提供过四个原始 CSV 哈希。
记录路径和实际路径一致，无需迁移映射，无缺失文件。没有重跑、改写旧四组试验。

### 历史 PPO 检查点

指定外置盘任务根目录下发现 16 个 `best_agent.pt`，均已登记完整路径、SHA-256、
训练 env/agent YAML 哈希、动作/几何/命令配置及配置训练步数。
完整附件：[historical_ppo_index.csv](2026-11-23_experiment_start/historical_ppo_index.csv)。
只安全解析 YAML 和计算文件哈希，没有用 pickle 加载检查点，也没有开展 PPO play。

| 组别 | 数量 | 训练转向尺度 | 训练几何 |
| --- | ---: | --- | --- |
| 10 月 5 日至 10 月 10 日历史组 | 10 | 0.8 rad | 半径 0.035、轴距 0.407、轮距 0.23346 m |
| 10 月 29 日 scale_0243 短训与续训 | 2 | 0.243 rad | 同上 |
| 10 月 29 日四种任务 | 4 | 0.24 rad | 同上，仍是旧几何 |

重点路径均为 `D/<任务目录>/<运行名>/checkpoints/best_agent.pt`：

| 任务目录 | 运行名 | SHA-256 |
| --- | --- | --- |
| traction_protective_control_command_follow_steer_scale_0243 | 2026-10-29_11-34-44_ppo_torch_short_steer_scale_0243 | `4e08ecd6c8eec28c9c9dcc0697b94437ed7f0fec995216044ab99b078a124494` |
| traction_protective_control_command_follow_steer_scale_0243 | 2026-10-29_11-38-16_ppo_torch_medium_500_from_short_steer_scale_0243 | `cced224d4d3f5cf2eb87a4619d350fa8a3f274e6bef8d5d8b9ff373b72a01ef9` |
| traction_protective_control_command_follow | 2026-10-29_15-51-35_ppo_torch | `30e7595a2c5c27c07d2e7109bdf44745a2349f256ebe2c48d41568900acfcd02` |
| traction_protective_control_command_follow_multistep_speed | 2026-10-29_16-21-05_ppo_torch | `ccbfb45a16f35de8d599ba1466c8ea572ac9d4c8c7f2de7ae8f351d9d46ac955` |
| traction_protective_control_command_follow_heading_step | 2026-10-29_16-56-02_ppo_torch | `28bd651286d1b38cef8b6aa01b03d19d1d3167ec7d963cb71082539cade07ce8` |
| traction_protective_control_command_follow_multistep_speed_heading | 2026-10-29_19-38-14_ppo_torch | `b8f4df52da56cbb11574ad4292bef54858246d4a9605aadf8653886bc5e34453` |

0243 短训/续训配置为 64 环境、1200/12000 timesteps；四个 0.24 任务配置为
1024 环境、100000 timesteps。它们是配置上限，不是 best_agent 的实际保存步数或完成训练证明。
命令数据均为前进：基础任务 [0.3,1.2]，速度阶跃为正值，航向单独阶跃固定 0.8 m/s。
动作可倒车并不代表已经接受倒车指令训练。
这些文件可作为历史或迁移候选，不能直接认定为当前几何、双向指令的 PPO 基线。

历史配置中的外置盘目录与实际位置一致。仓库 `logs/skrl/` 下的部分同名目录不存在，
不是外置盘检查点丢失，也没有证据证明发生了新的目录迁移；应使用附件中的实际路径。

## 指标口径

令 `v_ref` 为规划速度，`psi_ref` 为世界期望航向，`psi` 为车体航向，
`v_x_body` 为车体系纵向速度，`v_plan = v_world_x*cos(psi_ref)+v_world_y*sin(psi_ref)`。

| 语义 | PPO CSV / 实现 | MPC CSV / 实现 |
| --- | --- | --- |
| 规划速度 | desired_speed | desired_speed_m_s |
| 车体纵向速度 | actual_body_forward_speed | actual_body_vx_m_s |
| 沿规划航向速度 | actual_speed；由已记录的世界 vx/vy 投影 | 未直接记录世界 vx/vy 或 v_plan |
| 原始动作 | raw_action_speed / raw_action_steering，归一化动作 | normalized_action_speed / normalized_action_steering |
| 物理目标，限速率之前 | physical_target_speed / physical_target_steering | mpc_*_target_*、physical_target_* |
| 限速率后输入 | executed_speed / executed_steering | executed_speed_m_s / executed_steering_rad |
| 速度误差 | speed_error = v_plan - v_ref | analyzer 使用 v_ref - v_x_body |
| 航向误差 | heading_error = wrap(psi - psi_ref) | actual_heading_error_rad = wrap(psi_ref - psi) |
| 时间 | command_time_s=k*dt，sample_time_s=(k+1)*dt | 相同；pre_* 是求解前，actual_* 是执行后 |

`executed_speed` 是已执行的速度目标，不是车辆实测速度。
PPO policy observation 的 heading_error 同 MPC 为“期望减实际”，但 PPO 评估 CSV 是相反号。
PPO speed reward 使用 v_plan，不能把它直接和 MPC 的纵向状态误差当成同一指标。

正式比较可选择两种明确口径之一：

1. 车体纵向跟随：两边都计算 `e_v=v_ref-v_x_body`，PPO 使用 actual_body_forward_speed；
   `e_psi=wrap(psi_ref-psi)`，PPO CSV 的 heading_error 取反并 wrap。
2. 规划方向投影跟随：两边都计算 `e_v=v_ref-v_plan`；PPO CSV speed_error 取反。
   MPC 现有 CSV 缺少精确世界速度投影字段，后续正式记录需补齐，不能本轮悄悄替换指标。

平面无侧倾假设下，可由 MPC 字段近似重建
`v_plan = v_x_body*cos(e_psi)+v_y_body*sin(e_psi)`；有 roll/pitch 时不等于精确三维变换，
本记录不把近似值登记为已经观测的真实字段。
MAE/RMSE 不受纯符号翻转影响，但会受速度定义不同影响；稳态有符号偏差必须统一。

两套稳定时间均为首次进入容差并在该阶段剩余样本内保持的时间，默认速度/航向容差
0.05 m/s / 0.05 rad，不是永久稳定性证明。稳态窗口默认最后 1 s，PPO 包含窗口左端点，
MPC 使用严格大于左端点；首阶段超调 PPO 可计算，MPC 返回 null。
PPO 角度超调 wrap，MPC 的超调实际差值未 wrap；跨 pi 比较需先统一。
正式统计应预先确定这些边界规则、阶段长度、重置姿态和种子，不直接拼接现有 stages 表。

辨识误差另行定义：速度/转向是给定实测 executed input 的执行器预测误差；
偏航使用实测 v_x 和中心转角作代数预测。它们既非闭环跟踪误差，也非整车开环航向预测误差。
自动 reset 后的终止样本应剔除或单独标记，不能作为连续响应；本次与四组历史检查未发生该情况。

## 本机运行结果

唯一一次新仿真运行：2026-11-23 14:58，seed=42，num_envs=1，headless，
固定速度 0.6 m/s、航向 0 rad、仿真控制时间 4 s。
通过审计包装器调用**未经修改的** `eval_nominal_mpc.py`，只观察 `gym.make` 返回环境并保存配置。
没有更换控制算法、模型、权重或动作缩放。

新结果目录：
`D/traction_protective_control_baseline_audit/20261123_start_db778457/fixed_check/`。

| 项目 | 真实记录 |
| --- | --- |
| 控制周期 / 步数 | 0.05 s / 80 |
| 求解状态 / 回退次数 | 80 solved / 0 |
| 结束原因 | evaluation_horizon，非 flip、out_of_bounds 或 timeout |
| 平均 / 最大完整 solve 调用耗时 | 27.976 / 30.449 ms；不是实时性保证 |
| 最大物理目标映射误差 | 5.05e-8 |
| 最大限速率前后差 | 2.98e-8 |
| 速度 MAE / RMSE | 0.062439 / 0.155170 m/s，包含起步瞬态 |
| 航向 MAE | 0.001392 rad |
| 最后 1 s 速度偏差 | +0.006527 m/s，期望减实际 |
| steps.csv SHA-256 | `fdb7083806aad925b91180079490ddcc93c7430e5684738c3398a3d38a2a2534` |

以上只证明当前闭环能运行并提供基线核验值，不是优化结果、多种子统计或论文最终控制性能。

“默认配置”需区分任务默认与既有 evaluator 的覆盖项，已分别保存 runtime_defaults.json 和
runtime_environment.json：训练任务默认噪声开、episode=12 s、命令周期=50 s、xy/yaw 小范围随机；
本次 evaluator 默认逻辑关闭噪声、episode=duration+2=6 s、命令周期=duration+10=14 s、
xy/yaw=0、越界范围 ±2000 m，再逐步写入固定参考。控制几何、ActionTerm 和 MPC 参数未覆盖。
AGENTS 中 6–10 s 是推荐起始周期，当前源码实际默认为 50 s，本轮仅登记、不改动。

完整控制台仍有 6 条 contact FixedJoint 指向不存在 prim 的错误消息，以及重复 Gym 注册、
ImplicitActuator 双重限值警告。仿真实际正常完成，但不应记为“没有任何错误日志”。
与历史记录一致，本轮没有修改接触结构、驱动限矩或 USD。

## 待处理差异

1. 跨机器复现尚需处理缺失的 WheeledLab gitmodules 映射及未提交子仓库差异。
   本轮已保存完整补丁、实际源文件哈希和源文件快照，未擅自提交或还原。
2. 指定外置盘索引中没有当前几何且接受双向指令训练的 PPO 检查点；旧文件不是当前公平基线。
   本轮没有验证它们的实际加载兼容性或训练完成步数。
3. PPO 与 MPC 的速度定义、误差符号、稳态窗口端点、超调和重置协议不同，见上节。
   MPC 还使用实测转向关节重建中心角，PPO observation 没有这个同名实测量；论文需交代信息接口。
4. 辨识 manifest 没有采集当时 USD 的文件哈希。当前 USD 与四组闭环记录一致，
   但不能反推辨识采集时 USD 的逐字节身份。此历史溯源缺口明确保留，不伪造补写。
5. 1.2 m/s 的历史短测仍有饱和/稳态欠速；换向有瞬态误差。不得为了“固定起点”改权重或重拟合。
6. GitHub 远端未联网核验，文件已与本地指定 commit 对象核对；当前日期远端分支状态不在本轮结论内。
7. 目标数据、验证证书、四组短测和已索引 16 个检查点无文件缺失或新迁移。
   缺的是当前口径 PPO 基线、部分历史来源证据与干净跨机依赖描述，不是这些既存数据。

## 正式评估条件

| 评估 | 当前判断 |
| --- | --- |
| 当前本机的名义 MPC 固定、速度阶跃、航向阶跃、联合指令，多种子重跑 | 具备启动条件；沿用本记录冻结模型/几何，另建目录并预先登记协议 |
| 倒车与跨零换向名义评估 | 具备启动条件，参考范围保持 [-0.5,1.2]，不把输入余量当成参考范围 |
| 历史辨识/验证结果的离线复核 | 数据和冻结证据齐备；禁止把独立验证集改为调参集 |
| 旧 PPO 检查点在当前几何下的迁移诊断 | 文件及入口齐备，可安排独立加载/短测；结果必须标为迁移，不是原训练基线 |
| 当前几何下 PPO 与 MPC 的论文正式性能对比 | 尚不具备完整条件：缺当前口径 PPO 基线，且公平协议/指标需要先固定 |
| 全新机器仅按 Git commit 一键复现 | 尚不具备，需补齐子模块映射和本地差异/外部资产交付策略 |
| 低附着、风险预测、保护控制、实机能力评估 | 本节点不涉及，也没有由本次短测获得许可或能力证明 |

因此，“固定实验起点”已完成为可审计的本机基线记录，不等于 Phase 1 全部研究验收完成。

## 证据与复核

审计父目录：`D/traction_protective_control_baseline_audit/20261123_start_db778457/`。
包含 inventory.json、runtime_defaults.json、runtime_environment.json、completion.json、
artifact_manifest.json、子仓库补丁、源文件快照、historical_ppo_index.csv、console.log 和 fixed_check。
completion.json 的 `baseline_source_or_data_changed_during_audit=[]`，
已逐一重新哈希源文件、旧 CSV/配置/指标/图像以及索引检查点。

审计脚本：[collect_inventory.py](2026-11-23_experiment_start/collect_inventory.py)、
[run_fixed_check.py](2026-11-23_experiment_start/run_fixed_check.py)、
[finalize_record.py](2026-11-23_experiment_start/finalize_record.py)。

本次实际调用顺序如下；再次复核必须使用新目录，脚本拒绝覆盖已完成审计：

```bash
AUDIT=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_baseline_audit/20261123_start_db778457
SCRIPT=agent/audits/2026-11-23_experiment_start
./isaaclab.sh -p "$SCRIPT/collect_inventory.py" --output-dir "$AUDIT"
./isaaclab.sh -p "$SCRIPT/run_fixed_check.py" --audit-dir "$AUDIT" > "$AUDIT/console.log" 2>&1
./isaaclab.sh -p scripts/user/analyze_nominal_mpc.py "$AUDIT/fixed_check"
./isaaclab.sh -p "$SCRIPT/finalize_record.py" --audit-dir "$AUDIT"
```

审计包装器实际传给既有 evaluator 的参数：
`--headless --profile fixed --speed 0.6 --heading 0.0 --duration 4 --seed 42 --output_dir <AUDIT>/fixed_check`。
运行与分析均 exit code 0；新增脚本语法检查通过，`git diff --check` 通过。
