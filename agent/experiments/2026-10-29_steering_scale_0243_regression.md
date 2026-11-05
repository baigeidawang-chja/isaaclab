# Steering Scale 0.243 Regression

## Goal

Verify the revised `0.243 rad` steering action contract through policy-free
execution, PPO smoke/short training, and the four existing CSV evaluations.

## Hypothesis

The revised action range removes Ackermann target saturation and remains
learnable without introducing a directional execution bias.

## Baseline

- Task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`
- Algorithm: skrl MLP-PPO
- Action scale: `(1.5 m/s, 0.243 rad)`
- Seed: 42
- Training environments: 64
- External run root:
  `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow_steer_scale_0243`
- No reward, observation, command, rate-limit, or PPO hyperparameter was
  changed during this regression.

## Commands

```bash
./isaaclab.sh -p scripts/user/command_follow_open_loop.py \
  --speed 0.8 --steer 0.22 --duration 8 --warmup 3 \
  --output /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/diagnostics/open_loop_scale_0243_20261029.csv \
  --headless --disable_fabric

NUM_ENVS=4 MAX_ITERATIONS=2 ./scripts/agent_smoke_test.sh \
  agent.agent.experiment.directory=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow_steer_scale_0243 \
  agent.agent.experiment.experiment_name=smoke_steer_scale_0243 \
  agent.agent.experiment.wandb=False

NUM_ENVS=64 MAX_ITERATIONS=50 ./scripts/agent_train_short.sh \
  agent.agent.experiment.directory=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow_steer_scale_0243 \
  agent.agent.experiment.experiment_name=short_steer_scale_0243 \
  agent.agent.experiment.wandb=False

NUM_ENVS=64 MAX_ITERATIONS=500 ./scripts/agent_train_short.sh \
  --checkpoint /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow_steer_scale_0243/2026-10-29_11-34-44_ppo_torch_short_steer_scale_0243/checkpoints/best_agent.pt \
  agent.agent.experiment.directory=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow_steer_scale_0243 \
  agent.agent.experiment.experiment_name=medium_500_from_short_steer_scale_0243 \
  agent.agent.experiment.wandb=False
```

The four task-specific evaluators were run with `--headless --disable_fabric`
and the corresponding `best_agent.pt`.

## Open-Loop Result

- CSV rows: 480; all numeric values finite.
- Unexpected termination/truncation: 0.
- Zero-steer heading rate: `-0.000652 rad/s`.
- Zero-steer mean yaw rate: `-0.004636 rad/s`.
- Positive/negative yaw-rate magnitude ratio: `1.03441`.
- Maximum actual steering-joint magnitude: `0.235828 rad`.

The execution layer passes the predefined less-than-5% yaw-symmetry gate and
does not show sustained one-sided zero-steer drift.

## Training Result

### 50-iteration run

- Run: `2026-10-29_11-34-44_ppo_torch_short_steer_scale_0243`
- Value loss: `1.25175 -> 0.00290`.
- Policy loss: `0.00703 -> -0.00978`; oscillatory as expected for PPO.
- Signed entropy loss: `-0.0009187 -> -0.0008963`.
- Mean instantaneous reward: `0.02843 -> 0.05535`.
- Mean episode return: `11.94 -> 12.84`.
- Non-finite policy fraction: 0.

### 500-iteration continuation

- Run: `2026-10-29_11-38-16_ppo_torch_medium_500_from_short_steer_scale_0243`
- Value loss remained finite in `0.00234..0.01043` and ended at `0.00648`.
- Policy loss remained finite in `-0.00992..0.01356`; no monotonic decrease.
- Signed entropy loss moved from `-0.0008895` to `-0.0004933`.
- Mean instantaneous reward: `0.06775 -> 0.10212`.
- Mean episode return: `16.97 -> 23.59`.
- Non-finite policy fraction: 0.

The longer run improves reward, but the three losses do not all follow a
monotonic decreasing-and-stable pattern. No such claim is made.

## Evaluation Result

| Evaluation | Speed MAE 50 / 500 (m/s) | Heading MAE 50 / 500 (rad) | Heading steady abs. 50 / 500 (rad) | Raw steering saturation at 500 | Heading stages settled at 500 |
|---|---:|---:|---:|---:|---:|
| Heading step | 0.262 / 0.082 | 0.380 / 0.269 | 0.419 / 0.207 | 45.4% | 1 / 6 |
| Speed step | 0.255 / 0.081 | 0.171 / 0.134 | 0.188 / 0.117 | 40.1% | 3 / 6 |
| Speed + heading step | 0.299 / 0.089 | 0.283 / 0.225 | 0.311 / 0.177 | 38.4% | 2 / 6 |
| CommandFollow | 0.207 / 0.067 | 0.397 / 0.264 | 0.655 / 0.560 | 45.6% | 0 / 1 |

- All evaluation values are finite.
- No evaluation terminated or truncated unexpectedly.
- Positive and negative raw-steering saturation fractions are nearly balanced.
- Raw steering changes sign on 52% to 57% of control steps.
- Mean absolute physical-target steering change is `0.219..0.243 rad/step`.
- Mean absolute rate-limited executed steering change is
  `0.085..0.088 rad/step`.
- Maximum Ackermann target stays at approximately `0.260795 rad`.
- Maximum measured steering reached `0.261911 rad`, approximately
  `0.000112 rad` above the nominal joint limit due to tracking overshoot.

## Result

The physical action mapping is symmetric and mechanically consistent at the
tested open-loop point. PPO learning improves speed and heading errors with
additional training, but the 500-iteration checkpoint is not an acceptable
Phase 1 baseline because heading settling remains weak and deterministic raw
steering exhibits high-frequency, nearly symmetric boundary-to-boundary
chatter.

## Conclusion

- Open-loop action-contract hypothesis: supported.
- Usable learned CommandFollow baseline: not yet supported.
- Do not change heading reward solely to hide this behavior.
- Do not freeze or promote the 500-iteration checkpoint as the Phase 1
  baseline.
- Next isolate deterministic policy-output sensitivity to observation terms
  and quantify action chatter before deciding between more training, policy
  output regularization, observation scaling, or action-rate changes.
