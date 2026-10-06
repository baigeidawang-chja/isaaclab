# Experiment

## Goal

Determine whether a conservative fixed learning rate improves PPO loss stability for the Phase 1 CommandFollow task.

## Hypothesis

Replacing the rapidly varying KL-adaptive learning rate with a fixed `3.0e-4` rate will reduce policy-loss spikes while preserving the low, stable value loss produced by timeout bootstrapping and allowing policy standard deviation to decline smoothly.

## Baseline

- Commit / working tree: dirty working tree; preserve unrelated user changes
- Task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`
- Config: skrl MLP-PPO, 64 environments, 50 iterations, 24-step rollouts, `time_limit_bootstrap: true`
- Checkpoint: `2026-10-05_20-18-43_ppo_torch/checkpoints/agent_1200.pt`
- Log directory: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-05_20-18-43_ppo_torch`
- Relevant baseline metrics:
  - value loss first-5 / last-5 mean: 0.7801 / 0.02766
  - value loss minimum / maximum: 0.00855 / 1.1486
  - policy loss minimum / maximum: -0.01399 / 0.05960
  - policy loss first-5 / last-5 mean: -0.00564 / -0.00631
  - entropy loss: -0.0009206 at step 24, -0.0008431 at step 1176
  - policy standard deviation: 0.60837 at step 24, 0.56033 at step 1176
  - adaptive learning rate range: approximately `8.8e-5` to `1.0e-3`
  - no non-finite loss samples

## Change

- Files: `traction_protective_control/config/car4wd/agents/skrl_ppo_cfg.yaml`
- Parameters:
  - `learning_rate: 1.0e-3 -> 3.0e-4`
  - `learning_rate_scheduler: KLAdaptiveLR -> null`
  - `learning_rate_scheduler_kwargs: {kl_threshold: 0.01} -> null`
- Reason: remove order-of-magnitude optimizer step-size changes as a source of policy-loss spikes.

## Command

```bash
./scripts/agent_smoke_test.sh
./scripts/agent_train_short.sh
MAX_ITERATIONS=200 ./scripts/agent_train_short.sh
CHECKPOINT=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-05_20-44-17_ppo_torch/checkpoints/best_agent.pt ./scripts/agent_eval.sh
```

## Run

- Start: 2026-10-05 20:38
- End: 2026-10-05 20:54
- Log directory:
  - smoke: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/agent_runs/smoke_20261005_203851`
  - 50-update console: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/agent_runs/short_20261005_204008`
  - 50-update skrl run: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-05_20-40-13_ppo_torch`
  - 200-update console: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/agent_runs/short_20261005_204413`
  - 200-update skrl run: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-05_20-44-17_ppo_torch`
  - evaluation: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/agent_runs/eval_20261005_205008`
- Exit code: 0 for smoke, both training runs, and evaluation wrapper
- Evaluation note: `agent.env` overrode the attempted temporary 30-second timeout with 300 seconds; the process group was stopped with `SIGINT` after successful sustained execution, which the wrapper handles as a normal bounded evaluation exit.

## Metrics

- value loss first-5 / last-5 mean and maximum
- policy loss mean, standard deviation, and maximum absolute value
- entropy loss trend (noting that skrl logs negative entropy regularization)
- policy standard-deviation trend
- learning-rate constancy
- NaN / Inf count

## Result

- Smoke test passed with finite observations, actions, rewards, and PPO updates.
- In the matched 50-update comparison:
  - value-loss last-5 mean improved from 0.02766 to 0.00468
  - policy-loss standard deviation decreased from 0.01267 to 0.01049
  - maximum positive policy-loss spike decreased from 0.05960 to 0.05081
  - policy standard deviation changed smoothly from 0.6067 to 0.5995
- In the extended 200-update validation:
  - value loss decreased from a first-10 mean of 0.5609 to a last-10 mean of 0.01362
  - value-loss tail-50 mean / standard deviation were 0.01339 / 0.01045
  - policy-loss tail-50 mean / standard deviation were -0.00323 / 0.00658, with range [-0.01009, 0.02130]
  - policy standard deviation decreased from a first-10 mean of 0.6043 to a last-10 mean of 0.5187
  - skrl's signed entropy loss moved from -0.0009155 to -0.0007630 as entropy declined; this upward movement toward zero is the expected convergence direction because the implementation logs `-entropy_loss_scale * entropy`
  - instantaneous reward mean increased from a first-10 mean of 0.0360 to a last-10 mean of 0.0774
- Evaluation loaded `best_agent.pt` and ran without NaN, runtime failure, or fatal log signatures.

## Conclusion

- hypothesis supported
- keep the fixed `3.0e-4` learning rate and `time_limit_bootstrap: true`
- interpret PPO policy loss as stable when its magnitude and variance settle near zero, not when its signed value decreases monotonically
- interpret skrl entropy convergence through decreasing policy standard deviation; the logged negative entropy-loss value should approach zero
- do not change rewards, observations, network recurrence, or Phase 1 task semantics based on this optimizer-only experiment
