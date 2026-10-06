# Experiment

## Goal

Determine whether bootstrapping normal episode time limits removes the periodic Phase 1 critic-loss spikes.

## Hypothesis

Setting `time_limit_bootstrap: true` will prevent the 12-second timeout from being treated as a zero-value terminal state, reducing value-loss spikes at 240-step episode boundaries without degrading command-tracking rewards.

## Baseline

- Commit / working tree: dirty working tree; preserve unrelated user changes
- Task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`
- Config: skrl MLP-PPO, 64 environments, 50 iterations, 24-step rollouts
- Checkpoint: `2026-10-05_20-04-27_ppo_torch/checkpoints/agent_1200.pt`
- Log directory: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-05_20-04-27_ppo_torch`
- Relevant baseline metrics:
  - value loss first-5 mean: 0.7801
  - value loss last-5 mean: 0.2386
  - value loss minimum / maximum: 0.00685 / 4.2215
  - periodic spikes occur at or immediately after steps 240, 480, 720, 960, and 1200
  - no non-finite value-loss samples

## Change

- Files: `traction_protective_control/config/car4wd/agents/skrl_ppo_cfg.yaml`
- Parameters: `time_limit_bootstrap: false -> true`
- Reason: episode timeout is truncation, not task failure; critic targets should retain the next-state value.

## Command

```bash
./scripts/agent_smoke_test.sh
./scripts/agent_train_short.sh
```

## Run

- Start: 2026-10-05 20:16 (smoke), 2026-10-05 20:18 (short training)
- End: 2026-10-05 20:19
- Log directory:
  - smoke: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/agent_runs/smoke_20261005_201641`
  - short console: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/agent_runs/short_20261005_201838`
  - skrl run: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-05_20-18-43_ppo_torch`
- Exit code: 0 for smoke and short training

## Metrics

- value loss first-5 and last-5 mean
- value loss minimum and maximum
- value loss near 240-step episode boundaries
- speed-tracking reward
- heading-tracking reward
- NaN / Inf count

## Result

- Smoke test passed without registration, import, environment-step, or non-finite-value failures.
- Value loss first-5 mean remained 0.7801 because the run is deterministic up to the first timeout.
- Value loss last-5 mean decreased from 0.2386 to 0.02766.
- Value loss maximum decreased from 4.2215 to 1.1486; the new maximum occurred during initial critic fitting.
- Boundary-region value losses were:
  - step 240: 0.4432 instead of 4.2215
  - step 480: 0.0195, followed by 0.2493 at step 504 instead of 1.0312
  - step 720: 0.1340 instead of 2.1743
  - step 960: 0.0142, followed by 0.1027 at step 984 instead of 3.0160
  - step 1200: 0.0332 instead of 0.9784
- No value-loss samples were NaN or Inf.
- The 50-iteration command-tracking rewards did not improve over the baseline run, so policy-performance improvement is not claimed from this experiment.

## Conclusion

- hypothesis supported for critic stability
- keep `time_limit_bootstrap: true`
- next action: evaluate command-follow behavior separately before changing another PPO or reward parameter
