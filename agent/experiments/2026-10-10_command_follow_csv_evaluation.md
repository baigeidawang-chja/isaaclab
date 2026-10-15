# CommandFollow Step Evaluation and CSV Logging

## Goal
Provide reproducible, task-specific evaluation entry points for the three existing CommandFollow tasks, and add a constant-speed heading-step task that isolates steering dynamics.

## Hypothesis
One shared evaluator can record the complete policy-to-physics control chain and compute comparable per-stage speed and heading response metrics without changing the environments' rewards or the trained policy architecture.

## Baseline
- Commit / working tree: local `main` working tree with existing uncommitted Phase 1 task changes preserved
- Tasks:
  - `Isaac-TractionProtect-CommandFollow-Car4WD-Play-v0`
  - `Isaac-TractionProtect-CommandFollow-MultiStepSpeed-Car4WD-Play-v0`
  - `Isaac-TractionProtect-CommandFollow-MultiStepSpeedHeading-Car4WD-Play-v0`
- Config: existing CommandFollow MLP-PPO and Play environment configs
- Checkpoint: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-09_16-46-00_ppo_torch/checkpoints/best_agent.pt`
- Relevant baseline metrics: no common per-step CSV or per-stage response metric output

## Change
- Files:
  - `scripts/user/command_follow_evaluation.py`
  - `scripts/user/eval_command_follow.py`
  - `scripts/user/eval_multi_step_speed.py`
  - `scripts/user/eval_multi_step_speed_heading.py`
  - `scripts/user/eval_heading_step.py`
  - `source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/config/car4wd/heading_step_env_cfg.py`
  - `source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/config/car4wd/__init__.py`
  - `source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/config/car4wd/agents/skrl_ppo_heading_step_cfg.yaml`
- Parameters:
  - heading-only command: constant speed `0.8 m/s`
  - heading offsets: `[0.0, 0.2, -0.2, 0.4, -0.4, 0.0] rad`
  - stage duration: `5.0 s`
  - default settling tolerances: speed `0.05 m/s`, heading `0.05 rad`
  - steady-state window: final `1.0 s` of each stage
- Reason: isolate speed and steering responses while retaining checkpoint-compatible observations, actions, rewards, and MLP model shape.

## Commands
```bash
./isaaclab.sh -p scripts/user/eval_command_follow.py --checkpoint <checkpoint> --headless
./isaaclab.sh -p scripts/user/eval_multi_step_speed.py --checkpoint <checkpoint> --headless
./isaaclab.sh -p scripts/user/eval_multi_step_speed_heading.py --checkpoint <checkpoint> --headless
./isaaclab.sh -p scripts/user/eval_heading_step.py --checkpoint <checkpoint> --headless

WANDB_MODE=offline ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-TractionProtect-CommandFollow-HeadingStep-Car4WD-v0 \
  --num_envs 4 --max_iterations 1 --algorithm PPO \
  --checkpoint <checkpoint> --headless
```

## Run
- Date: 2026-10-10
- Evaluation output root: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_evaluations`
- Latest verified runs:
  - `command_follow/20261010_201325`
  - `multi_step_speed/20261010_201436`
  - `multi_step_speed_heading/20261010_201505`
  - `heading_step/20261010_201141`
- Heading training smoke run: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow_heading_step/2026-10-10_20-12-39_ppo_torch`
- Exit codes: all `0`

## Metrics
- `steps.csv` records commands, actual projected speed and heading, raw policy action, bounded physical target, rate-limited executed action, steering targets and measurements, wheel targets and measurements, yaw rate, and termination reason.
- `stages.csv` records speed and wrapped-heading MAE, RMSE, signed and absolute steady-state error, overshoot, overshoot percentage, and settling time.
- Actual speed is world velocity projected onto the desired-heading direction, matching the speed-tracking reward definition. Body-forward and world XY velocities are also retained for diagnosis.
- Overshoot is measured in the direction of the transition from the previous stage target to the current target. It is left empty when that command component does not step.
- Settling time is the first point where absolute error enters its tolerance and remains there through the stage.
- Every evaluation produced `599` step rows. The base task produced one stage; each staged task produced six stages.
- All required numeric command, state, and yaw-rate fields were finite.

## Result
The four evaluation entry points load the existing checkpoint and produce `steps.csv`, `stages.csv`, and `metadata.json` on the external drive. The new heading-only task is registered for training and play, and a one-iteration continuation smoke test loaded the original checkpoint successfully.

No reward term or existing task behavior was changed as part of this work.

## Conclusion
- Hypothesis supported for interface and data-pipeline validation.
- Longer checkpoint comparisons should use the generated stage metrics rather than aggregate return alone.
