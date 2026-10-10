# Experiment

## Goal
Add a harder Phase 1 CommandFollow variant that tracks multiple speed steps while holding heading constant within each episode, without changing the original task or breaking checkpoint compatibility.

## Hypothesis
The original MLP-PPO checkpoint can be reused if the new task changes only planner-command timing while preserving policy observation shape, action shape, model structure, rewards, and vehicle execution.

## Baseline
- Task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`
- Algorithm: skrl MLP-PPO
- Policy observation: 25 values
- Action: 2 values
- Checkpoint: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-09_16-46-00_ppo_torch/checkpoints/best_agent.pt`

## Change
- Added `MultiStepSpeedCommand` and its configuration.
- Added independent MultiStepSpeed Train/Play environment configs and Gym IDs.
- Added an agent YAML with the same model and PPO settings but a separate external-drive log directory.
- Added a one-episode command/interface smoke test.
- Did not change the original task's command configuration, observations, actions, rewards, or terminations.

## Commands
```bash
./isaaclab.sh -p scripts/user/multi_step_speed_smoke.py --headless

WANDB_MODE=disabled ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py --task Isaac-TractionProtect-CommandFollow-MultiStepSpeed-Car4WD-v0 --num_envs 4 --max_iterations 1 --algorithm PPO --checkpoint /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-09_16-46-00_ppo_torch/checkpoints/best_agent.pt --headless

WANDB_MODE=disabled timeout --signal=INT --kill-after=15s 25s \
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task Isaac-TractionProtect-CommandFollow-MultiStepSpeed-Car4WD-Play-v0 \
  --num_envs 1 \
  --checkpoint /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-09_16-46-00_ppo_torch/checkpoints/best_agent.pt \
  --headless
```

## Run
- Date: 2026-10-09
- MultiStepSpeed command smoke exit code: 0
- Four-environment, one-iteration checkpoint continuation smoke exit code: 0
- Checkpoint play exit code: 124, expected safety timeout after successful model load
- New log root: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow_multistep_speed`
- Continuation smoke run: `2026-10-09_19-44-15_ppo_torch`

## Metrics
- Speed profile: `[0.3, 0.6, 0.9, 1.2, 0.6, 0.3] m/s`
- Transition times: `[0, 2, 4, 6, 8, 10] s`
- Episode duration: `12 s`
- Desired heading changes within episode: 0
- Policy observation shape: `(25,)`
- Action shape: `(2,)`
- Reward terms and weights: unchanged from CommandFollow
- PPO/model diff: only logging directory and W&B paths/project differ

## Result
The command smoke completed a full episode with the exact six-level profile and one fixed sampled heading. The new task instantiated with the same policy/action dimensions, and the original task's `best_agent.pt` loaded without a key or shape mismatch. A four-environment continuation run then completed all 24 rollout steps and exited normally after one PPO iteration, writing checkpoints under the new log root.

## Conclusion
- Hypothesis supported.
- The new task can initialize or resume from an original CommandFollow checkpoint using `--checkpoint`.
- Fresh MultiStepSpeed checkpoints are written to the separate external-drive directory.
- The one-iteration smoke establishes compatibility only; learning quality still requires a longer baseline and per-step speed-tracking metrics.
