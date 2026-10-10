# Experiment

## Goal
Extend the Phase 1 multi-level speed task with variable world-heading commands while preserving the original and fixed-heading tasks and retaining checkpoint compatibility.

## Hypothesis
A synchronized speed/heading step task can reuse the existing MLP-PPO checkpoint because the policy-facing planner command remains `[desired_speed, heading_error]` and observation/action dimensions do not change.

## Baseline
- Original task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`
- Fixed-heading speed-step task: `Isaac-TractionProtect-CommandFollow-MultiStepSpeed-Car4WD-v0`
- Algorithm: skrl MLP-PPO
- Source checkpoint: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-09_16-46-00_ppo_torch/checkpoints/best_agent.pt`

## Change
- Added `MultiStepSpeedHeadingCommand` with a sampled episode base heading and staged heading offsets.
- Added independent Train/Play environment configs and Gym IDs.
- Added a model-compatible PPO config with a separate external-drive log root.
- Added a full-episode command smoke test.
- Did not modify the existing reward, observation, action, termination, original task, or fixed-heading speed-step task behavior.

## Command
```bash
./isaaclab.sh -p scripts/user/multi_step_speed_heading_smoke.py --headless

WANDB_MODE=offline ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-TractionProtect-CommandFollow-MultiStepSpeedHeading-Car4WD-v0 \
  --num_envs 4 --max_iterations 1 --algorithm PPO \
  --checkpoint /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/2026-10-09_16-46-00_ppo_torch/checkpoints/best_agent.pt \
  --headless
```

## Run
- Date: 2026-10-10
- Command smoke exit code: 0
- Four-environment, one-iteration continuation smoke exit code: 0
- Log root: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow_multistep_speed_heading`
- Smoke run: `2026-10-10_11-27-31_ppo_torch`

## Metrics
- Speed levels: `[0.3, 0.6, 0.9, 1.2, 0.6, 0.3] m/s`
- Heading offsets: `[0.0, +0.2, -0.2, +0.4, -0.4, 0.0] rad`
- Training base-heading range: `[-0.2, +0.2] rad`
- Play base heading: `0 rad`
- Transition times: `[0, 5, 10, 15, 20, 25] s`
- Episode duration: `30 s`
- Policy observation shape: `(25,)`
- Action shape: `(2,)`
- Reward terms and weights: unchanged

## Result
The Play command smoke observed the exact speed and heading profiles at the expected transition times. The original `best_agent.pt` loaded without key or shape errors, completed all 24 rollout steps, performed one PPO iteration, and wrote checkpoints under the new log root.

## Conclusion
- Hypothesis supported.
- Existing CommandFollow checkpoints can initialize the variable-heading task with `--checkpoint`.
- The task is structurally ready for a longer baseline.
- Tracking quality must be evaluated separately using stage-wise speed error, wrapped heading error, overshoot, settling time, and left/right response symmetry.
