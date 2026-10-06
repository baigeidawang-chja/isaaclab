# Agent objectives

Use this file for medium-term engineering objectives, not permanent Codex rules.

## Active objective: Phase 1 CommandFollow baseline

Build and validate a clean MLP-PPO baseline for wheeled-robot planner-command following on flat, high-traction ground.

The active task is:

`Isaac-TractionProtect-CommandFollow-Car4WD-v0`

The policy must follow planner-provided desired speed and heading commands using only proprioceptive, real-robot-available observations and the rate-limited continuous action semantics:

`[executed_speed, steering_control]`

### Engineering objectives

- Keep the task isolated under `isaaclab_tasks.user.traction_protective_control`.
- Verify command generation, observation ordering and scaling, action scaling and rate limiting, reset behavior, rewards, and terminations independently before tuning PPO.
- Establish a reproducible skrl MLP-PPO baseline before introducing recurrent policies.
- Keep policy observations free of friction, terrain-type, slip, stuck, contact, failure, and other simulator-privileged labels.
- Store training outputs, checkpoints, console logs, and local W&B data on the configured external drive.
- Change one RL mechanism or hypothesis at a time and record each experiment with `agent/experiment_template.md`.

### Success criteria

- `scripts/agent_smoke_test.sh` completes its bounded run without registration, import, NaN/Inf, or environment-step failures.
- `scripts/agent_train_short.sh` completes the configured short run and produces a loadable checkpoint.
- Evaluation records mean absolute speed error, mean absolute heading error, vehicle tilt/upright statistics, and action smoothness.
- A trained policy improves speed and heading tracking over zero-action and random-action baselines under the same command distribution.
- Train and play runs use the intended task IDs and write their outputs to the configured external-drive directories.

## Out of scope for Phase 1

Do not add the following until the CommandFollow baseline is stable and measured:

- low-traction or oil-contaminated terrain
- slip, stuck, contact, or failure labels in policy observations
- proprioceptive traction-state estimation
- failure prediction
- protective intervention or recovery behaviors
- waypoint navigation and obstacle recovery rewards
- GRU-PPO or DreamerV3 changes

## Planned research sequence

After Phase 1 is stable, advance in separate, controlled stages:

1. Add low-traction terrain and evaluation scenarios without privileged policy inputs.
2. Add proprioceptive state estimation and validate estimation quality independently.
3. Add failure-risk prediction with explicit labels used for supervision, not as policy observations.
4. Add protective execution behavior such as active slowdown, stopping, reversing, or takeover.
5. Compare MLP-PPO with GRU-PPO only after the environment and baseline metrics are stable.
