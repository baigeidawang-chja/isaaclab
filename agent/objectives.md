# Agent objectives

Use this file for medium-term engineering objectives, not permanent Codex rules.

## Active objective: Phase 1 CommandFollow baseline

Build and validate a clean MLP-PPO baseline for wheeled-robot planner-command following on flat, high-traction ground.

Configured execution geometry is wheel diameter `0.130 m`, wheelbase
`0.411 m`, and track width `0.290 m`; steering action scale is `0.24 rad`.
The current research assumption is ideal Ackermann geometry with steering-axis
and wheel-center offsets ignored. Contact structure and drive limits remain
fixed under the user's 2026-11-19 scope decision.
Earlier PPO checkpoints and tracking results were recorded with a different
execution geometry. They are historical and do not establish a current-geometry
CommandFollow baseline.

The active task is:

`Isaac-TractionProtect-CommandFollow-Car4WD-v0`

The policy must follow planner-provided desired speed and heading commands using only proprioceptive, real-robot-available observations and the rate-limited continuous action semantics:

`[executed_speed, steering_control]`

Primary Train/Play planner speed range is `[-0.5, 0.8] m/s`. Negative speed
means reversing with the same body-heading reference. The ActionTerm retains
`[-1.5, 1.5] m/s` input headroom; this is not a hard measured-speed limiter.
Scripted step-task profiles keep their separate prescribed levels.

### Engineering objectives

- Keep the task isolated under `isaaclab_tasks.user.traction_protective_control`.
- Verify command generation, observation ordering and scaling, action scaling and rate limiting, reset behavior, rewards, and terminations independently before tuning PPO.
- Establish a reproducible skrl MLP-PPO baseline before introducing recurrent policies.
- Re-evaluate and, if needed, retrain PPO on the current geometry before making
  current-geometry policy-performance claims.
- Keep policy observations free of friction, terrain-type, slip, stuck, contact, failure, and other simulator-privileged labels.
- Store training outputs, checkpoints, console logs, and local W&B data on the configured external drive.
- Change one RL mechanism or hypothesis at a time and record each experiment with `agent/experiment_template.md`.

### Success criteria

- `scripts/agent_smoke_test.sh` completes its bounded run without registration, import, NaN/Inf, or environment-step failures.
- `scripts/agent_train_short.sh` completes the configured short run and produces a loadable checkpoint.
- Evaluation records mean absolute speed error, mean absolute heading error, vehicle tilt/upright statistics, and action smoothness.
- A trained policy improves speed and heading tracking over zero-action and random-action baselines under the same command distribution.
- Train and play runs use the intended task IDs and write their outputs to the configured external-drive directories.

### Nominal-model progress

The current default is the independently validated bidirectional model:
`mpc/config/nominal_car4wd_bidirectional_model.json`, with matching
`nominal_car4wd_bidirectional_osqp_mpc.json` and validation evidence.
Identification-only model selection used 8000 samples; the separate 5040-sample
validation was evaluated once after freezing parameters. Physical speed inputs
are [-1.5, 1.5] m/s; current planner/MPC references are [-0.5, 0.8] m/s.
The model has dt=0.05 s, three signed speed regions, zero selected delays,
first-order steering, and unit-gain Ackermann yaw. Independent validation RMSE:
0.04834 m/s speed, 0.000452 rad steering, 0.05258 rad/s algebraic yaw rate.
The previous forward-only model and 2026-11-16 results remain historical files.

Reverse MPC now uses signed heading linearization and compatible constraints.
35 offline tests and 1460 real Isaac Sim steps (reverse fixed, reverse heading,
signed speed and joint steps) passed with no solver fallback. Current record:
`agent/experiments/2026-11-20_bidirectional_nominal_mpc.md`.

The 2026-11-24 Chapter 2 runs and 2026-11-26 high-speed diagnostic used the
former 1.2 m/s planner upper bound and remain historical, immutable records.
The new 0.8 m/s longitudinal matrix is complete locally: 12 formal runs,
29 stages, plus three technical short tests and four longer holds. Results:
`agent/experiments/2026-11-26_nominal_mpc_longitudinal_ref08.md`. The high-end
mean tracks the command, but sustained in-band stability remains unproven.
Do not retune the model or MPC weights to match historical results.
Keep validation data out of tuning. Actuator/contact settings remain fixed
under the user's scope decision; PPO re-evaluation is a separate work item.

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
