# Signed CommandFollow Speed Range and Reverse Interface Check

## Goal

Allow planner-requested signed body speed in [-0.5, 1.2] m/s and locate the
four distinct speed ranges in the engineering interface.

## Hypothesis

The existing no_reverse=False ActionTerm already transmits negative speed
targets correctly. Enabling negative planner sampling should not change
observation shape, physical action scaling, or rate limiting.

## Baseline

- Task: Isaac-TractionProtect-CommandFollow-Car4WD-v0 and its Play variant.
- Algorithm: existing MLP-PPO task; this check bypasses the policy.
- Working tree: existing geometry/model/MPC work retained without rollback.
- Previous planner range: training [0.3, 1.2], Play [0.6, 0.8] m/s.
- ActionTerm input range: [-1.5, 1.5] m/s from scale=(1.5, 0.24), zero offset.
- Identified model input range: [0, 1.2] m/s; reverse was not identified.
- MPC physical control range: [0, 1.2] m/s, steering [-0.22, 0.22] rad.

## Change

- command_follow_env_cfg.py: primary Train and Play desired_speed_range
  changed to (-0.5, 1.2).
- mdp/commands.py: same generic PlannerCommandCfg default, with signed-speed
  semantics documented.
- command_follow_open_loop.py: log actual_body_vx_m_s and its steady mean;
  clarify that --speed accepts signed physical inputs.
- AGENTS.md and agent/objectives.md: current reference range and model gate.
- Existing scripted speed/heading-step task profiles remain unchanged.

The requested range is interpreted as a planner reference range, retaining
existing execution input headroom. This does not enforce a hard
measured-speed bound.

Negative speed means reversing while the desired heading still describes
the vehicle's forward body axis. No pi rotation is added to the heading.
Action scaling, rate limits, reward, observation, geometry, and torque
settings are unchanged. Existing PPO checkpoints need reverse evaluation
or retraining before they can be treated as reverse-capable controllers.

## Command

```bash
./isaaclab.sh -p scripts/user/command_follow_open_loop.py \
  --speed -0.5 --steer 0.1 --duration 4 --warmup 2 --headless \
  --output /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_execution/reverse_smoke_20261119_165544/steps.csv
```

## Run

- Date: 2026-11-19, local start 16:55:44.
- Exit code: 0.
- Three four-second cases: zero, +0.1, and -0.1 rad center steering.
- Seed: not explicitly set by this existing diagnostic entry point.
- Data: 240 real Isaac Sim samples on the external drive.
- Metrics average the samples at t >= 2 s.
- The existing missing contact_* FixedJoint warnings remained visible.
  Contact and actuator investigation is deferred under the user's current
  fixed-environment assumption; it is not an additional gate for this check.

## Metrics

| Center steering | Mean body vx (m/s) | Mean body yaw rate (rad/s) |
|---|---:|---:|
| 0 | -0.45663 | +0.00583 |
| +0.1 | -0.45347 | -0.10336 |
| -0.1 | -0.45222 | +0.10456 |

All 240 records were finite, all wheel-speed targets negative, and each
executed-action increment stayed within 0.100001 in its physical units.
There were no unexpected terminations.

A separate runtime check confirmed both Train and Play ranges, unchanged
scale and no_reverse=False. With seed 42, 256 planner samples spanned
[-0.49485, 1.18747] m/s with 69 negative samples. A negative-action step
produced finite reward and policy observations of shape (1, 25).

## MPC Status

The existing OSQP controller still returns invalid_input with a zero fallback
from rest for a -0.5 m/s reference. Its frozen model applicability is
nonnegative, its predicted-speed lower bound is zero, and _build_qp uses
max(0.1, abs(measured_speed), abs(reference_speed)) for heading linearization.
Changing only JSON control bounds would not implement correct reverse MPC.

Next work is independent reverse longitudinal and steering identification,
including stop/direction changes, followed by held-out validation. Then
extend the input/state constraints, keep the signed speed in the heading
linearization, and validate reverse/forward transitions. Do not relabel
this short interface check as reverse identification or MPC performance.

## Conclusion

The planner and existing execution interface support signed speed commands.
No learned-policy or reverse-MPC performance claim follows from this check.
Use ideal Ackermann geometry L=0.411 m, W=0.290 m, r=0.065 m; steering-pivot
offsets are intentionally ignored as specified by the user.

