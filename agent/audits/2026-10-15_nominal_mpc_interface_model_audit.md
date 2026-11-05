# Nominal MPC Interface and Model-Parameter Audit

Date: 2026-10-15

Steering-interface revision: 2026-10-29

## Scope and decision

This audit covers only the nominal, flat, high-traction CommandFollow interface. It does not add an MPC solver, alter any PPO task, or introduce low-traction estimation, failure prediction, or protective control.

**Decision: stop at interface and model structure design.** The repository contains useful step-response and steering-symmetry data, but it does not contain an identified longitudinal model, steering-actuator model, validated yaw model, model uncertainty, or delay estimates. Simulator steering joint positions are available, but the repository does not establish that actual steering-angle feedback exists on the target robot. No numerical MPC model should be created from assumed values.

## Audited control path

The current control path is:

```text
planner [v_ref, psi_ref]
  -> policy input [v_ref, wrap(psi_ref - psi)]
  -> raw normalized action a = [a_v, a_delta]
  -> clip(a, -1, 1)
  -> physical target u_target = [v_target, delta_target]
  -> per-step rate limiter
  -> executed action u_exec = [v_exec, delta_exec]
  -> Ackermann geometry
  -> [delta_left_target, delta_right_target]
     and [omega_RL, omega_RR, omega_FL, omega_FR]
  -> articulation joint targets
```

Joint ordering is explicitly preserved in the task-local action implementation:

```text
steering: [front-left, front-right]
wheels:   [rear-left, rear-right, front-left, front-right]
```

This matches the column order emitted by `AckermannAction` after the 2026-10-09 mapping correction.

## Parameter provenance

| Parameter | Current value | Source | Audit status |
|---|---:|---|---|
| Physics step | `0.01 s` | `CommandFollowEnvCfg.sim.dt`; runtime verified | Confirmed |
| Control step `T_s` | `0.05 s` (`20 Hz`) | decimation `5`; runtime `step_dt` | Confirmed |
| Train speed reference | `0.3..1.2 m/s` | `CommandsCfg` | Confirmed |
| Train world-heading reference | `-0.4..0.4 rad` | `CommandsCfg` | Confirmed |
| Base command resampling | fixed `50 s` | `CommandsCfg` | Confirmed current code; differs from the earlier 6-10 s recommendation |
| Action bounding | componentwise clip to `[-1, 1]` | `RateLimitedCarVWAction.process_actions` | Confirmed |
| Action scale | `(1.5 m/s, 0.243 rad)` | `ActionsCfg` | Confirmed; steering scale revised 2026-10-29 |
| Action offset | `(0, 0)` | `ActionsCfg` | Confirmed |
| Physical target speed | `-1.5..1.5 m/s` | clip, scale, offset; `no_reverse=False` | Confirmed interface bound |
| Virtual center-steer target | `-0.243..0.243 rad` | clip, scale, offset | Confirmed interface bound; inside the mechanical Ackermann limit |
| Speed-command rate | `2.0 m/s^2` | `max_v_rate` | Confirmed |
| Steering-command rate | `2.0 rad/s` | `max_steer_rate` | Confirmed |
| Per-control-step speed change | `0.1 m/s` | rate times `T_s` | Derived exactly from code |
| Per-control-step steer change | `0.1 rad` | rate times `T_s` | Derived exactly from code |
| Rate-limiter reset | `[0, 0]` | `reset_to_zero=True` | Confirmed |
| Ackermann wheelbase `L` | `2.035/5 = 0.407 m` | hard-coded `ActionsCfg`; runtime wheel-center X separation is approximately `0.4068 m` | Numerically cross-checked; original measurement provenance absent |
| Ackermann steering-track `W` | `1.1673/5 = 0.23346 m` | hard-coded `ActionsCfg`; runtime steering-pivot Y separation approximately `0.2314 m` | Plausible cross-check; original measurement provenance absent |
| Wheel radius | `0.035 m` | hard-coded `ActionsCfg` | Not independently verified from USD geometry or identification |
| Steering joint position limits | `+-0.261799 rad` (`+-15 deg`) | USD and runtime soft limits | Confirmed |
| Steering joint velocity limit | `5 rad/s` | `CAR_CFG` and runtime | Confirmed; command rate limit `2 rad/s` is tighter |
| Wheel joint velocity limit | `450 rad/s` | `CAR_CFG` and runtime | Confirmed |
| Wheel motor effort / saturation | `1.0 / 1.05` | `CAR_CFG` | Confirmed simulator configuration, not identified hardware behavior |
| Rigid-body maximum linear velocity | `10 m/s` | `CAR_CFG` | Confirmed; action bound is tighter |
| Nominal ground friction | static `0.8`, dynamic `0.6` | `CommandFollowSceneCfg` | Environment setting only; not an MPC state/input |

### Steering-range resolution

`delta_target` is a virtual center steering angle. `AckermannAction` maps it to separate left/right wheel angles. With the configured `L`, `W`, and the `+-15 deg` physical steering-joint limits, the largest symmetric center-angle magnitude that avoids saturating the inner wheel is:

```text
|delta_target| <= 0.243874 rad = 13.973 deg
```

The action interface was revised on 2026-10-29 to use a `0.243 rad` steering
scale. Its complete normalized range now maps to:

```text
|a_delta| <= 1
|delta_target| <= 0.243 rad
```

The effective control interval is therefore `+-0.243 rad`, the intersection of
the revised action interface and the `+-0.243874 rad` geometric interval. The
`0.000874 rad` center-angle difference keeps commanded Ackermann wheel angles
inside the `+-0.261799 rad` steering-joint limits. The former `+-0.8 rad`
mapping is obsolete for the current task.

This revision intentionally changes the physical meaning of normalized PPO
steering outputs. Legacy checkpoints remain network-shape compatible, but they
are not behaviorally equivalent under the revised scale and must not be used
as unchanged performance baselines.

The executable constraint definition is stored in:

```text
traction_protective_control/mpc/config/nominal_car4wd_steering.json
traction_protective_control/mpc/steering_constraints.py
```

This decision does not alter `RateLimitedCarVWActionCfg`, PPO observations, or
the action tensor shape. It does alter steering magnitude semantics relative
to checkpoints trained with the former `0.8 rad` scale.

The existing `+-0.25 rad` open-loop steering test slightly saturates its inner steering wheel, so it is unsuitable as an unsaturated steering-model identification point.

## Available references and measurements

### Reference input

The planner command is:

```text
r = [v_ref, psi_ref]
```

- `v_ref`: desired speed in m/s.
- `psi_ref`: desired absolute world heading in radians.
- The controller-facing error is `e_psi = wrap(psi_ref - psi)`.
- Existing speed tracking evaluates world velocity projected onto the desired-heading direction. Body longitudinal velocity is logged separately.

### Measurements available in the current simulator task

| Measurement | Current access | MPC use |
|---|---|---|
| Body linear velocity `(v_x, v_y, v_z)` | policy observation | Use `v_x`; retain `v_y` for validation |
| Body angular velocity `(p, q, r)` | policy observation and IMU | Use yaw rate `r` |
| Heading / heading error | command term has both; policy receives heading error | Use wrapped `e_psi`; absolute heading requires localization/odometry on hardware |
| Roll, pitch, projected gravity | policy observation / IMU | Monitor validity and attitude, not required in minimal flat-ground model |
| Four wheel speeds | policy observation, ordered FL/FR/RL/RR | Validation and later estimation; not required in minimal nominal state |
| Executed speed/steer command | policy observation | Required to represent rate-limiter state |
| Actual left/right steering angles | `robot.data.joint_pos`; evaluation CSV | Required for actuator/model validation; not currently in policy observation |
| World position `(x, y)` | simulator state only, not policy observation | Not needed for heading-only CommandFollow; would be needed for path tracking |

Simulator availability does not prove hardware availability. Before implementation, document the real vehicle's steering-angle encoder, heading/localization source, timestamps, units, sign convention, update rate, and latency.

## MPC input and normalized-action adapter

The recommended MPC decision input is the **physical pre-rate-limit target**:

```text
u_target = [v_target (m/s), delta_target (rad)]
```

The existing action term remains the authority for bounding, rate limiting, and Ackermann joint-target generation. The adapter relationship is:

```text
a_v     = (v_target     - 0.0) / 1.5
a_delta = (delta_target - 0.0) / 0.243
a       = clip([a_v, a_delta], -1, 1)
```

For an MPC-generated feasible target, clipping should be inactive. A clip event must be logged as an interface error. The optimizer must model the existing rate limiter:

```text
u_exec[k] = u_exec[k-1]
          + clip(u_target[k] - u_exec[k-1], -rate*T_s, +rate*T_s)
```

with componentwise rates `[2.0 m/s^2, 2.0 rad/s]`. Do not send wheel angular velocities or left/right steering targets directly from the MPC in the first nominal implementation; that would bypass the current physical-control contract.

## Recommended minimum model structure

Because the task regulates speed and heading rather than a path, lateral position error is not required initially. The smallest useful structure that preserves measured actuator state is:

```text
x = [v_x, e_psi, r, delta, v_exec, delta_exec]
r_ref = [v_ref, psi_ref]
u = [v_target, delta_target]
```

- `delta` is an equivalent bicycle steering angle reconstructed from measured left/right steering angles, or a directly measured center-steer state if hardware provides one.
- `v_exec` and `delta_exec` are deterministic rate-limiter states and must be initialized from the action term, not assumed equal to the new target.
- For a piecewise-constant heading command, the known kinematic error update is `e_psi[k+1] = wrap(e_psi[k] - T_s*r[k])`. A command step contributes the known reference-heading increment.
- Longitudinal dynamics `v_x[k+1]`, yaw dynamics `r[k+1]`, and steering dynamics `delta[k+1]` remain unidentified placeholders.

A kinematic bicycle equation `r = v_x*tan(delta)/L` may be retained as a comparison baseline only after validating its residuals over the intended speed/steering envelope. It must not silently replace the missing identified dynamics.

## Proposed constraints

These are structural constraints based on current code and USD, not tuned MPC settings:

```text
T_s                         = 0.05 s
-1.5 <= v_target           <= 1.5 m/s
-0.243 <= delta_target     <= 0.243 rad (current action-interface bound)
|Delta v_exec|             <= 0.1 m/s per control step
|Delta delta_exec|         <= 0.1 rad per control step
|delta_left/right_target|  <= 0.261799 rad
|omega_wheel_target|       <= 450 rad/s
```

The training-reference envelope remains `0.3..1.2 m/s` and `-0.4..0.4 rad` world heading; reference bounds are not actuator bounds. State/output constraints for yaw rate, lateral velocity, acceleration, and tracking error cannot be assigned until data establishes safe and achievable values.

## Existing identification evidence

### Policy-free open-loop test

Source:

```text
/media/chja/CE54D158C95990271/IsaacLabTrainingData/
  traction_protective_control_command_follow/diagnostics/
  open_loop_mapping_fixed_20261009.csv
  open_loop_mapping_fixed_20261009.summary.json
```

At `0.8 m/s`:

| Executed center steer | Mean yaw rate | Heading slope |
|---:|---:|---:|
| `0 rad` | `-0.004636 rad/s` | `-0.000652 rad/s` |
| `+0.25 rad` | `+0.595008 rad/s` | `+0.527500 rad/s` |
| `-0.25 rad` | `-0.591526 rad/s` | `-0.515875 rad/s` |

The positive/negative yaw-rate magnitude ratio is `1.005886`. This validates mapping symmetry at one operating condition. It does **not** identify gain scheduling, transient time constants, delay, or model uncertainty, and `+-0.25 rad` is slightly beyond the unsaturated center-angle limit derived above.

### PPO closed-loop step evaluations

Existing `multi_step_speed`, `multi_step_speed_heading`, and `heading_step` CSVs contain per-stage MAE, RMSE, steady-state error, overshoot, and settling time. The heading-only run at `0.8 m/s` reports heading settling times of approximately `0.75, 1.20, 1.70, 2.00, 1.40 s` for its five non-initial steps. These are useful closed-loop acceptance baselines, but they combine the PPO policy, rate limiter, actuator, and vehicle. They are not open-loop plant-identification parameters.

### Missing identification products

No repository or external result found in the audited paths provides:

- longitudinal gain, time constant, dead time, or discrete `A/B` coefficients;
- steering servo gain, time constant, dead time, hysteresis, or saturation fit;
- yaw-rate/curvature model across speed and steering operating points;
- equivalent steering reconstruction definition and validation;
- residual distributions, uncertainty bounds, train/validation split, or fit score;
- real-hardware steering-angle availability and latency.

## Solver dependency audit

No QP/NLP package is declared in the searched project manifests or documentation. The current Isaac Lab Python environment contains:

| Package | Version | Assessment |
|---|---:|---|
| `osqp` | `0.6.5` | Available direct QP solver candidate after a linear model exists |
| `scipy` | `1.10.1` | Available numerical/optimization utilities; not by itself a selected real-time MPC stack |

`cvxpy`, `clarabel`, `scs`, `qpsolvers`, `proxsuite`, `casadi`, `do_mpc`, `acados_template`, `cyipopt`, `nlopt`, `gekko`, `gurobipy`, `highspy`, `qpth`, and `jaxopt` were not present. No dependency was installed. Availability through the current environment is not yet a reproducible project dependency declaration.

## Required MPC log fields

Retain all existing evaluation fields and add, when MPC implementation is authorized:

- monotonic timestamp, simulation timestamp, actual control `dt`, stage index;
- `v_ref`, `psi_ref`, `e_v`, wrapped `e_psi`;
- body `v_x`, `v_y`, yaw rate, absolute heading, optional world `x/y`;
- actual left/right steering and reconstructed equivalent `delta`;
- four actual wheel speeds;
- normalized adapter action;
- physical pre-rate target and post-rate executed action;
- Ackermann left/right steering targets and four wheel-speed targets;
- target clipping, rate-limit activity, steering-limit activity, wheel-limit activity;
- model version and parameter-set identifier;
- solver status, solve time, iterations, objective and residuals;
- first predicted state/input and horizon trajectories, preferably in a separate compact artifact;
- termination/truncation reason and reset marker.

## Proposed future files (not created in this audit)

```text
source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/mpc/
  __init__.py
  interfaces.py            # typed measurement/reference/control records
  action_adapter.py        # physical target <-> normalized action only
  nominal_model.py         # model equations after identification
  model_parameters.py      # validated values plus provenance/version

scripts/user/
  collect_nominal_vehicle_id.py
  fit_nominal_vehicle_model.py
  eval_nominal_mpc.py

agent/audits/
  nominal_vehicle_identification.md
```

No separate MPC environment should be introduced until the model and interface are validated. Initial data collection can reuse the existing flat CommandFollow environment while bypassing the PPO policy, as the open-loop symmetry script already does.

## Implementation gate

Do not proceed to a solver or closed-loop MPC task until all of the following are complete:

1. Resolved: align the PPO and nominal MPC virtual center-steering interface at `+-0.243 rad`, inside the independently derived `+-0.243874 rad` Ackermann/joint limit.
2. Verify wheel radius and document the measurement/provenance of `L` and `W`.
3. Confirm real-robot steering-angle, heading, velocity, and timestamp interfaces.
4. Collect policy-independent excitation data over multiple speeds and unsaturated positive/negative steering levels, including zero-input repeats.
5. Fit longitudinal, steering, and yaw models; estimate delay and uncertainty; validate on held-out runs.
6. Select a model form and horizon from measured residuals and timing, then evaluate whether direct OSQP is sufficient.
7. Add solver dependencies to the project only through an explicit, reviewed dependency change.

Until this gate is satisfied, the nominal MPC stage remains intentionally at structure design.
