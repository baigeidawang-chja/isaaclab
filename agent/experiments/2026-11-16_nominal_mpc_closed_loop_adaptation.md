# Current-Geometry Nominal MPC Closed-Loop Pilot

## Goal

Check the identified-model OSQP MPC, current ActionTerm mapping, solver timing,
and tracking under fixed, speed-step, heading-step, and joint-step commands.
These are single-environment, seed-42 Isaac Lab pilot runs on flat ground.
The frozen model SHA-256 is
`57f5e56b7803c63908ce90df4219f2f1af65b4fa401b0de2a7147ef9dc444023`.
No parameter or MPC weight was selected from the independent validation CSV.

## Commands and data

```bash
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --profile fixed --duration 4 --speed 0.6 --heading 0 --headless
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --profile speed_step --duration 12 --segment_duration 3 --speed_sequence 0.3,0.8,1.2,0.6 --heading 0 --headless
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --profile heading_step --duration 12 --segment_duration 3 --speed 0.8 --heading_sequence 0.0,0.2,-0.2,0.0 --headless
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --profile joint_step --duration 12 --segment_duration 3 --speed_sequence 0.3,0.8,1.2,0.6 --heading_sequence 0.0,0.2,-0.2,0.0 --headless
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --profile speed_step --duration 12 --segment_duration 3 --speed_sequence 0.3,0.6,0.8,0.9 --heading 0 --headless
```

Run directories under `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_mpc/`:

```text
fixed/20261116_212829
speed_step/20261116_212909
heading_step/20261116_212931
joint_step/20261116_212953
speed_step/20261116_213304
```

Each contains raw `steps.csv`, prediction horizons, metadata, `analysis.json`,
and `stage_metrics.csv`. Recompute the latter two with:

```bash
python3 scripts/user/analyze_nominal_mpc.py <one-or-more-run-directories>
```

The analyzer checks model/config hashes and reports stage MAE, RMSE, last-second
bias, overshoot, settling time at `0.05 m/s` and `0.05 rad`, one-step residuals,
wheel-speed target error, command saturation, and solver timing.

## Results

All `1040/1040` steps solved, with zero fallback or early termination. Maximum
Python solve time was `35.85 ms` against a `50 ms` control period. The maximum
physical-target mapping difference was about `1e-7` in the logged units;
target versus executed action differed by at most about `3e-8`.

| Run/stage | Speed command | Last-second speed bias (command - actual) | Last-second heading bias | Speed-target limit fraction |
|---|---:|---:|---:|---:|
| fixed | 0.6 m/s | -0.0092 m/s | +0.0006 rad | 0% |
| speed step, 0.8 | 0.8 m/s | +0.0101 m/s | +0.0005 rad | 0% |
| speed step, 1.2 | 1.2 m/s | +0.2808 m/s | +0.0006 rad | 95% |
| joint step, 1.2 / -0.2 | 1.2 m/s | +0.1981 m/s | +0.0004 rad | 97% |
| lower-range step, 0.9 | 0.9 m/s | +0.0379 m/s | +0.0005 rad | 13% |

At constant `0.8 m/s`, heading settling times were `0.50 s` for
`0 -> +0.2 rad`, `1.00 s` for `+0.2 -> -0.2 rad`, and `0.50 s` for
`-0.2 -> 0 rad`. These steps have different magnitudes, so they do not prove
left/right symmetry. Across stages, one-step speed prediction RMSE was about
`0.027..0.056 m/s`, heading below `0.002 rad`, and center steering below
`0.0005 rad`. These are one-step results only.

The top identified speed region has DC gain `0.83814`. At the maximum
identified physical input of `1.2 m/s`, the model predicts about `1.006 m/s`
steady body speed. Identification data measured about `1.014 m/s` in the last
second of its `1.2 m/s` segment. Closed-loop last-second actual speeds were
about `0.919` and `1.002 m/s`, with MPC physical speed target at `1.2 m/s`.
Wheel-speed targets were about `18.46 rad/s`; actual wheel speeds were lower
and varied by wheel. This indicates an execution-response limit or mismatch,
without isolating a physical cause. The model-derived `1.006 m/s` is not a
hardware maximum. The lower-range run still had last-second speed bias of
`0.055 m/s` at `0.8 m/s` and `0.038 m/s` at `0.9 m/s`.

## Decision

Keep the frozen model, physical bounds, and initial QP weights unchanged.
Inspect wheel actuator effort/velocity tracking and actual USD geometry;
vary preceding speed profiles and hold duration to estimate the
reachable-speed envelope. If acceleration/deceleration residuals persist,
compare a direction-split candidate using identification data only. Then
assess controller weights on current-geometry closed-loop data, without
using the held-out model-validation set for tuning.

## Follow-up drive diagnostics

Additional commands:

```bash
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --profile fixed --duration 6 --speed 1.2 --heading 0 --seed 42 --headless
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --profile fixed --duration 6 --speed 1.2 --heading 0 --seed 43 --headless
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --profile fixed --duration 6 --speed 0.9 --heading 0 --seed 42 --headless
```

These pre-torque-log runs are `fixed/20261116_213859`,
`fixed/20261116_213915`, and `fixed/20261116_213932`. Seeds 42 and 43
produced identical physical state, wheel-speed, and MPC control trajectories
for the fixed `1.2 m/s` case; only solve-time values differed. A seed change
alone is therefore not an independent physical trial under this reset.

After adding per-wheel computed and applied torque to the evaluator, the
fixed `1.2` and `0.9 m/s` runs were repeated with seed 42. Their directories
are `fixed/20261116_214234` and `fixed/20261116_214250`; both include the
new torque fields. Across all ten runs (`1640` control steps), every QP
solved without fallback or early termination; maximum Python solve time was
`40.05 ms`.

| Six-second hold | Last-second actual speed | Mean applied wheel torque magnitude | Torque clip fraction | Negative actual wheel-speed fraction |
|---|---:|---:|---:|---:|
| 1.2 m/s command | 0.9825 m/s | 0.974 Nm | 96.3% | 7.5% |
| 0.9 m/s command | 0.8827 m/s | 0.961 Nm | 88.7% | 10.0% |

A wheel sample is counted as clipped when its computed torque magnitude
exceeds the applied magnitude by more than `1e-4 Nm`. `CAR_CFG` configures
wheel-drive effort limit `1.0 Nm` and saturation effort `1.05 Nm`. Torque
clipping is directly observed; its role relative to wheel contact, joint
orientation, and actual USD geometry still needs investigation. No asset,
actuator limit, PPO setting, model parameter, or MPC weight was changed.

## USD geometry audit and adaptation gate

The loaded `/home/chja/myproject/car_no_damper_04.usd` declares meters per
unit as `1.0`. Direct USD joint-body world anchors place front wheel axes at
`x=+0.2035 m` and rear wheel axes at `x=-0.2035 m`: axle-center spacing is
`0.4070 m`, versus the configured/model `0.411 m`. Front steering pivots
are at `y=-0.116728` and `+0.116731 m`, giving `0.233459 m` pivot spacing.
Wheel axle anchors have a different lateral span: front approximately
`0.26758 m`, rear approximately `0.26906 m`. None equals the configured
`0.290 m` Ackermann width. These are USD anchor measurements, not a
wheel-contact-patch measurement; the user-provided physical dimensions
`0.411/0.290/0.130 m` are not silently overwritten.

The current ActionTerm, steering constraints, and identified model agree
with each other at `0.411/0.290/0.065 m`, so the MPC-to-ActionTerm mapping
check passes, but it does not prove agreement with the USD mechanism.
Because identification and these pilots use the same simulation asset, the
fitted yaw gain may partly absorb this mismatch. The model is therefore
a reproducible baseline for this exact USD and interface, not a
geometry-validated vehicle model or a transferable hardware result.
Before changing weights or claiming performance, resolve which measured
width should define the Ackermann geometry and inspect the wheel-contact
and actuator-limit behavior. Any geometry change requires fresh
identification and held-out validation; do not retrofit the frozen JSON.

Isaac Sim also emitted six errors for `base_link/FixedJoint*` relationships
to missing `contact_*` prims during the two-step metadata smoke run. The run
completed with both QPs solved, but these unresolved asset references should
be inspected before interpreting contact or wheel-load behavior.
The metadata-enabled run is `fixed/20261116_215210`, using USD SHA-256
`28c56b286c78df43a6f9d06a9f9393be205f9b3279211b5c05b14b95ee7a5940`.
It is an interface smoke test, not a tracking-performance result.
