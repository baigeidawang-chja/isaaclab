# Car4WD Geometry Synchronization and Re-identification

## Scope

Synchronize the current `traction_protective_control` Car4WD execution geometry
with the user-confirmed measurements, recollect policy-independent nominal data,
fit from identification only, and independently evaluate the frozen model.

Current geometry:

```text
wheel diameter = 0.130 m
wheel radius   = 0.065 m
wheelbase      = 0.411 m
track width    = 0.290 m
```

The normalized action interface remains `[1.5 m/s, 0.24 rad]`; rate limits
remain `2.0 m/s^2` and `2.0 rad/s`. At the `0.24 rad` center-angle boundary,
the inner steering target is `0.261698 rad`, inside the existing `0.261799 rad`
joint limit.

## Impact and cleanup

The changed wheel radius changes joint-speed targets for the same physical
speed and the numerical wheel-speed observations. The changed wheelbase and
track change Ackermann left/right steering and wheel-speed targets, the
geometric center-angle limit, and the yaw model. Thus previous policy tracking
metrics, open-loop response numbers, identified parameters, model predictions,
and MPC smoke results cannot be reported as current-geometry results. Earlier
PPO checkpoints can still load but must be re-evaluated and may need retraining.

Task registration, command definitions, observation-field ordering, action
ordering, reward definitions, evaluator code, and the `0.05 s` control period
are structurally reusable. The old model/steering JSON copies, old derived fit
and validation outputs, old MPC smoke outputs, and superseded conclusion
documents were removed. The old raw CSVs and checkpoints remain historical
evidence, excluded from current model fitting and performance claims.

The geometry is user-confirmed and synchronized with the task ActionTerm.
This does not independently establish the corresponding USD wheel-center and
steering-pivot geometry; those physical asset measurements need a separate
check before interpreting the model as a hardware geometry validation.

## Open-loop gate

Command:

```bash
./isaaclab.sh -p scripts/user/command_follow_open_loop.py \
  --task Isaac-TractionProtect-CommandFollow-Car4WD-v0 \
  --speed 0.8 --steer 0.22 --duration 8 --warmup 3 --headless \
  --output /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/diagnostics/open_loop_geometry_0411_0290_r0065_20261116.csv
```

- zero-steer heading rate: `0.0001195 rad/s`;
- positive/negative mean yaw rates: `+0.34790/-0.35436 rad/s`;
- absolute positive/negative ratio: `0.98178`;
- odd yaw bias: `-0.003229 rad/s`.

The response remains within the existing 5% symmetry gate.

## New data

Data directory:

```text
/media/chja/CE54D158C95990271/IsaacLabTrainingData/
traction_protective_control_nominal_identification/20261116_205318/
```

| Split | Rows | SHA-256 |
|---|---:|---|
| identification | 2980 | `00c979c031b8b8c45b50e53a62f22168332ec7362603a67793880aa26b8bca1a` |
| validation | 1560 | `2715ac5903a92fcacd90febab6c016be3d202cc352a69e5d0e661b2cdc0c0026` |

All required physical/model fields are finite, no run terminated or truncated,
and all recorded wheel targets are finite. Empty `termination_reason` cells are
read as NaN by Pandas but are not model inputs. Near `1 m/s`, straight wheel
targets are `15.3846 rad/s`, consistent with the corrected `0.065 m` radius.

Identification and validation use disjoint nonzero excitation levels:

- speed: identification `{0.3, 0.6, 0.9, 1.2}`, validation `{0.45, 0.75, 1.05}`;
- steering: identification `{+-0.08, +-0.16, +-0.22}`, validation `{+-0.12, +-0.20}`.

## Final frozen model

The fit read `identification.csv` only. Validation was opened only after the
parameter file was frozen.

```text
nominal_car4wd_model.json
SHA-256 57f5e56b7803c63908ce90df4219f2f1af65b4fa401b0de2a7147ef9dc444023
```

- selected speed delay: 0 samples;
- speed regions `[0, 0.45]`, `[0.45, 1.05]`, `[1.05, 1.2]` m/s;
- speed region `(a, b, gain, tau_s)`:
  - `(0.451374, 0.506397, 0.923027, 0.062857)`;
  - `(0.642905, 0.321374, 0.899968, 0.113184)`;
  - `(0.821411, 0.149684, 0.838144, 0.254153)`;
- steering delay: 0 samples;
- steering `(a, b, gain, tau_s)`:
  `(0.0105582, 0.988476, 0.999024, 0.010987)`;
- yaw gain: `0.9335923` with `L = 0.411 m`.

## Independent metrics

| Output | Identification RMSE | Validation RMSE |
|---|---:|---:|
| Body speed | `0.02787 m/s` | `0.03799 m/s` |
| Equivalent center steering | `0.000378 rad` | `0.000367 rad` |
| Yaw rate | `0.05298 rad/s` | `0.05178 rad/s` |

Validation did not refit or select any parameter. Validation speed acceleration
RMSE is `0.05134 m/s`, while deceleration RMSE is `0.01936 m/s`. Identification
also gives an acceleration/deceleration RMSE ratio of `1.69`, crossing the
declared `1.5` diagnostic threshold. The current frozen model remains the
minimal direction-independent baseline; this asymmetry must be evaluated before
claiming high-quality speed-step MPC performance.

## Verification

- corrected geometry/model/OSQP tests: 23 passed;
- corrected-model MPC smoke: 20/20 `solved`, 0 fallback, 19 warm starts;
- mean/max solve time: `28.61/30.20 ms` at a `50 ms` control period;
- model-to-ActionTerm target difference: at most `4.04e-8`;
- target-to-executed difference: at most `2.98e-8`.

Smoke artifacts:

```text
/media/chja/CE54D158C95990271/IsaacLabTrainingData/
traction_protective_control_nominal_mpc/fixed/20261116_205707/
```

This smoke run verifies integration, not closed-loop tracking performance.
The existing CAR USD missing-contact-prim warnings remain unchanged.

## Current artifact locations and next checks

- Final fit: `20261116_205318/nominal_model_fit_20261116_geometry_current_v2/`.
- Held-out validation: `20261116_205318/nominal_model_validation_20261116_geometry_current_v2/`.
- Integration smoke: `traction_protective_control_nominal_mpc/fixed/20261116_205707/`.
- The QP uses state `[heading_error, body_speed, center_steering]`, input
  `[physical_speed_target, physical_center_steering_target]`, and 20 steps
  at `0.05 s`. Current bounds are speed `[0,1.2] m/s`, steering
  `[-0.22,0.22] rad`, with per-step changes at most `0.1 m/s` and `0.1 rad`.
- Collect fixed, speed-step, heading-step, and joint-step closed-loop CSVs
  before making tracking-performance claims. Revisit acceleration dynamics
  on identification data before tuning MPC weights around speed steps.

## Reproduction

```bash
DATA_DIR=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_identification/20261116_205318
RUN_ROOT=$(mktemp -d /tmp/nominal_reproduction_XXXXXX)
MODEL="$RUN_ROOT/model.json"

./isaaclab.sh -p scripts/user/fit_nominal_vehicle_model.py fit \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_ROOT/fit" \
  --parameter-output "$MODEL" --max-delay-steps 6

./isaaclab.sh -p scripts/user/fit_nominal_vehicle_model.py validate \
  --data-dir "$DATA_DIR" \
  --output-dir "$RUN_ROOT/validation" \
  --parameters "$MODEL"
```

Use new output directories when repeating the experiment so the frozen
artifacts remain intact. Validation must follow model freezing and must not
be used for parameter selection.
