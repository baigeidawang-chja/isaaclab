# Experiment: Bidirectional Nominal MPC

## Goal

Make planner, identified model, OSQP, action adapter, and vehicle execution
agree on signed speed, without changing PPO or the physical execution mapping.
Negative speed means reversing with the same desired body heading, not heading + pi.

## Hypothesis

The forward-only MPC restriction is an interface/model-support limitation:
independent bidirectional identification, signed yaw linearization, and consistent
constraints should permit reverse command following with the existing OSQP weights.

## Baseline

- Working tree: main at `4cb68f955414ca9039967eced70d9b387cae1bb9`, with existing
  uncommitted geometry/MPC work preserved. No commit or checkout performed.
- Task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`, one environment.
- Controller: nominal sparse OSQP MPC; no policy or checkpoint used.
- Geometry: radius 0.065 m, wheelbase 0.411 m, track 0.290 m.
- Action: scale (1.5, 0.24), offset zero, clip, no_reverse=False; rates 2 m/s^2
  and 2 rad/s; control period 0.05 s. Unchanged in this experiment.
- Old model/config files `nominal_car4wd_model.json` and
  `nominal_car4wd_osqp_mpc.json` remain unchanged by this experiment for historical use.
- Old MPC rejected negative references and used absolute speed in heading linearization.

## Change

- Added `mpc/config/nominal_vehicle_bidirectional_protocol.json`.
- Extended `scripts/user/fit_nominal_vehicle_model.py` to select signed-input
  schedules using identification data only, and distinguish reverse, acceleration,
  deceleration (absolute speed), and direction-change residuals.
- Generated `mpc/config/nominal_car4wd_bidirectional_model.json` and
  `nominal_car4wd_bidirectional_validation.json`. The latter is an unchanged copy
  of the independent validation output, including the frozen parameter SHA256.
- Added `mpc/config/nominal_car4wd_bidirectional_osqp_mpc.json`; switched current
  model, collector, fitter, and MPC defaults to the bidirectional artifacts.
- Updated `mpc/osqp_mpc.py`: separate planner references from physical inputs,
  validate input support against the loaded model, load the configured model file,
  and preserve signed speed in local yaw linearization. Above 0.02 m/s absolute
  measured speed, use actual travel direction even when the reference reverses;
  near rest use reference speed. Zero-speed rest has no turning authority.
- Updated `eval_nominal_mpc.py`: validate reference range/no_reverse contract,
  save per-run configuration snapshots, synchronize environment/GUI commands with
  MPC references, and log environment reference values.
- Updated `analyze_nominal_mpc.py`: add lower saturation and reverse reachability.
- Added signed-speed regression tests and synchronized current development notes.
- No PPO reward, observation, ActionTerm mapping, task registration, actuator,
  contact, USD, checkpoint, or low-traction module changes in this experiment.

### Current Contract

| Quantity | Current range | Location |
| --- | --- | --- |
| Planner reference and MPC speed reference | [-0.5, 1.2] m/s | CommandFollow config / planner_reference_bounds |
| ActionTerm physical speed target | [-1.5, 1.5] m/s | scale=(1.5, 0.24), no_reverse=False |
| Independently identified executed speed input | [-1.5, 1.5] m/s | bidirectional model applicability/regions |
| MPC physical speed output | [-1.5, 1.5] m/s | effective_mpc_control_bounds |
| MPC predicted body-speed guard | [-1.6, 1.6] m/s | predicted_state_bounds |
| ActionTerm center-steering target | [-0.24, 0.24] rad | unchanged action scale |
| Identified/MPC center-steering target | [-0.22, 0.22] rad | measured protocol support |

Physical inputs retain compensation headroom. These reference/input bounds are
not a hard measured-speed guarantee of [-0.5, 1.2] m/s. No new safety limiter is
claimed. The adapter applies `[v_target/1.5, delta_target/0.24]` exactly once.

## Command

Commands below are equivalent to the executed commands after shell variable
expansion. Existing output directories must not be overwritten when reproducing;
choose fresh result directories and a separate parameter-output filename.

```bash
CFG=source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/mpc/config
DATA=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_identification/20261120_173554
RUN=/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_mpc/bidirectional_20261120

./isaaclab.sh -p scripts/user/collect_nominal_vehicle_identification.py \
  --headless --protocol "$CFG/nominal_vehicle_bidirectional_protocol.json"
./isaaclab.sh -p scripts/user/fit_nominal_vehicle_model.py fit \
  --data-dir "$DATA" --output-dir "$DATA/fit" \
  --parameter-output "$CFG/nominal_car4wd_bidirectional_model.json"
./isaaclab.sh -p scripts/user/fit_nominal_vehicle_model.py validate \
  --data-dir "$DATA" --output-dir "$DATA/validation" \
  --parameters "$CFG/nominal_car4wd_bidirectional_model.json"

./isaaclab.sh -p -m unittest discover -s source/isaaclab_tasks/test \
  -p 'test_traction_protective_control_*.py' -v

./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --headless \
  --profile fixed --speed=-0.5 --heading=0 --duration 8 \
  --output_dir "$RUN/reverse_fixed"
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --headless \
  --profile heading_step --speed=-0.5 --heading_sequence=0,0.2,-0.2,0 \
  --segment_duration 5 --duration 20 --output_dir "$RUN/reverse_heading"
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --headless \
  --profile joint_step --speed_sequence=0.6,-0.5,0,1.2,-0.5 \
  --heading_sequence=0,0.2,0.2,-0.2,0 --segment_duration 5 --duration 25 \
  --output_dir "$RUN/signed_joint"
./isaaclab.sh -p scripts/user/eval_nominal_mpc.py --headless \
  --profile speed_step --speed_sequence=-0.5,0.6,0,1.2,-0.5 --heading=0 \
  --segment_duration 4 --duration 20 --output_dir "$RUN/signed_speed"
./isaaclab.sh -p scripts/user/analyze_nominal_mpc.py \
  "$RUN/reverse_fixed" "$RUN/reverse_heading" "$RUN/signed_joint" "$RUN/signed_speed"
```

## Run

- Date: 2026-11-20, Asia/Shanghai; collection launched 17:33, completed 17:35;
  closed-loop checks launched between 17:39 and 17:41.
- Collection output: `$DATA/identification.csv` (8000 rows) and
  `$DATA/validation.csv` (5040 rows). Distinct resets, amplitudes, timing, and order;
  no policy. All checked states/inputs finite; no terminated/truncated samples.
- Fits/selection: `$DATA/fit/`; only identification.csv was read by `fit`.
- Independent evaluation: `$DATA/validation/`; run once after freezing parameters,
  with no parameter or weight changes from its results.
- Both directories contain prediction CSVs, prediction curves and residual PNGs.
- Frozen model SHA256:
  `78fe257a39f108a103c0a0edb3dcc1afb53aff3b861bcac9926cfca7a28da3f3`.
- Validation CSV SHA256:
  `3258542fde8b5a403a54ef9bf3998bfde813f2055470c0958418aa44fbf35529`.
- Closed-loop outputs: `$RUN/<profile>/steps.csv`, predictions.jsonl, metadata,
  model/mpc/steering config snapshots, analysis.json, stage_metrics.csv.
- Commands completed with exit code 0. Isaac Sim still emitted existing missing
  contact FixedJoint prim and duplicate-registration warnings. They were recorded,
  not repaired; physical settings remain fixed under the user's scope decision.

## Metrics

### Identification And Independent Validation

| Quantity | Identification RMSE | Validation RMSE |
| --- | ---: | ---: |
| Longitudinal speed | 0.041851 m/s | 0.048338 m/s |
| Equivalent center steering | 0.000446 rad | 0.000452 rad |
| Yaw rate | 0.054398 rad/s | 0.052580 rad/s |

Speed/steering are free-running actuator predictions with measured executed inputs.
Yaw is an algebraic prediction using measured body speed and center steering, not
an end-to-end open-loop heading prediction. These metrics must not be conflated.
Validation forward/reverse speed-stage RMSE: 0.062154 / 0.042035 m/s.

Identification-only selection tried scalar, direction, magnitude, direction-speed,
and direction-three-speed schedules with delays 0 through 6. Minimum BIC with a
complexity tie-break selected three regions; blocked-CV speed RMSE was 0.047984 m/s.

| Executed speed interval (m/s) | DC gain | Time constant (s) | Delay (steps) |
| --- | ---: | ---: | ---: |
| [-1.5, -0.75) | 0.799231 | 0.278434 | 0 |
| [-0.75, 0.75) | 0.909265 | 0.093176 | 0 |
| [0.75, 1.5] | 0.817684 | 0.373025 | 0 |

Reverse parameters were fitted, not obtained by negating forward parameters.
Steering has gain 1, time constant 0.011196 s, zero selected delay. Its time
constant is below the 0.05 s sampling period and should not be overinterpreted.
The yaw gain remains 1: fitting 0.958249 improved blocked-CV RMSE by only 1.25%,
below the predeclared 2% criterion. Acceleration/deceleration ID error ratio 1.44
and left/right steering gain difference 0.007% did not justify extra branches.

### Tests And Real Isaac Sim Closed Loop

35 offline tests passed, including reverse departure, sign of steering, braking
through zero, frozen prediction consistency, input/rate constraints, zero-speed
hold, reverse fallback, action conversion, rejection of an old forward model with
reverse bounds, and validation artifact hashes/independence.

| Run | Steps | Fallbacks | Mean/max full solve time (ms) |
| --- | ---: | ---: | ---: |
| Reverse fixed | 160 | 0 | 27.85 / 31.43 |
| Reverse heading | 400 | 0 | 28.27 / 44.35 |
| Signed joint | 500 | 0 | 28.24 / 48.51 |
| Signed speed | 400 | 0 | 28.72 / 33.98 |

All 1460 solves reported solved and ended at the evaluation horizon. This is
observed timing, not a real-time guarantee. Fixed reverse speed MAE was 0.03401
m/s, heading MAE 0.000961 rad; last-second mean body speed was -0.49222 m/s.
At reverse speed -0.5 m/s, heading stages +0.2/-0.2 rad settled within the 0.05 rad
band in 0.65/1.30 s; their last-second heading biases were -0.000460/-0.000713 rad.
Bias convention is reference minus measured. All results include actual CSV data.

In the signed-joint run, forward 1.2 m/s saturated the upper physical input on
81% of its stage and retained 0.05153 m/s steady underspeed. The signed-speed run
had 0.03178 m/s steady underspeed at 1.2 m/s. Neither result establishes full-range
tracking acceptance. Large direction changes have substantial transient MAE
(1.2 to -0.5 m/s stage: 0.23558 m/s joint, 0.29442 m/s speed-only).

The final speed-step CSV verified environment reference vs MPC reference within
4.77e-8, physical target mapping within 9.72e-8, and ActionTerm rate-limiter
difference within 2.99e-8. Earlier three runs predate environment-reference
telemetry synchronization but used the same frozen model, bounds, weights and
controller; their logged MPC references were correct. GUI rendering itself was
not visually tested; these checks were headless.

## Result

The reverse interface hypothesis is supported in the tested nominal simulator:
references, identified physical inputs, signed yaw model, QP constraints, adapter,
and executed actions now support reverse consistently. Existing forward data and
PPO artifacts are preserved. There was no MPC weight tuning or validation refit.

## Conclusion

Proceed to broader repeat-seed fixed/step/joint nominal evaluation using the new
defaults. Retain the high-speed saturation and transition-error findings as open
performance work, with new experimental evidence before changes. Do not claim
low-traction robustness, protective behavior, hardware validity, or guaranteed
speed limits. The QP still freezes one local speed region and yaw linearization
over its horizon; cross-region/zero-speed prediction error and absence of offset
estimation remain explicit limitations.
