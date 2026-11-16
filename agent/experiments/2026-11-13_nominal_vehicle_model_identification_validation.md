# Nominal Vehicle Model Identification and Independent Validation

## Goal

Identify a nominal Car4WD model from policy-independent flat, high-traction
data and evaluate the frozen model once on a disjoint validation set. This
experiment does not implement an MPC solver or closed-loop control.

## Baseline and interface audit

- Base commit: `9e581b7ceb4be3abc4aba35ee2a2e2ec652560e0`.
- Branch and `main` both pointed to the base commit before editing.
- Isaac Lab Python: Python 3.10.16, NumPy 1.26.0, SciPy 1.10.1,
  Matplotlib 3.8.4, OSQP 0.6.5.
- Control period: `0.05 s`.
- Current environment and `nominal_car4wd_steering.json` steering scale:
  `0.24 rad`.
- Existing `0.243 rad` documents are historical regression records.
- The action scale was not changed by this experiment.

## Data isolation

Identification source only:

```text
/media/chja/CE54D158C95990271/IsaacLabTrainingData/
traction_protective_control_nominal_identification/20261028_152550/
identification.csv
```

- Rows: 2980.
- SHA-256:
  `67f609b63243bf65fb50a815855101459fa3b14a6275643968f1071c09f4e031`.

The parameter file was frozen before `validation.csv` was loaded. Its
SHA-256 at independent evaluation was:

```text
c646c66c91010a8803a0972b4f7a211e0c42da72a7ed0a86b25525ab8a37043d
```

Validation source:

- Rows: 1560.
- SHA-256:
  `45063905773faaf40dc31fca8edb33b5ae019e589bac049e9766ac19bce15234`.
- No validation sample was used to select structure, delay, region,
  coefficient, or empirical gain.

## Method

### Longitudinal speed

Candidate discrete delays were 0 through 6 samples. A single linear gain was
not adequate because the identification steady response changes from about
`0.3 -> 0.51 m/s` and `0.6 -> 0.94 m/s` to `1.2 -> 1.18 m/s`.

The selected minimal model is a four-region gain-scheduled first-order model:

```text
v[k] = a_i v[k-1] + b_i v_cmd[k]
```

where `i` is selected by the physical, rate-limited speed command. Region
boundaries are midpoints between identification command levels. Structure and
delay use identification BIC; candidates within delta BIC 2 prefer fewer
parameters and then shorter delay. Segment holdout simulation is retained as
a diagnostic, but is not used alone because holding out a unique operating
level confounds gain coverage and delay.

| Command region (m/s) | a | b | DC gain | Time constant (s) |
|---|---:|---:|---:|---:|
| [0.00, 0.45) | 0.621862 | 0.646049 | 1.708499 | 0.105255 |
| [0.45, 0.75) | 0.727056 | 0.425624 | 1.559382 | 0.156862 |
| [0.75, 1.05) | 0.844007 | 0.203633 | 1.305397 | 0.294820 |
| [1.05, 1.20] | 0.938013 | 0.061613 | 0.993970 | 0.781354 |

Selected delay: 0 samples. After speed scheduling, identification acceleration
and deceleration RMSE are `0.02735` and `0.03281 m/s`, with a ratio of
`1.20`. Separate acceleration/deceleration parameters are not justified.

### Equivalent center steering

The selected model is:

```text
delta[k] = 0.0106801 delta[k-1] + 0.9879822 delta_cmd[k]
```

- Delay: 0 samples.
- DC gain: 0.998648.
- Time constant: 0.011015 s, below one 0.05 s control interval.
- Left/right fitted DC gains: 0.998009 and 0.999237.
- Relative left/right gain difference: 0.123%.

BIC detects the very small left/right residual difference because of the
sample count, but its physical gain difference is below the declared 5%
segmentation threshold. A left/right split is not used.

### Yaw rate

The measured-state relation is:

```text
r = k_r v_x tan(delta) / L
L = 0.407 m
k_r = 0.9413673454
```

The single empirical gain was selected using identification only. Compared
with unit gain, blocked leave-one-speed-run-out RMSE improves by 2.58%, and
delta BIC is 126.08. No speed- or direction-dependent yaw gain is added.

## Metrics

| Output | Identification MAE | Identification RMSE | Validation MAE | Validation RMSE |
|---|---:|---:|---:|---:|
| Body speed (m/s) | 0.01892 | 0.02790 | 0.06552 | 0.09631 |
| Equivalent steering (rad) | 0.000250 | 0.000333 | 0.000268 | 0.000354 |
| Yaw rate (rad/s) | 0.04853 | 0.06348 | 0.04818 | 0.06353 |

Validation coefficients were not refitted. Validation speed `R2=0.9563`;
steering `R2=0.99999`; yaw rate `R2=0.9414`.

## Residual analysis

- Validation speed bias is `-0.00161 m/s`; maximum absolute error is
  `0.31597 m/s`.
- Validation acceleration RMSE is `0.12334 m/s`; deceleration RMSE is
  `0.07072 m/s`. The larger unseen acceleration error must be retained as
  nominal-model uncertainty rather than removed by validation tuning.
- Validation steering bias is `-6.84e-5 rad`; maximum absolute error is
  `0.001323 rad`.
- Validation yaw bias is `0.00144 rad/s`; maximum absolute error is
  `0.26131 rad/s`.
- The empirical yaw gain improves validation RMSE from `0.06544` to
  `0.06353 rad/s`, consistent with the identification-only decision.
- Prediction curves, residual histograms, per-speed, per-direction, and
  acceleration/deceleration metrics are stored with the CSV predictions.

## Applicability

The model is valid only for the recorded Isaac Sim Car4WD configuration:

- flat, fixed high-traction surface;
- 0.05 s control period;
- forward executed speed commands from 0 to 1.2 m/s;
- center-steering commands within the identified +/-0.22 rad range;
- current 0.24 rad action-interface limit;
- wheelbase 0.407 m and track width 0.23346 m.

It is not validated for reverse, low traction, obstacles, different vehicle
mass/actuators, a different control period, real hardware, or steering beyond
the identification range. The simulated steering time constant is shorter
than one control sample and should not be interpreted as a hardware servo
bandwidth measurement.

## Reproduction

Fit and freeze parameters using identification only:

```bash
./isaaclab.sh -p scripts/user/fit_nominal_vehicle_model.py fit \
  --data-dir /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_identification/20261028_152550 \
  --output-dir <new-identification-output-directory> \
  --parameter-output source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/mpc/config/nominal_car4wd_model.json \
  --max-delay-steps 6
```

Evaluate the already frozen parameters:

```bash
./isaaclab.sh -p scripts/user/fit_nominal_vehicle_model.py validate \
  --data-dir /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_identification/20261028_152550 \
  --output-dir <new-validation-output-directory> \
  --parameters source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/mpc/config/nominal_car4wd_model.json
```

Final artifacts:

```text
/media/chja/CE54D158C95990271/IsaacLabTrainingData/
traction_protective_control_nominal_identification/20261028_152550/
nominal_model_fit_20261113_identification_final/

/media/chja/CE54D158C95990271/IsaacLabTrainingData/
traction_protective_control_nominal_identification/20261028_152550/
nominal_model_validation_20261113/
```

## OSQP readiness

The model is suitable for the next offline nominal OSQP stage under these
conditions:

1. Select or freeze one longitudinal operating region over each QP horizon.
2. Linearize `v_x tan(delta) / L` around the nominal reference trajectory.
3. Include the measured validation speed and yaw residuals in offline
   robustness and constraint-margin studies.
4. Keep commands inside the identified range and do not claim low-traction or
   hardware validity.

It is not a single globally linear model over 0 to 1.2 m/s. Optimizing region
switching inside the horizon would require mixed-integer or nonlinear logic
and is outside a plain OSQP QP. This experiment therefore supports proceeding
to local, region-fixed offline nominal MPC formulation, not to simulator
closed-loop deployment.
