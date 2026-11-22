# Known issues

Keep concrete, reproduced issues here.

Recommended format:

## <short issue title>
- Reproduction command:
- Observed behavior:
- Expected behavior:
- Relevant files:
- Last known good commit/checkpoint:
- Current hypothesis:
- Status:

Do not add speculative issues that have not been reproduced.

## Historical PPO steering chatter requires current-geometry re-evaluation

- Reproduction command: evaluate
  `2026-10-29_11-38-16_ppo_torch_medium_500_from_short_steer_scale_0243/checkpoints/best_agent.pt`
  with the four `scripts/user/eval_*.py` CommandFollow evaluators.
- Observed behavior: raw steering reaches `|action| >= 1` on 38% to 46% of
  evaluation steps and changes sign on 52% to 57% of steps. The rate limiter
  reduces, but does not remove, the resulting executed steering activity.
- Expected behavior: deterministic mean actions should track heading without
  sustained boundary-to-boundary switching.
- Relevant files: `command_follow_env_cfg.py`, `mdp/actions.py`,
  `skrl_ppo_cfg.yaml`, and `scripts/user/command_follow_evaluation.py`.
- Last known good commit/checkpoint: none established for the revised
  `0.243 rad` action scale.
- Current hypothesis: no directional execution bias is present; policy-output
  sensitivity, observation scaling, and insufficient convergence remain to be
  separated experimentally.
- Status: reproduced on the former execution geometry. Its measured rates are
  historical; current-geometry re-evaluation is pending.

## Historical forward-only model has unequal acceleration and deceleration error

- Reproduction: inspect the identification and validation metrics under
  `traction_protective_control_nominal_identification/20261116_205318/`.
- Observed behavior: identification acceleration/deceleration RMSE ratio is
  `1.69`; held-out validation RMSE is `0.05134/0.01936 m/s` respectively.
- Relevant file: `mpc/config/nominal_car4wd_model.json`.
- Status: retained as a historical forward-only baseline. The 2026-11-20
  bidirectional model is independently fitted and validated, not a bound edit
  or mirrored reverse model. Its identification acceleration/deceleration
  RMSE ratio is 1.44; validation errors remain unequal (0.05837/0.03631 m/s).
  This residual asymmetry is a model limitation, not a reason to refit on
  validation data.

## Wheel-drive torque clipping in current-geometry MPC runs

- Reproduction: fixed `1.2` and `0.9 m/s` runs at
  `traction_protective_control_nominal_mpc/fixed/20261116_214234` and
  `20261116_214250` on the external drive.
- Observed behavior: last-second computed torque exceeded applied torque on
  `96.3%` and `88.7%` of wheel samples. Mean applied torque magnitudes were
  `0.974` and `0.961 Nm`; some actual wheel speeds reversed despite positive
  targets.
- Relevant files: `isaaclab_assets/robots/car_edit.py` and
  `scripts/user/eval_nominal_mpc.py`.
- Status: observed and deferred by the user on 2026-11-19. Keep the current
  actuator/contact settings fixed for nominal MPC work; the clipping data
  remain a limitation of the recorded simulation, not a new work prerequisite.

## Configured Ackermann geometry differs from loaded USD joint anchors

- Reproduction: open `/home/chja/myproject/car_no_damper_04.usd` with USD,
  transform the wheel and steering joint `body1` anchors to world space.
- Observed behavior: axle-center spacing is `0.4070 m`; front steering
  pivot spacing is `0.233459 m`; front/rear wheel-axis lateral spans are
  about `0.26758/0.26906 m`. The ActionTerm, nominal model, and steering
  constraint JSON all use `0.411 m` wheelbase and `0.290 m` width.
- Relevant files: `car_no_damper_04.usd`, `command_follow_env_cfg.py`,
  `mpc/config/nominal_car4wd_model.json`, and
  `mpc/config/nominal_car4wd_steering.json`.
- Status: documented; not a current MPC prerequisite under the user's
  2026-11-19 ideal-Ackermann assumption. Steering-pivot spacing is not tire
  center-plane track width. Keep L=0.411, W=0.290, r=0.065 m and ignore pivot
  offsets for this phase; the anchor measurements above remain historical.

## Reverse MPC interface gap resolved with independent bidirectional data

- Resolved on 2026-11-20: new independent identification/validation cover
  physical inputs [-1.5, 1.5] m/s; planner references remain [-0.5, 1.2] m/s.
- Default parameters/config are `nominal_car4wd_bidirectional_model.json`
  and `nominal_car4wd_bidirectional_osqp_mpc.json`; older files are preserved.
- Heading linearization now preserves measured travel direction, using the
  reference direction only near rest. Zero-speed rest cannot turn in place.
- Validation speed RMSE is 0.04834 m/s (reverse stages 0.04203 m/s).
  See `agent/experiments/2026-11-20_bidirectional_nominal_mpc.md`.
- Remaining limits: locally frozen speed regions/heading dynamics, no offset
  estimator, no hard measured-speed guarantee, and no real-hardware validation.
  Short closed-loop checks are not a general robustness or safety certificate.
