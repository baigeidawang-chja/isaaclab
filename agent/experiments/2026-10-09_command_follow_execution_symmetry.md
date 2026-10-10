# Experiment

## Goal
Determine whether the persistent positive actual-heading bias in the Phase 1 CommandFollow task is caused by a left/right mismatch or physical asymmetry in the vehicle execution layer.

## Hypothesis
If the execution layer is asymmetric, a policy-free constant-speed command with zero steering will produce a sustained heading drift, or equal positive and negative steering commands will produce materially different yaw-rate magnitudes.

## Baseline
- Commit / working tree: local `main` working tree, with pre-existing uncommitted CommandFollow changes
- Task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`
- Config: current local CommandFollow environment and `CAR_CFG`
- Checkpoint: none; this is a policy-free open-loop test
- Relevant baseline metrics: reported learned-policy actual heading remains biased positive

## Change
- Files: `scripts/user/command_follow_open_loop.py`
- Parameters: speed `0.8 m/s`, steering cases `0.0`, `+0.25`, and `-0.25 rad`, `8 s` per case, first `3 s` excluded from steady-state metrics
- Reason: isolate joint mapping and vehicle physics from PPO and reward behavior

No reward, environment, PPO, asset, or action mapping was changed for this diagnostic.

## Command
```bash
./isaaclab.sh -p scripts/user/command_follow_open_loop.py \
  --headless \
  --disable_fabric \
  --speed 0.8 \
  --steer 0.25 \
  --duration 8.0 \
  --warmup 3.0 \
  --output /media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/diagnostics/open_loop_mapping_20261009.csv
```

## Run
- Start: 2026-10-09
- End: 2026-10-09
- Log directory: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/diagnostics`
- Exit code: 0

## Metrics

### Runtime mapping

The articulation's resolved joint order is:

```text
rear-left wheel, rear-right wheel, front-right steer, front-left steer,
front-right wheel, front-left wheel
```

For the selected action joints, the runtime orders are:

```text
steering ids: [front-right, front-left]
wheel ids:    [rear-left, rear-right, front-right, front-left]
```

WheeledLab Ackermann output semantics are:

```text
steering columns: [front-left, front-right]
wheel columns:    [rear-left, rear-right, front-left, front-right]
```

Therefore the two steering columns and the two front-wheel columns are applied in reversed left/right order. The cause is that `find_joints` defaults to asset order while `AckermannAction.apply_actions` assumes a fixed semantic order.

The policy wheel observation has the same order-contract issue: its configured order is not preserved, so the observed tensor follows asset order rather than the listed `WHEEL_JOINTS` order.

### Open-loop response

| Case | Heading endpoint rate (rad/s) | Mean yaw rate (rad/s) | Yaw-rate std (rad/s) |
|---|---:|---:|---:|
| steering `0.0` | -0.000192 | -0.011363 | 0.056427 |
| steering `+0.25` | +0.494136 | +0.549896 | 0.072872 |
| steering `-0.25` | -0.487601 | -0.546117 | 0.058420 |

- Zero-steering heading linear-regression rate: `+0.000804 rad/s`.
- Positive/negative absolute mean-yaw-rate ratio: `1.006921`.
- Positive/negative odd yaw bias: `+0.001890 rad/s`.
- Zero-steering mean actual wheel speeds were front-left `15.6817`, front-right `16.3520`, rear-left `16.2083`, rear-right `16.1321 rad/s`.
- Zero-steering yaw rate oscillated between approximately `-0.1905` and `+0.1418 rad/s`, but did not accumulate a sustained positive heading drift during this run.

The CSV records every requested signal: raw/target/executed commands, semantic Ackermann outputs, target and actual left/right steering angles, target and actual four-wheel speeds, heading, unwrapped heading, heading error, and yaw rate.

### Asset audit

The composed USD contains physical/modeling asymmetries that warrant a separate asset-quality investigation:

- Left/right wheel lateral positions are not exact mirrors; the discrepancy is approximately `6.1 mm` at the front and `7.6 mm` at the rear.
- Rear suspension joint definitions differ in axis and local transforms (`X` on the left and `Y` on the right).
- Some wheel-link inertias are invalid or zero-authored and PhysX reports that it substitutes approximate inertia.
- Several fixed joints reference missing `contact_*` prims.
- The base body's authored lateral center of mass is centered (`y = 0`), so no obvious lateral COM offset was found.

These findings are suspicious, but this open-loop run does not establish them as the cause of the learned policy's persistent positive-heading bias.

## Result
The joint-order contract is definitively incorrect. However, the policy-free vehicle did not show sustained positive drift at zero steering, and equal positive/negative steering produced yaw-rate magnitudes within about `0.7%` of each other. Thus the current experiment does not support a strong one-sided physical execution response as the direct cause of the reported long-term positive heading.

## Conclusion
- The hypothesis is not supported for a large one-sided open-loop response under the tested command, but an execution mapping defect is confirmed.
- Correct the action and observation ordering in a focused follow-up change, then rerun this exact open-loop diagnostic.
- Do not tune heading reward or analyze PPO steering-mean bias until the execution mapping contract is corrected and symmetry is revalidated.

## Mapping correction follow-up

The task-local action now resolves configured joints with `preserve_order=True`. Its configured semantic orders are:

```text
steering: [front-left, front-right]
wheels:   [rear-left, rear-right, front-left, front-right]
```

The policy wheel observation also resolves its declared joint list with `preserve_order=True`.

The original open-loop command was repeated without changing speed, steering magnitude, duration, warmup, reward, or vehicle asset. The new outputs are:

```text
/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/diagnostics/open_loop_mapping_fixed_20261009.csv
/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_command_follow/diagnostics/open_loop_mapping_fixed_20261009.summary.json
```

### Corrected mapping evidence

For `+0.25 rad`, Ackermann left/right steering outputs `0.268849/0.233576 rad` now match the left/right joint targets respectively. Ackermann front-left/front-right wheel outputs `21.972548/25.215790 rad/s` likewise match their named joint targets.

### Before/after response

| Metric | Before | Corrected |
|---|---:|---:|
| zero-steer heading rate (rad/s) | -0.000192 | -0.000652 |
| zero-steer mean yaw rate (rad/s) | -0.011363 | -0.004636 |
| positive mean yaw rate (rad/s) | +0.549896 | +0.595008 |
| negative mean yaw rate (rad/s) | -0.546117 | -0.591526 |
| positive/negative absolute yaw ratio | 1.006921 | 1.005886 |
| odd yaw bias (rad/s) | +0.001890 | +0.001741 |

The corrected execution remains symmetric to within approximately `0.59%` at the tested steering magnitude, and zero steering still does not produce sustained positive heading drift. The ordering contract is now valid; checkpoint-level steering-mean analysis can proceed as a separate experiment.
