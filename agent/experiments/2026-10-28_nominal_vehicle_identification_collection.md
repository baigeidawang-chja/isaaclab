# Nominal Vehicle Identification Data Collection

## Goal

Collect policy-independent nominal Car4WD response data for fitting the speed and steering first-order execution dynamics used by the offline MPC model.

## Hypothesis

The current flat, high-traction `ManagerBasedRLEnv` can execute a predeclared open-loop protocol without early termination, non-finite samples, or steering-limit violations, while keeping identification and validation excitation points independent.

## Baseline

- Commit / working tree: dirty local tree; legacy and user changes were preserved.
- Task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`
- Data source: Isaac Sim, policy bypassed.
- Policy checkpoint: none.
- Control period: 0.05 s.
- Steering constraints: nominal center angle +/-0.2438743454 rad; USD wheel limit +/-0.2617993878 rad.

## Change

- Added `scripts/user/collect_nominal_vehicle_identification.py`.
- Added `mpc/config/nominal_vehicle_identification_protocol.json`.
- Added measured left/right steering to equivalent center-steering reconstruction in `mpc/steering_constraints.py`.
- Extended the steering-constraint unit test for the reconstruction.
- The PPO task, rewards, observations, action configuration, and Gym registration were not modified.

Identification protocol:

- Zero-steering speed steps: 0.0, 0.3, 0.6, 0.9, 1.2, 0.6, 0.3, 0.0 m/s; 4 s per segment.
- Steering steps at 0.4, 0.8, and 1.2 m/s: 0.0, +0.08, 0.0, -0.08, 0.0, +0.16, 0.0, -0.16, 0.0, +0.22, 0.0, -0.22, 0.0 rad; 3 s per segment.

Validation protocol:

- Zero-steering speed steps: 0.0, 0.45, 0.75, 1.05, 0.45, 0.0 m/s; 4 s per segment.
- Steering steps at 0.55 and 1.0 m/s: 0.0, +0.12, 0.0, -0.12, 0.0, +0.20, 0.0, -0.20, 0.0 rad; 3 s per segment.

## Command

```bash
PYTHONUNBUFFERED=1 ./isaaclab.sh -p \
  scripts/user/collect_nominal_vehicle_identification.py \
  --headless --disable_fabric
```

## Run

- Date: 2026-10-28
- Output directory: `/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_identification/20261028_152550`
- Exit code: 0
- Files: `identification.csv`, `validation.csv`, `manifest.json`, `protocol.json`

## Metrics

- Identification rows: 2980 across four profiles.
- Validation rows: 1560 across three profiles.
- Logged fields: 45.
- Non-finite or unparsable numeric values: 0.
- Blank numeric values: 0.
- Terminated or truncated rows: 0.
- Maximum requested steering: 0.22 rad identification; 0.20 rad validation.
- Maximum target front-wheel steering magnitude: 0.2345465 rad identification; 0.2119890 rad validation.
- Maximum measured front-wheel steering magnitude: 0.2361639 rad identification; 0.2136098 rad validation.
- Nonzero identification and validation speed excitation levels are disjoint.
- Nonzero identification and validation steering excitation levels are disjoint.

## Result

The open-loop collection completed successfully. Both CSV files match the manifest row counts, contain finite numeric data, remain inside the audited steering limits, and contain no unexpected termination. The policy was not loaded or evaluated.

## Conclusion

- Hypothesis supported for the Isaac Sim nominal environment.
- This is simulated identification data, not physical-robot data.
- No dynamic parameter or prediction-performance conclusion is made by this collection run.
- Next action: fit candidate first-order speed and steering dynamics using only `identification.csv`, then report prediction error once using the untouched `validation.csv`.

## Subsequent Interface Revision

On 2026-10-29, the CommandFollow steering action scale changed from `0.8 rad`
to `0.243 rad`. This does not change the conditions or manifest of this
historical collection. Its largest requested steering magnitude was
`0.22 rad`, so all collected excitation remains inside the revised action
range and can still be used for identification and held-out validation.
