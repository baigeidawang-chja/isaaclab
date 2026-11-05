# Steering Action Scale Alignment

## Goal

Align the CommandFollow normalized steering action with the feasible virtual
center-steering range of the current Ackermann geometry and steering joints.

## Hypothesis

Changing the steering action scale from `0.8 rad` to `0.243 rad` removes the
large infeasible command region and makes normalized policy output, physical
target steering, and nominal MPC constraints use one consistent contract.

Whether this improves heading tracking or training stability remains an
experimental question and is not established by the interface change alone.

## Baseline

- Task: `Isaac-TractionProtect-CommandFollow-Car4WD-v0`
- Previous action scale: `(1.5 m/s, 0.8 rad)`
- Geometric center-steering limit: approximately `+-0.243874 rad`
- Front steering-joint limits: `+-0.261799 rad`
- Previous normalized outputs above approximately `|0.304843|` requested
  steering beyond the unsaturated geometric range.

## Change

- Current action scale: `(1.5 m/s, 0.243 rad)`
- Current normalized steering range: `[-1, 1]`
- Current physical center-steering range: `[-0.243, 0.243] rad`
- Ackermann wheel targets at the action boundary:
  - inner wheel magnitude: approximately `0.260795 rad`
  - outer wheel magnitude: approximately `0.227439 rad`
- Rewards, observations, rate limits, action dimensions, and task registration
  are unchanged.

Synchronized files:

- `AGENTS.md`
- `agent/audits/2026-10-15_nominal_mpc_interface_model_audit.md`
- `mpc/config/nominal_car4wd_steering.json`
- `mpc/steering_constraints.py`
- `scripts/user/command_follow_open_loop.py`
- steering-constraint tests

## Checkpoint Compatibility

Legacy checkpoints trained with the `0.8 rad` scale are network-shape
compatible, so their parameters can be loaded. They are not behaviorally
equivalent because the same normalized steering output now produces only
`0.243 / 0.8 = 0.30375` of its former physical center-steering target.

Use a legacy checkpoint only as an explicitly labeled initialization candidate.
Do not compare it directly against its old evaluation metrics without
re-evaluating under the new action scale.

## Verification

```bash
./isaaclab.sh -p -m unittest \
  source/isaaclab_tasks/test/test_traction_protective_control_steering_constraints.py -v

./isaaclab.sh -p scripts/user/check_nominal_steering_constraints.py
```

- Seven steering-constraint tests passed.
- Effective center bounds: `[-0.243, 0.243] rad`.
- Normalized feasible bounds: `[-1, 1]`.
- Boundary Ackermann wheel targets remain inside both joint limits.
- No PPO training or learned-policy evaluation was run as part of this
  documentation synchronization.

## Next Evaluation

1. Run the policy-free zero and `+-0.22 rad` open-loop symmetry diagnostic.
2. Start a new short PPO run with the revised scale; do not overwrite the
   legacy `0.8 rad` checkpoint.
3. Evaluate speed/heading tracking, steering saturation, action distribution,
   and positive/negative yaw symmetry under the revised interface.
4. Only claim improvement after comparison using separately labeled runs and
   identical command schedules.
