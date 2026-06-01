# Copilot Instructions for IsaacLab DreamerV3 Car Tasks

## Scope first
- Primary task code lives under `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/`.
- Environment config pattern is centered on `@configclass` + `ManagerBasedRLEnvCfg`.
- Example anchor file: `.../env/car_env_cfg.py`.

## Architecture patterns to follow
- Keep MDP config split into dedicated classes:
  - `MySceneCfg` (assets/sensors/terrain)
  - `ActionsCfg`, `ObservationsCfg`, `EventCfg`, `RewardsCfg`, `TerminationsCfg`
  - final env presets (`LocomotionVelocityRoughEnvCfg`, `MyCar*` variants)
- Use `__post_init__()` for runtime overrides (dt, episode length, device, curriculum-like toggles).
- Prefer inheritance for task variants:
  - base training env -> specialized train env -> `*_PLAY` env for batched visualization/eval.

## Project-specific conventions
- Scene assets are namespaced with `{ENV_REGEX_NS}` and referenced through `SceneEntityCfg(name="...")`.
- Vehicle asset is injected via:
  - `robot: ArticulationCfg = CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")`
- Keep obstacle generation deterministic when based on fixed position tables (e.g., `OBSTACLE_POSITIONS`).
- Keep disabled observation/reward terms as commented blocks when they document optional training signals.

## Action/observation integration rules
- Action config must stay consistent with robot joint names (wheel + steering joint lists).
- `CarVWActionCfg` is used for throttle/steer control; do not silently swap action type in-place.
- When adding observation terms, ensure shape/stability expectations remain compatible with existing policy training.

## Event/reset and termination patterns
- Local navigation reset is centralized in `reset_local_nav_task` with rich params in `EventCfg.reset_local_nav`.
- Scenario variants should override only needed keys via:
  - `self.events.reset_local_nav.params["..."] = ...`
- Keep reward/termination parameter tuning local to env subclass instead of editing base defaults unless globally intended.

## Simulation defaults seen in this codebase
- Typical base values in this task:
  - `decimation = 5`
  - `sim.dt = 0.01`
  - `episode_length_s` task-dependent
- GPU target is set in env variants: `self.sim.device = "cuda:0"`.

## Editing guidance for AI agents
- Make minimal, local edits; preserve config layering and class names.
- Avoid renaming existing terms referenced by registry/config consumers.
- If changing joint names, update all dependent action/observation entity filters together.
- Prefer param tweaks in subclass envs (`MyCarSimpleEnvCfg`, `MyCarRoughEnvCfg`) over base-class churn.

## Key files to inspect before major changes
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3/env/car_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3/mdp/rewards.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3/mdp/observation.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3/mdp/terminations_user.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3/mdp/actions.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3/mdp/events.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/envs/wrappers.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/configs.yaml`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/dreamer.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/exploration.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/isaaclab_adapter.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/models.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/networks.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/parallel.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/play_isaaclab.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/tools.py`
- `source/isaaclab_tasks/isaaclab_tasks/user/dreamerv3_torch/train_isaaclab.py`