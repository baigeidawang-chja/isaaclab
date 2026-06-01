import torch
from isaaclab.managers import SceneEntityCfg
from . import observation
from . import rewards

def _get_env_origins_xy(env) -> torch.Tensor:
    """Best-effort fetch of per-env origin offsets (xy). Falls back to zeros."""
    device = env.device if hasattr(env, "device") else None
    num_envs = env.num_envs if hasattr(env, "num_envs") else None

    origins = None
    if hasattr(env, "scene") and hasattr(env.scene, "env_origins"):
        origins = env.scene.env_origins  # (N, 3)
    elif hasattr(env, "env_origins"):
        origins = env.env_origins

    if origins is None:
        if num_envs is None:
            # last resort: infer from robot state later
            raise AttributeError("Cannot find env origins on env/scene.")
        return torch.zeros((num_envs, 2), device=device)

    return origins[:, :2]


def out_of_bounds(
    env,
    x_min: float = -10.0,
    x_max: float = 120.0,
    y_min: float = -20.0,
    y_max: float = 20.0,
    use_env_frame: bool = True,
):
    robot = env.scene["robot"]
    p_w = robot.data.root_pos_w[:, :2]  # (N,2)

    if use_env_frame:
        origins_xy = _get_env_origins_xy(env)  # (N,2)
        p = p_w - origins_xy
    else:
        p = p_w

    x, y = p[:, 0], p[:, 1]
    return (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)


def is_flipped(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), up_z_threshold: float = 0.2) -> torch.Tensor:
    """Terminate when robot is flipped/rolled over.

    Robust rule: compute base "up" vector in world frame and terminate if its z component is too small.
      - upright: up_z ~ 1
      - on side: up_z ~ 0
      - upside down: up_z ~ -1
    """
    asset = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w  # expected (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # up_z for rotating local +Z into world frame: up_z = 1 - 2(x^2 + y^2)
    up_z = 1.0 - 2.0 * (x * x + y * y)

    # flipped if base up has insufficient world-z component
    flipped = up_z < float(up_z_threshold)
    return flipped.to(torch.bool)


def lateral_error_exceeded(env, max_lateral_error: float = 1.2) -> torch.Tensor:
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    return track["lateral_error"].abs() > max_lateral_error


def heading_error_exceeded(env, max_heading_error: float = 1.0) -> torch.Tensor:
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    return track["heading_error"].abs() > max_heading_error


def target_distance_exceeded(env, max_target_distance: float = 3.0) -> torch.Tensor:
    """Terminate when robot drifts too far away from current target waypoint."""
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    return track["target_distance"] > max_target_distance


def progress_state_tick(env) -> torch.Tensor:
    """Explicit per-step progress update (no termination).

    This keeps waypoint-index/progress state refresh independent of any specific
    termination term and guarantees one update per environment step.
    """
    rewards.update_local_progress_state(env)
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def no_progress_too_long(
    env,
    max_no_progress_steps: int = 80,
    min_progress_per_step: float = 0.002,
    warmup_steps: int = 20,
) -> torch.Tensor:
    # Consume the explicit per-step progress update; fallback if order changes.
    step_id = int(getattr(env, "common_step_counter", -1))
    if getattr(env, "_local_nav_progress_update_step", -1) != step_id:
        rewards.update_local_progress_state(env)
    if hasattr(env, "_local_nav_episode_steps"):
        env._local_nav_episode_steps = env._local_nav_episode_steps + 1
    if hasattr(env, "_local_nav_last_delta_s"):
        active = env._local_nav_episode_steps >= warmup_steps if hasattr(env, "_local_nav_episode_steps") else True
        stalled = env._local_nav_last_delta_s < min_progress_per_step
        env._local_nav_no_progress_steps = torch.where(
            active & stalled,
            env._local_nav_no_progress_steps + 1,
            torch.zeros_like(env._local_nav_no_progress_steps),
        )
    return env._local_nav_no_progress_steps >= max_no_progress_steps


def local_goal_reached(env) -> torch.Tensor:
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    reach_thresh = float(getattr(env, "_local_nav_waypoint_reach_thresh", 0.45))
    return (env._local_nav_progress_idx >= env._local_nav_goal_idx) & (track["target_distance"] <= reach_thresh)
