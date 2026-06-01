from __future__ import annotations

import torch
import isaaclab.envs.mdp as mdp

from . import observation


def time_penalty(env, penalty_per_step: float = -0.002) -> torch.Tensor:
    return torch.full((env.num_envs,), penalty_per_step, device=env.device)


def update_local_progress_state(env) -> torch.Tensor:
    """Refresh cached per-step progress delta used by recovery and termination logic."""
    step_id = int(getattr(env, "common_step_counter", -1))
    if getattr(env, "_local_nav_progress_update_step", -1) != step_id:
        prev_idx = env._local_nav_progress_idx.clone() if hasattr(env, "_local_nav_progress_idx") else None
        track = observation.compute_local_nav_tracking(env, update_progress=True)
        delta_s = track["s"] - env._local_nav_prev_s
        env._local_nav_last_delta_s[:] = delta_s
        env._local_nav_prev_s[:] = track["s"]
        if prev_idx is not None:
            env._local_nav_waypoint_advanced[:] = (env._local_nav_progress_idx > prev_idx).float()
        else:
            env._local_nav_waypoint_advanced[:] = 0.0
        env._local_nav_progress_update_step = step_id
        env._local_nav_cached_track = {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in track.items()}
    else:
        track = getattr(env, "_local_nav_cached_track", None)
        if track is None:
            track = observation.compute_local_nav_tracking(env, update_progress=False)
        delta_s = env._local_nav_last_delta_s
    return delta_s


def local_waypoint_reached_bonus(env, bonus: float = 1.0) -> torch.Tensor:
    """Stage reward: positive pulse whenever the current waypoint index advances by one."""
    update_local_progress_state(env)
    return bonus * env._local_nav_waypoint_advanced


def local_progress_reward(
    env,
    scale: float = 2.0,
    min_delta: float = -0.2,
    max_delta: float = 0.4,
) -> torch.Tensor:
    """Main dense reward: forward progress along the local reference path."""
    delta_s = update_local_progress_state(env)
    clipped = torch.clamp(delta_s, min=min_delta, max=max_delta)
    return scale * clipped

def waypoint_exp_progress_reward(env, alpha=2.0, scale=1.0, clip=0.2):
    """Potential-based waypoint progress reward with safe lazy init.

    Uses current target distance from local tracking and keeps an internal
    previous-distance buffer per env to compute a smooth progress delta.
    """
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    d_curr = track["target_distance"]

    if not hasattr(env, "_local_nav_prev_dist_to_waypoint"):
        env._local_nav_prev_dist_to_waypoint = d_curr.detach().clone()
        return torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    d_prev = env._local_nav_prev_dist_to_waypoint
    if d_prev.shape != d_curr.shape:
        d_prev = d_curr.detach().clone()

    phi_curr = torch.exp(-alpha * d_curr)
    phi_prev = torch.exp(-alpha * d_prev)
    reward = scale * (phi_curr - phi_prev)
    reward = torch.clamp(reward, min=-clip, max=clip)

    env._local_nav_prev_dist_to_waypoint = d_curr.detach().clone()
    return reward

def local_progress_delta_bonus(env, scale: float = 1.0) -> torch.Tensor:
    """Auxiliary reward matching the attachment's explicit progress-delta target."""
    return scale * torch.clamp(env._local_nav_last_delta_s, min=0.0)


def local_lateral_error_penalty(env, scale: float = 1.0) -> torch.Tensor:
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    return -scale * track["lateral_error"].abs()


def local_heading_error_penalty(env, scale: float = 0.5) -> torch.Tensor:
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    return -scale * track["heading_error"].abs()


def local_heading_alignment_reward(
    env,
    std: float = 0.35,
    directional: bool = True,
    reverse_floor: float = 0.15,
    reverse_speed_thresh: float = 0.05,
) -> torch.Tensor:
    """Heading alignment reward with optional forward/reverse directionality.

    - Base term: gaussian on heading error magnitude.
    - Directional term: scales by max(cos(err), 0), so facing opposite direction
      gets near-zero reward.
    - Reverse floor: when actively reversing, keep a small floor so the agent can
      still back up for recovery without killing reward entirely.
    """
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    err = track["heading_error"]
    base = torch.exp(-0.5 * (err / std) ** 2)
    if not directional:
        return base

    dir_scale = torch.clamp(torch.cos(err), min=0.0, max=1.0)
    v_bx = mdp.base_lin_vel(env)[:, 0]
    reversing = v_bx < -abs(reverse_speed_thresh)
    dir_scale = torch.where(
        reversing,
        torch.clamp(reverse_floor + (1.0 - reverse_floor) * dir_scale, 0.0, 1.0),
        dir_scale,
    )
    return base * dir_scale


def local_target_heading_alignment_reward(
    env,
    std: float = 0.35,
    directional: bool = True,
    reverse_floor: float = 0.15,
    reverse_speed_thresh: float = 0.05,
) -> torch.Tensor:
    """Align robot heading with the bearing from robot to current target waypoint."""
    (
        path_library,
        path_ids,
        start_idx,
        progress_idx,
        goal_idx,
        _prev_s,
        _start_s,
        _no_progress_steps,
    ) = observation._local_nav_state(env)
    robot_xy = observation._robot_xy_local(env)
    robot_yaw = env.scene["robot"].data.heading_w

    bearing_err = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for env_id in range(env.num_envs):
        path = path_library[int(path_ids[env_id].item())]
        tgt = int(progress_idx[env_id].item())
        tgt = max(int(start_idx[env_id].item()), min(tgt, int(goal_idx[env_id].item())))
        tgt_xy = path["xy"][tgt]
        rel = tgt_xy - robot_xy[env_id]
        target_bearing = torch.atan2(rel[1], rel[0])
        bearing_err[env_id] = torch.atan2(
            torch.sin(robot_yaw[env_id] - target_bearing),
            torch.cos(robot_yaw[env_id] - target_bearing),
        )

    base = torch.exp(-0.5 * (bearing_err / std) ** 2)
    if not directional:
        return base
    dir_scale = torch.clamp(torch.cos(bearing_err), min=0.0, max=1.0)
    v_bx = mdp.base_lin_vel(env)[:, 0]
    reversing = v_bx < -abs(reverse_speed_thresh)
    dir_scale = torch.where(
        reversing,
        torch.clamp(reverse_floor + (1.0 - reverse_floor) * dir_scale, 0.0, 1.0),
        dir_scale,
    )
    return base * dir_scale


def local_action_smoothness_penalty(env, scale: float = 0.02) -> torch.Tensor:
    action = mdp.last_action(env)
    if not hasattr(env, "_local_nav_prev_action"):
        env._local_nav_prev_action = action.detach().clone()
        return torch.zeros(env.num_envs, device=env.device)
    delta = action - env._local_nav_prev_action
    env._local_nav_prev_action = action.detach().clone()
    return -scale * torch.sum(delta * delta, dim=-1)


def local_stuck_penalty(env, patience: int = 60, scale: float = 1.0) -> torch.Tensor:
    update_local_progress_state(env)
    return -scale * (env._local_nav_no_progress_steps >= patience).float()


def local_escape_contact_bonus(
    env,
    min_speed: float = 0.15,
    stuck_steps: int = 20,
    scale: float = 2.0,
) -> torch.Tensor:
    """Reward escape from blockage using contact when available, otherwise a stuck proxy."""
    contact = observation.get_contact_sensor_feedback(env)[:, 1] > 0.5
    stuck_now = env._local_nav_no_progress_steps >= stuck_steps
    blocked = contact | stuck_now
    if not hasattr(env, "_local_nav_prev_in_contact"):
        env._local_nav_prev_in_contact = blocked.detach().clone()
    escaped = env._local_nav_prev_in_contact & (~blocked)
    env._local_nav_prev_in_contact = blocked.detach().clone()
    speed_ok = mdp.base_lin_vel(env)[:, 0] > min_speed
    return scale * (escaped & speed_ok).float()


def local_recovery_progress_bonus(
    env,
    stuck_steps: int = 20,
    scale: float = 2.0,
) -> torch.Tensor:
    """When recently stuck, reward positive progress increments to encourage unblocking."""
    update_local_progress_state(env)
    stuck = env._local_nav_no_progress_steps >= stuck_steps
    delta = torch.clamp(env._local_nav_last_delta_s, min=0.0)
    return scale * stuck.float() * delta


def local_forward_progress_reward(env, scale: float = 6.0, max_delta: float = 0.15) -> torch.Tensor:
    """Dense reward for making forward progress along the current waypoint segment."""
    delta_s = update_local_progress_state(env)
    return scale * torch.clamp(delta_s, min=0.0, max=max_delta)


def local_reverse_penalty(
    env,
    speed_scale: float = 0.6,
    heading_threshold: float = 0.35,
    allow_when_stuck_steps: int = 20,
) -> torch.Tensor:
    """Discourage reverse driving during normal tracking while allowing recovery maneuvers."""
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    backward_speed = torch.clamp(-mdp.base_lin_vel(env)[:, 0], min=0.0)
    well_aligned = track["heading_error"].abs() < heading_threshold
    not_stuck = env._local_nav_no_progress_steps < allow_when_stuck_steps
    penalize = well_aligned & not_stuck
    return -speed_scale * backward_speed * penalize.float()


def local_success_reward(env, scale: float = 10.0) -> torch.Tensor:
    track = observation.compute_local_nav_tracking(env, update_progress=False)
    success = (env._local_nav_progress_idx >= env._local_nav_goal_idx) & (track["target_distance"] <= getattr(env, "_local_nav_waypoint_reach_thresh", 0.45))
    return scale * success.float()


def local_speed_tracking_reward(env, target_speed: float = 0.8, std: float = 0.35) -> torch.Tensor:
    """High-level action target is speed/turn rate, so keep a weak speed-tracking prior."""
    v_b = mdp.base_lin_vel(env)[:, 0]
    err = v_b - target_speed
    return torch.exp(-0.5 * (err / std) ** 2)
