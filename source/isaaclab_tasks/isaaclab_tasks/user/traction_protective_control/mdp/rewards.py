from __future__ import annotations

import torch


def _planner_direction(env, command_name: str = "planner_command") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return torch.stack([torch.cos(command.desired_heading), torch.sin(command.desired_heading)], dim=-1)


def speed_tracking(env, command_name: str = "planner_command", std: float = 0.35) -> torch.Tensor:
    """Reward velocity projected onto the planned heading matching desired speed."""
    command = env.command_manager.get_term(command_name)
    speed_along_plan = torch.sum(env.scene["robot"].data.root_lin_vel_w[:, :2] * _planner_direction(env, command_name), dim=-1)
    error = speed_along_plan - command.desired_speed
    return torch.exp(-torch.square(error / float(std)))


def heading_tracking(env, command_name: str = "planner_command", std: float = 0.35) -> torch.Tensor:
    """Reward small world-frame heading error."""
    command = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(command.heading_error / float(std)))


def upright_stability(env, scale: float = 1.0) -> torch.Tensor:
    """Penalize body tilt away from upright using projected gravity."""
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    tilt_xy = torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
    return -float(scale) * tilt_xy


def yaw_rate_penalty(env, scale: float = 0.05) -> torch.Tensor:
    """Lightly penalize excessive yaw rate beyond what heading tracking needs."""
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return -float(scale) * torch.square(yaw_rate)


def lateral_velocity_penalty(env, scale: float = 0.05) -> torch.Tensor:
    """Lightly penalize lateral body velocity on high-traction flat ground."""
    lateral_velocity = env.scene["robot"].data.root_lin_vel_b[:, 1]
    return -float(scale) * torch.square(lateral_velocity)


def action_smoothness(env, action_term_name: str = "throttle_steer", scale: float = 0.02) -> torch.Tensor:
    """Penalize changes in the physical target action requested by the policy."""
    try:
        action_term = env.action_manager.get_term(action_term_name)
    except (AttributeError, KeyError):
        action = env.action_manager.action
        if not hasattr(env, "_tpc_prev_action") or env._tpc_prev_action.shape != action.shape:
            env._tpc_prev_action = torch.zeros_like(action)
        delta = action - env._tpc_prev_action
        env._tpc_prev_action = action.detach().clone()
        return -float(scale) * torch.sum(torch.square(delta), dim=-1)

    target = getattr(action_term, "target_actions", None)
    prev_target = getattr(action_term, "previous_target_action", None)
    if target is None or prev_target is None:
        return torch.zeros(env.num_envs, device=env.device)
    delta = target - prev_target
    return -float(scale) * torch.sum(torch.square(delta), dim=-1)
