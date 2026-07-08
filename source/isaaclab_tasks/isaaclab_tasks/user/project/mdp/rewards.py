from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply, wrap_to_pi

from . import observations


def _planner_direction(env, command_name: str = "planner_command") -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return torch.stack([torch.cos(command.heading_plan), torch.sin(command.heading_plan)], dim=-1)


def progress_along_plan(env, command_name: str = "planner_command", scale: float = 1.0) -> torch.Tensor:
    robot = env.scene["robot"]
    vel_w = robot.data.root_lin_vel_w[:, :2]
    progress_speed = torch.sum(vel_w * _planner_direction(env, command_name), dim=-1)
    return torch.clamp(progress_speed, -1.0, 2.0) * float(scale)


def heading_following(env, command_name: str = "planner_command", std: float = 0.5) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    err = wrap_to_pi(command.heading_plan - env.scene["robot"].data.heading_w)
    return torch.exp(-torch.square(err / float(std)))


def no_progress_penalty(env, threshold: float = 0.03) -> torch.Tensor:
    plan_dir = _planner_direction(env)
    progress_speed = torch.sum(env.scene["robot"].data.root_lin_vel_w[:, :2] * plan_dir, dim=-1)
    return -(progress_speed < float(threshold)).float()


def slip_penalty(env, scale: float = 0.05) -> torch.Tensor:
    return -float(scale) * observations.wheel_slip_proxy(env)[:, 0].clamp(0.0, 10.0)


def stuck_penalty(env, scale: float = 1.0) -> torch.Tensor:
    return -float(scale) * observations.stuck_label(env).squeeze(-1)


def action_rate_penalty(env, scale: float = 0.02) -> torch.Tensor:
    action = env.action_manager.action
    prev_action = getattr(env, "_failure_aware_prev_action", torch.zeros_like(action))
    penalty = torch.sum(torch.square(action - prev_action), dim=-1)
    env._failure_aware_prev_action = action.detach().clone()
    return -float(scale) * penalty


def plan_speed_tracking(env, command_name: str = "planner_command", std: float = 0.35) -> torch.Tensor:
    """Reward forward speed matching v_plan in the commanded world direction."""
    command = env.command_manager.get_term(command_name)
    plan_dir = torch.stack([torch.cos(command.heading_plan), torch.sin(command.heading_plan)], dim=-1)
    speed_along_plan = torch.sum(env.scene["robot"].data.root_lin_vel_w[:, :2] * plan_dir, dim=-1)
    error = speed_along_plan - command.v_plan
    return torch.exp(-torch.square(error / float(std)))


def heading_tracking(env, command_name: str = "planner_command", std: float = 0.35) -> torch.Tensor:
    """Reward small heading error to heading_plan."""
    command = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(command.heading_error / float(std)))


def yaw_rate_penalty(env, scale: float = 0.25) -> torch.Tensor:
    """Lightly penalize excessive yaw rate."""
    yaw_rate = env.scene["robot"].data.root_ang_vel_b[:, 2]
    return -float(scale) * torch.square(yaw_rate)


def lateral_velocity_penalty(env, scale: float = 0.5) -> torch.Tensor:
    """Lightly penalize lateral body velocity on the straight-line first version."""
    lat_vel = env.scene["robot"].data.root_lin_vel_b[:, 1]
    return -float(scale) * torch.square(lat_vel)
