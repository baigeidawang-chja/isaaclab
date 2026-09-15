from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def _finite(tensor: torch.Tensor, min_value: float = -1.0e6, max_value: float = 1.0e6) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=max_value, neginf=min_value).clamp(min_value, max_value)


def planner_command(env, command_name: str = "planner_command") -> torch.Tensor:
    """Policy input [desired_speed, heading_error]."""
    command = env.command_manager.get_term(command_name)
    return _finite(torch.stack([command.desired_speed, command.heading_error], dim=-1))


def base_lin_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _finite(env.scene[asset_cfg.name].data.root_lin_vel_b)


def base_ang_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _finite(env.scene[asset_cfg.name].data.root_ang_vel_b)


def projected_gravity(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _finite(env.scene[asset_cfg.name].data.projected_gravity_b)


def wheel_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    joint_vel = robot.data.joint_vel
    if asset_cfg.joint_ids is not None:
        joint_vel = joint_vel[:, asset_cfg.joint_ids]
    return _finite(joint_vel)


def imu_state(env) -> torch.Tensor:
    """Compact real-robot style IMU state: linear acceleration, angular velocity, roll, pitch."""
    imu = env.scene.sensors.get("imu", None)
    if imu is None:
        return torch.zeros((env.num_envs, 8), device=env.device)
    data = imu.data
    roll, pitch, _yaw = math_utils.euler_xyz_from_quat(data.quat_w)
    return _finite(torch.cat([data.lin_acc_b, data.ang_vel_b, roll.unsqueeze(-1), pitch.unsqueeze(-1)], dim=-1))


def executed_action(env, action_term_name: str = "throttle_steer") -> torch.Tensor:
    """Return the physical action executed by the rate-limited action term."""
    fallback = torch.zeros_like(env.action_manager.action)
    try:
        action_term = env.action_manager.get_term(action_term_name)
    except (AttributeError, KeyError):
        return fallback
    action = getattr(action_term, "previous_executed_action", None)
    if action is None:
        action = getattr(action_term, "processed_actions", None)
    if action is None:
        return fallback
    return _finite(action)
