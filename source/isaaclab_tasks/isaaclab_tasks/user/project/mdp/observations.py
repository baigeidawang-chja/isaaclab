from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def _finite(tensor: torch.Tensor, min_value: float = -1.0e6, max_value: float = 1.0e6) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=max_value, neginf=min_value).clamp(min_value, max_value)


def planner_command(env, command_name: str = "planner_command") -> torch.Tensor:
    """Policy input [v_plan, heading_error]."""
    command = env.command_manager.get_term(command_name)
    return _finite(torch.stack([command.v_plan, command.heading_error], dim=-1))


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
    """Compact IMU/orientation observation: linear accel, angular vel, roll, pitch."""
    imu = env.scene.sensors.get("imu", None)
    if imu is None:
        return torch.zeros((env.num_envs, 8), device=env.device)
    data = imu.data
    roll, pitch, _yaw = math_utils.euler_xyz_from_quat(data.quat_w)
    return _finite(torch.cat([data.lin_acc_b, data.ang_vel_b, roll.unsqueeze(-1), pitch.unsqueeze(-1)], dim=-1))


def wheel_slip_proxy(env, wheel_radius: float = 0.035) -> torch.Tensor:
    wheel_speed = torch.abs(env.scene["robot"].data.joint_vel[:, :4]) * float(wheel_radius)
    body_speed = torch.abs(env.scene["robot"].data.root_lin_vel_b[:, 0]).unsqueeze(-1)
    slip = torch.clamp(torch.abs(wheel_speed - body_speed) / (body_speed + 0.08), 0.0, 20.0)
    return _finite(torch.stack([slip.mean(dim=-1), slip.max(dim=-1).values], dim=-1))


def stuck_label(env, min_progress_speed: float = 0.03, wheel_speed_threshold: float = 0.18) -> torch.Tensor:
    robot = env.scene["robot"]
    body_forward = torch.abs(robot.data.root_lin_vel_b[:, 0])
    wheel_speed = torch.abs(robot.data.joint_vel[:, :4]).mean(dim=-1) * 0.035
    return ((body_forward < min_progress_speed) & (wheel_speed > wheel_speed_threshold)).float().unsqueeze(-1)


def slip_label(env, slip_threshold: float = 2.5) -> torch.Tensor:
    return (wheel_slip_proxy(env)[:, 0] > slip_threshold).float().unsqueeze(-1)


def front_traction_label(env, slip_threshold: float = 2.0) -> torch.Tensor:
    robot = env.scene["robot"]
    wheel_speed = torch.abs(robot.data.joint_vel[:, :4]) * 0.035
    body_speed = torch.abs(robot.data.root_lin_vel_b[:, 0]).unsqueeze(-1)
    slip = torch.clamp(torch.abs(wheel_speed - body_speed) / (body_speed + 0.08), 0.0, 20.0)
    return (slip[:, :2].mean(dim=-1) < slip_threshold).float().unsqueeze(-1)


def rear_traction_label(env, slip_threshold: float = 2.0) -> torch.Tensor:
    robot = env.scene["robot"]
    wheel_speed = torch.abs(robot.data.joint_vel[:, :4]) * 0.035
    body_speed = torch.abs(robot.data.root_lin_vel_b[:, 0]).unsqueeze(-1)
    slip = torch.clamp(torch.abs(wheel_speed - body_speed) / (body_speed + 0.08), 0.0, 20.0)
    return (slip[:, 2:4].mean(dim=-1) < slip_threshold).float().unsqueeze(-1)


def abort_required_label(env) -> torch.Tensor:
    stuck = stuck_label(env).squeeze(-1) > 0.5
    slip = slip_label(env).squeeze(-1) > 0.5
    flipped_risk = env.scene["robot"].data.projected_gravity_b[:, 2] > -0.45
    return (stuck | slip | flipped_risk).float().unsqueeze(-1)


def continue_feasible_label(env) -> torch.Tensor:
    return (1.0 - abort_required_label(env)).clamp(0.0, 1.0)

