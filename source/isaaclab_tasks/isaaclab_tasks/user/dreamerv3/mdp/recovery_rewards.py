from __future__ import annotations

import torch


def _finite_tensor(tensor: torch.Tensor, min_value: float = -1.0e6, max_value: float = 1.0e6) -> torch.Tensor:
    return torch.nan_to_num(tensor, nan=0.0, posinf=max_value, neginf=min_value).clamp(min_value, max_value)


def _robot_x_local(env) -> torch.Tensor:
    robot = env.scene["robot"]
    x = _finite_tensor(robot.data.root_pos_w[:, 0])
    if hasattr(env.scene, "env_origins"):
        x = x - env.scene.env_origins[:, 0]
    return x


def _wheel_stats(env, wheel_radius: float = 0.035):
    robot = env.scene["robot"]
    wheel_speed = _finite_tensor(robot.data.joint_vel[:, :4]).abs() * float(wheel_radius)
    body_forward = _finite_tensor(robot.data.root_lin_vel_b[:, 0]).abs()
    slip = torch.clamp((wheel_speed - body_forward.unsqueeze(-1)).abs() / (body_forward.unsqueeze(-1) + 0.08), 0.0, 20.0)
    if hasattr(robot.data, "applied_torque") and robot.data.applied_torque is not None:
        torque = _finite_tensor(robot.data.applied_torque[:, :4]).abs()
    elif hasattr(robot.data, "joint_acc") and robot.data.joint_acc is not None:
        torque = 0.02 * _finite_tensor(robot.data.joint_acc[:, :4]).abs()
    else:
        torque = 0.05 * _finite_tensor(robot.data.joint_vel[:, :4]).abs()
    return wheel_speed, body_forward, slip, torch.clamp(torque, 0.0, 100.0)


def _contact_flag(env, force_threshold: float = 2.0) -> torch.Tensor:
    sensor = env.scene.sensors.get("robot_contact_sensor", None)
    if sensor is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    try:
        forces = sensor.data.net_forces_w
    except (AttributeError, RuntimeError):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if forces is None or forces.numel() == 0:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return torch.norm(_finite_tensor(forces), dim=-1).amax(dim=1) > float(force_threshold)


def update_recovery_metrics(
    env,
    min_progress_per_step: float = 0.001,
    slip_threshold: float = 2.0,
    spin_speed_threshold: float = 0.18,
    low_body_speed_threshold: float = 0.04,
) -> dict[str, torch.Tensor]:
    """Refresh per-step blocked-recovery metrics once per env step."""
    step_id = int(getattr(env, "common_step_counter", -1))
    if getattr(env, "_blocked_recovery_update_step", -1) == step_id:
        return env._blocked_recovery_metrics

    x = _robot_x_local(env)
    if not hasattr(env, "_blocked_recovery_prev_x"):
        env._blocked_recovery_prev_x = x.detach().clone()
        env._blocked_recovery_start_x = x.detach().clone()
        env._blocked_recovery_best_progress = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env._blocked_recovery_progress_gain = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env._blocked_recovery_no_progress_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._blocked_recovery_slip_duration = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env._blocked_recovery_contact_duration = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        env._blocked_recovery_energy_proxy = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    delta_x = _finite_tensor(x - env._blocked_recovery_prev_x)
    env._blocked_recovery_prev_x = x.detach().clone()
    env._blocked_recovery_delta_x = delta_x
    progress = _finite_tensor(x - env._blocked_recovery_start_x)
    prev_best = env._blocked_recovery_best_progress
    progress_gain = _finite_tensor(torch.clamp(progress - prev_best, min=0.0))
    env._blocked_recovery_best_progress = torch.maximum(prev_best, progress.detach())
    env._blocked_recovery_progress_gain = progress_gain

    wheel_speed, body_forward, slip, torque = _wheel_stats(env)
    mean_slip = slip.mean(dim=-1)
    mean_wheel_speed = wheel_speed.mean(dim=-1)
    spin_ineffective = (mean_wheel_speed > spin_speed_threshold) & (body_forward < low_body_speed_threshold)
    little_progress = progress_gain < float(min_progress_per_step)
    primitive_id = getattr(
        env,
        "_recovery_primitive_id",
        torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
    )
    is_reverse = primitive_id == 2
    within_allowed_retreat = progress > -0.35
    allow_reverse_recovery = is_reverse & within_allowed_retreat
    little_progress = little_progress & (~allow_reverse_recovery)
    env._blocked_recovery_no_progress_steps = torch.where(
        little_progress,
        env._blocked_recovery_no_progress_steps + 1,
        torch.zeros_like(env._blocked_recovery_no_progress_steps),
    )
    env._blocked_recovery_slip_duration += (mean_slip > slip_threshold).float()
    env._blocked_recovery_contact_duration += _contact_flag(env).float()
    env._blocked_recovery_energy_proxy += torque.mean(dim=-1)

    metrics = {
        "x": x,
        "progress": progress,
        "best_progress": env._blocked_recovery_best_progress,
        "progress_gain": progress_gain,
        "delta_x": delta_x,
        "mean_slip": mean_slip,
        "max_slip": slip.max(dim=-1).values,
        "mean_wheel_speed": mean_wheel_speed,
        "body_forward": body_forward,
        "spin_ineffective": spin_ineffective.float(),
        "torque_mean": torque.mean(dim=-1),
        "torque_max": torque.max(dim=-1).values,
    }
    env._blocked_recovery_metrics = metrics
    env._blocked_recovery_update_step = step_id
    return metrics


def effective_displacement(env, scale: float = 8.0, max_delta: float = 0.08) -> torch.Tensor:
    """Reward only new best forward progress, not raw per-step displacement."""
    metrics = update_recovery_metrics(env)
    return _finite_tensor(scale * torch.clamp(metrics["progress_gain"], min=0.0, max=max_delta))


def success_bonus(env, scale: float = 5.0, success_distance: float | None = None) -> torch.Tensor:
    metrics = update_recovery_metrics(env)
    distance = float(success_distance if success_distance is not None else getattr(env, "_blocked_recovery_success_distance", 1.15))
    return _finite_tensor(scale * (metrics["progress"] >= distance).float())


def time_penalty(env, penalty_per_step: float = -0.002) -> torch.Tensor:
    return torch.full((env.num_envs,), float(penalty_per_step), dtype=torch.float32, device=env.device)


def wheel_spin_penalty(env, scale: float = 0.08) -> torch.Tensor:
    metrics = update_recovery_metrics(env)
    return _finite_tensor(-float(scale) * metrics["spin_ineffective"])


def slip_penalty(env, scale: float = 0.02) -> torch.Tensor:
    metrics = update_recovery_metrics(env)
    return _finite_tensor(-float(scale) * torch.clamp(metrics["mean_slip"], 0.0, 10.0))


def torque_proxy_penalty(env, scale: float = 0.002) -> torch.Tensor:
    metrics = update_recovery_metrics(env)
    return _finite_tensor(-float(scale) * metrics["torque_mean"])


def action_switch_penalty(env, scale: float = 0.04) -> torch.Tensor:
    pulse = getattr(env, "_recovery_action_switch_pulse", torch.zeros(env.num_envs, device=env.device))
    return _finite_tensor(-float(scale) * pulse)


def invalid_action_penalty(env, scale: float = 0.08) -> torch.Tensor:
    pulse = getattr(env, "_recovery_invalid_action_pulse", torch.zeros(env.num_envs, device=env.device))
    return _finite_tensor(-float(scale) * pulse)


def retreat_penalty(env, scale: float = 0.2, allowed_retreat: float = 0.35) -> torch.Tensor:
    metrics = update_recovery_metrics(env)
    excess = torch.clamp(-(metrics["progress"] + float(allowed_retreat)), min=0.0)
    return _finite_tensor(-float(scale) * excess)
