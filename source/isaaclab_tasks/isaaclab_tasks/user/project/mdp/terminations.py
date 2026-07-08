from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def success_distance(env, distance: float = 4.0) -> torch.Tensor:
    robot = env.scene["robot"]
    x = robot.data.root_pos_w[:, 0]
    if hasattr(env.scene, "env_origins"):
        x = x - env.scene.env_origins[:, 0]
    return x >= float(distance)


def is_flipped(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), up_z_threshold: float = 0.25) -> torch.Tensor:
    quat = env.scene[asset_cfg.name].data.root_quat_w
    x = quat[:, 1]
    y = quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return up_z < float(up_z_threshold)


def no_progress_failure(
    env,
    min_progress_speed: float = 0.01,
    max_no_progress_steps: int = 180,
    min_episode_steps: int = 30,
) -> torch.Tensor:
    if not hasattr(env, "_failure_aware_no_progress_steps"):
        env._failure_aware_no_progress_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    speed = env.scene["robot"].data.root_lin_vel_b[:, 0]
    no_progress = speed < float(min_progress_speed)
    env._failure_aware_no_progress_steps = torch.where(
        no_progress,
        env._failure_aware_no_progress_steps + 1,
        torch.zeros_like(env._failure_aware_no_progress_steps),
    )
    mature = env.episode_length_buf >= int(min_episode_steps) if hasattr(env, "episode_length_buf") else True
    return (env._failure_aware_no_progress_steps >= int(max_no_progress_steps)) & mature


def out_of_bounds(
    env,
    x_min: float = -2.0,
    x_max: float = 12.0,
    y_min: float = -3.0,
    y_max: float = 3.0,
) -> torch.Tensor:
    pos = env.scene["robot"].data.root_pos_w[:, :2]
    if hasattr(env.scene, "env_origins"):
        pos = pos - env.scene.env_origins[:, :2]
    x = pos[:, 0]
    y = pos[:, 1]
    return (x < float(x_min)) | (x > float(x_max)) | (y < float(y_min)) | (y > float(y_max))
