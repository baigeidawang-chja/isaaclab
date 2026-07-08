from __future__ import annotations

import numpy as np
import torch

from isaaclab.assets import RigidObjectCollection
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device)
    if not isinstance(env_ids, torch.Tensor):
        return torch.tensor(env_ids, device=env.device, dtype=torch.long)
    return env_ids.to(device=env.device, dtype=torch.long)


def reset_runtime_buffers(env: ManagerBasedEnv, env_ids: torch.Tensor | None):
    """Clear per-episode runtime buffers such as previous action."""
    env_ids = _resolve_env_ids(env, env_ids)

    if hasattr(env, "action_manager"):
        action = env.action_manager.action
        if not hasattr(env, "_failure_aware_prev_action") or env._failure_aware_prev_action.shape != action.shape:
            env._failure_aware_prev_action = torch.zeros_like(action)
        else:
            env._failure_aware_prev_action[env_ids] = 0.0
        for term_name in env.action_manager.active_terms:
            term = env.action_manager.get_term(term_name)
            if hasattr(term, "reset"):
                term.reset(env_ids=env_ids)

    if hasattr(env, "_failure_aware_no_progress_steps"):
        env._failure_aware_no_progress_steps[env_ids] = 0


def reset_failure_aware_task(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    obstacle_asset_cfg: SceneEntityCfg,
    start_x_range: tuple[float, float] = (-0.05, 0.05),
    start_y_range: tuple[float, float] = (-0.08, 0.08),
    heading_range: tuple[float, float] = (-0.08, 0.08),
    root_z: float = 0.18,
    curb_x_range: tuple[float, float] = (1.10, 1.45),
    curb_height_range: tuple[float, float] = (0.025, 0.07),
    low_mu_x_range: tuple[float, float] = (2.00, 2.50),
    low_mu_length_range: tuple[float, float] = (0.8, 1.4),
):
    """Reset robot, placeholder curb, and low-traction visual marker.

    The low-traction region is a first-version placeholder. It is intentionally not
    exposed to policy observations; only labels may infer heuristic risk from body response.
    """
    env_ids = _resolve_env_ids(env, env_ids)

    robot = env.scene[asset_cfg.name]
    obstacles: RigidObjectCollection = env.scene[obstacle_asset_cfg.name]
    num_objects = obstacles.num_objects
    device = env.device
    rng = np.random.default_rng()
    origins = env.scene.env_origins if hasattr(env.scene, "env_origins") else torch.zeros((env.num_envs, 3), device=device)

    root_pose = robot.data.default_root_state[env_ids, :7].clone()
    root_vel = torch.zeros((len(env_ids), 6), device=device)
    object_states = torch.zeros((len(env_ids), num_objects, 13), device=device)
    object_states[:, :, 2] = -10.0
    object_states[:, :, 3] = 1.0

    if not hasattr(env, "_failure_aware_curb_height"):
        env._failure_aware_curb_height = torch.zeros(env.num_envs, device=device)
        env._failure_aware_low_mu_start = torch.zeros(env.num_envs, device=device)
        env._failure_aware_low_mu_length = torch.zeros(env.num_envs, device=device)
        env._failure_aware_no_progress_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    for row, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        origin = origins[env_id]
        yaw = torch.tensor(float(rng.uniform(*heading_range)), device=device)
        quat = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=device),
            torch.tensor([0.0], device=device),
            yaw.unsqueeze(0),
        )[0]
        x0 = float(rng.uniform(*start_x_range))
        y0 = float(rng.uniform(*start_y_range))
        root_pose[row, :3] = torch.tensor([x0, y0, float(root_z)], device=device) + origin
        root_pose[row, 3:7] = quat

        curb_x = float(rng.uniform(*curb_x_range))
        curb_h = float(rng.uniform(*curb_height_range))
        low_mu_x = float(rng.uniform(*low_mu_x_range))
        low_mu_len = float(rng.uniform(*low_mu_length_range))
        env._failure_aware_curb_height[env_id] = curb_h
        env._failure_aware_low_mu_start[env_id] = low_mu_x
        env._failure_aware_low_mu_length[env_id] = low_mu_len
        env._failure_aware_no_progress_steps[env_id] = 0

        if num_objects >= 1:
            object_states[row, 0, 0:3] = torch.tensor([curb_x, 0.0, curb_h * 0.5], device=device) + origin
            object_states[row, 0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        if num_objects >= 2:
            object_states[row, 1, 0:3] = torch.tensor([low_mu_x + 0.5 * low_mu_len, 0.0, 0.001], device=device) + origin
            object_states[row, 1, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    obstacles.write_object_state_to_sim(
        object_states,
        env_ids=env_ids,
        object_ids=torch.arange(num_objects, device=device, dtype=torch.long),
    )
