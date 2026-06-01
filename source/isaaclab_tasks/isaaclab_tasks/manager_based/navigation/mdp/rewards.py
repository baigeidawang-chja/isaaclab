# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F
from isaaclab.utils.math import euler_xyz_from_quat, normalize, quat_apply, wrap_to_pi
import isaaclab.envs.mdp as mdp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import SceneEntityCfg

from . import observation, terminations_user

def position_command_error_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    distance = torch.norm(des_pos_b, dim=1)
    return 1 - torch.tanh(distance / std)


def heading_command_error_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_command(command_name)
    heading_b = command[:, 3]
    return heading_b.abs()


## 下面为自己补充的函数

def distance_reward(env, max_distance: float = 15.0, scale: float = 2.0):
    """
    距离奖励：基于到目标点的距离给予连续奖励
    使用指数衰减函数，越接近目标奖励越大
    
    Args:
        max_distance: 最大有效距离，超过此距离奖励为0
        scale: 奖励缩放系数
    """
    cmd = env.command_manager.get_term("sequential_waypoints")
    robot = env.scene["robot"]
    
    # 计算到目标点的距离
    target_pos_w = cmd.pos_command_w
    robot_pos_w = robot.data.root_pos_w[:, :3]
    distance = torch.norm(target_pos_w - robot_pos_w, dim=1)
    
    # 使用指数衰减函数：exp(-distance / scale)
    # 距离越小，奖励越大
    normalized_distance = distance / max_distance
    reward = torch.exp(-normalized_distance * scale) * scale
    
    # 如果距离超过最大值，奖励为0
    reward = torch.where(distance > max_distance, torch.zeros_like(reward), reward)
    
    return reward


def progress_reward(
    env,
    scale: float = 10.0,
    max_reward: float = 2.0,
    min_reward: float = -2.0,
) -> torch.Tensor:
    """Reward only *progress* toward current waypoint: (prev_dist - dist).

    - Positive when moving closer
    - ~0 when staying near the target (prevents farming near goal)
    - Negative when moving away
    """
    cmd = env.command_manager.get_term("sequential_waypoints")
    robot = env.scene["robot"]

    target_pos_w = cmd.pos_command_w
    robot_pos_w = robot.data.root_pos_w[:, :3]
    dist = torch.norm(target_pos_w - robot_pos_w, dim=1)

    if not hasattr(env, "_prev_wp_dist"):
        env._prev_wp_dist = dist.detach().clone()
        return torch.zeros(env.num_envs, device=env.device)

    progress = env._prev_wp_dist - dist
    env._prev_wp_dist[:] = dist.detach()

    # scale and clamp for stability
    rew = torch.clamp(progress * scale, min_reward, max_reward)
    return rew

def time_penalty(env, penalty_per_step=-0.1):
    """
    每步固定惩罚，鼓励快速完成任务
    """
    return torch.full((env.num_envs,), penalty_per_step, device=env.device)


def heading_alignment_reward(env):
    """朝向对齐奖励 + 可视化（适配math.py的函数接口）"""

    cmd = env.command_manager.get_term("sequential_waypoints")
    robot = env.scene["robot"]

    # root_quat: (N,4) wxyz
    root_quat = robot.data.root_quat_w
    root_pos  = robot.data.root_pos_w[:, :3]

    # 1. 定义机器人自身 forward（你已确认是 +X）
    forward_local = torch.tensor([1.0, 0.0, 0.0], device=root_pos.device)
    forward_local = forward_local.expand(env.num_envs, 3)

    # 2. 旋转到世界坐标系
    robot_heading = quat_apply(root_quat, forward_local)

    # 3. 只考虑平面
    robot_heading[:, 2] = 0.0
    robot_heading = normalize(robot_heading)

    # 4. 目标方向（你原来的写法是对的）
    target_dir = cmd.pos_command_w - root_pos
    target_dir[:, 2] = 0.0
    target_dir = normalize(target_dir)

    # print("robot_heading:", robot_heading[0])
    # print("target_dir  :", target_dir[0])

    # 5. 朝向奖励
    heading_sim = torch.sum(
        robot_heading[:, :2] * target_dir[:, :2],
        dim=1
    )
    heading_sim = torch.clamp(heading_sim, 0.0, 1.0)

    return heading_sim


def straight_line_reward(
    env,
    max_distance: float = 5.0,      # 最大有效距离（超过则无奖励）
    reward_scale: float = 1.0,      # 奖励缩放系数
) -> torch.Tensor:
    """
    直线行驶奖励：机器人前进方向与目标方向越一致，奖励越高
    """

    cmd = env.command_manager.get_term("sequential_waypoints")
    robot = env.scene["robot"]

    root_quat = robot.data.root_quat_w
    root_pos = robot.data.root_pos_w[:, :3]
    forward_local = torch.tensor([1.0, 0.0, 0.0], device=root_pos.device)
    forward_local = forward_local.expand(env.num_envs, 3)
    robot_heading = quat_apply(root_quat, forward_local)

    robot_heading[:, 2] = 0.0
    robot_heading = normalize(robot_heading)

    target_dir = cmd.pos_command_w - root_pos

    # 1. 计算机器人→目标点的方向向量（XY平面，忽略Z轴）
    target_dir = target_dir[:, :2] 
    # 归一化目标方向向量（避免距离影响相似度计算）
    target_dir = normalize(target_dir, eps=1e-6)  # (num_envs, 2)

    # 2. 提取机器人前进方向的XY分量（与目标方向维度匹配）
    robot_heading_xy = robot_heading[:, :2]  # (num_envs, 2)

    # 3. 计算两个向量的余弦相似度（范围[-1, 1] → 修正为[0, 1]）
    # 相似度=1 → 完全同向；相似度=0 → 垂直；相似度=-1 → 完全反向
    similarity = torch.sum(robot_heading_xy * target_dir, dim=1)  # (num_envs,)
    similarity = torch.clamp(similarity, 0.0, 1.0)  # 反向无奖励，只奖励同向

    # 4. 距离衰减系数：离目标越近，奖励越高（鼓励最后阶段直线冲刺）
    distance = torch.norm(target_dir[:, :2] - root_pos[:, :2], dim=1)  # (num_envs,)
    distance_coeff = 1.0 - torch.clamp(distance / max_distance, 0.0, 1.0)  # (num_envs,)

    # 5. 最终直线奖励 = 相似度 × 距离系数 × 缩放因子
    reward = similarity * distance_coeff * reward_scale

    return reward

def pure_straight_reward(
    env,
    yaw_stable_thresh: float = 0.05, # 航向波动阈值（弧度，越小越稳定）
    reward_scale: float = 1.0,      # 奖励缩放系数
) -> torch.Tensor:
    """
    纯直线行驶奖励：不依赖目标点，仅奖励方向稳定、位移与前进方向一致的行为
    """

    robot = env.scene["robot"]
    cmd = env.command_manager.get_term("sequential_waypoints")

    robot_pos = robot.data.root_pos_w
    robot_quat = robot.data.root_quat_w
    robot_roll, robot_pitch, robot_yaw = euler_xyz_from_quat(robot_quat)

    forward_local = torch.tensor([1.0, 0.0, 0.0], device=robot_pos.device)
    forward_local = forward_local.expand(env.num_envs, 3)
    robot_heading = quat_apply(robot_quat, forward_local)

    robot_heading[:, 2] = 0.0
    robot_heading = normalize(robot_heading)

    # 初始化历史数据（记录上一步的位置和yaw）
    if not hasattr(env, "_prev_robot_pos"):
        env._prev_robot_pos = robot_pos.clone()  # 上一步位置
        env._prev_robot_yaw = robot_yaw.clone()  # 上一步yaw角
        return torch.zeros(env.num_envs, device=env.device)  # 第一步无位移，奖励为0

    # ========== 特征1：位移方向与前进方向一致 ==========
    # 1. 计算当前步的位移向量（XY平面）
    displacement = robot_pos[:, :2] - env._prev_robot_pos[:, :2]  # (num_envs, 2)
    # 归一化位移向量（避免速度影响方向判断）
    displacement_dir = normalize(displacement, eps=1e-6)  # (num_envs, 2)
    # 2. 提取前进方向的XY分量
    heading_dir = robot_heading[:, :2]  # (num_envs, 2)
    # 3. 计算两个方向的余弦相似度（范围[-1,1] → 修正为[0,1]）
    dir_similarity = torch.sum(displacement_dir * heading_dir, dim=1)
    dir_similarity = torch.clamp(dir_similarity, 0.0, 1.0)  # 反向无奖励

    # ========== 特征2：航向（yaw）稳定，无大幅转弯 ==========
    # 1. 计算yaw角变化量（归一化到[-π, π]）
    yaw_delta = wrap_to_pi(robot_yaw - env._prev_robot_yaw)  # (num_envs,)
    # 2. 计算航向稳定系数：变化越小，系数越接近1
    yaw_stability = 1.0 - torch.clamp(torch.abs(yaw_delta) / yaw_stable_thresh, 0.0, 1.0)

    # ========== 最终奖励：两个特征的乘积 ==========
    # 只有同时满足「位移与前进同向」+「航向稳定」，才会获得高奖励
    reward = dir_similarity * yaw_stability * reward_scale

    # ========== 更新历史数据 ==========
    env._prev_robot_pos[:] = robot_pos
    env._prev_robot_yaw[:] = robot_yaw

    return reward

def in_place_oscillation_penalty(
        env,
        lin_speed_threshold: float = 0.15,
        yaw_rate_threshold: float = 0.25,
        lin_acc_scale: float = 10.0,
        yaw_acc_scale: float = 5,
        max_penalty: float = 10.0,
) -> torch.Tensor:
    robot = env.scene["robot"]

    lin_vel_xy = robot.data.root_lin_vel_b[:, :2]
    lin_speed = torch.norm(lin_vel_xy, dim=1)
    yaw_rate = torch.abs(robot.data.root_ang_vel_b[:, 2])

    dt = getattr(env, "physics_dt", None)
    if dt is None:
        dt = getattr(env, "step_dt", 1.0 / 60)
    dt = float(dt)

    if not hasattr(env, "_osc_prev_line_vel_xy"):
        env._osc_prev_line_vel_xy = lin_vel_xy.detach().clone()
        env._osc_prev_yaw_rate = robot.data.root_ang_vel_b[:, 2].detach().clone()

    prev_lin_vel_xy = env._osc_prev_line_vel_xy
    prev_yaw_rate_raw = env._osc_prev_yaw_rate

    lin_acc = torch.norm((lin_vel_xy - prev_lin_vel_xy) / max(dt, 1e-6), dim=1)
    yaw_acc = torch.abs((robot.data.root_ang_vel_b[:, 2] - prev_yaw_rate_raw) / max(dt, 1e-6))

    env._osc_prev_line_vel_xy = lin_vel_xy.detach().clone()
    env._osc_prev_yaw_rate = robot.data.root_ang_vel_b[:, 2].detach().clone()

    gate = (lin_speed < lin_speed_threshold) & (yaw_rate < yaw_rate_threshold)

    osc_strength = lin_acc_scale * lin_acc + yaw_acc_scale * yaw_acc
    osc_strength = torch.clamp(osc_strength, min=0.0, max=max_penalty)

    penalty = torch.where(gate, -osc_strength, torch.zeros_like(osc_strength))
    return penalty

def low_speed_penalty(
    env,
    speed_threshold: float = 0.2,
    scale: float = 1.0,
) -> torch.Tensor:
    """Penalize near-zero speed (sign doesn't matter).

    Args:
        speed_threshold: penalty active when |v| < speed_threshold (m/s)
        scale: penalty magnitude at v=0; linearly goes to 0 at threshold
    Returns:
        (num_envs,) penalty values (<= 0)
    """
    robot = env.scene["robot"]
    # base linear velocity in world frame (N,3)
    v = robot.data.root_lin_vel_w[:, :2]
    speed = torch.norm(v, dim=-1)  # (N,)

    # linear ramp:  v=0 -> -scale, v>=threshold -> 0
    frac = (speed_threshold - speed) / max(speed_threshold, 1e-6)
    frac = torch.clamp(frac, min=0.0, max=1.0)
    return -scale * frac

def waypoint_reached_reward(env, command_name: str = "sequential_waypoints", scale: float = 1.0) -> torch.Tensor:
    """One-shot reward when a waypoint is newly reached in the current step.

    This relies on SequentialWaypointCommand.newly_reached_waypoint being set BEFORE the command advances.
    Returns (num_envs,) tensor.
    """
    cmd = env.command_manager.get_term(command_name)
    # bool -> float32
    return scale * cmd.newly_reached_waypoint.float()

def _front_blocked_from_grid(
        grid_map: torch.Tensor,
        num_cells: int,
        front_cols: int = 3,
        center_rows: int = 4,        
) -> torch.Tensor:
    N, K = grid_map.shape
    assert K == num_cells * num_cells

    grid = grid_map.view(N, num_cells, num_cells) 
    y0 = num_cells // 2
    half = center_rows // 2
    y_min = max(0, y0 - half)
    y_max = min(num_cells, y0 + half + (center_rows % 2))

    x_min = num_cells - front_cols
    x_max = num_cells

    patch = grid[:, y_min:y_max, x_min:x_max]
    blocked = patch.amax(dim=(1, 2)) > 0.5
    return blocked

def stuck_when_blocked_penalty(
    env,
    grid_size: float = 6.0,
    num_cells: int = 10,
    obstacle_size: float = 1.0,
    speed_threshold: float = 0.3,
    front_cols: int = 3,
    center_rows: int = 4,
    scale: float = 1.0,      
) -> torch.Tensor:
    grid = observation.get_obstacle_grid_map_scene(
        env,
        grid_size=grid_size,
        num_cells=num_cells,
        obstacle_size=obstacle_size,
        debug_vis=False,
    )

    blocked = _front_blocked_from_grid(grid, num_cells=num_cells, front_cols=front_cols, center_rows=center_rows)

    robot = env.scene["robot"]
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=-1)
    stuck = speed < speed_threshold

    return -scale * (blocked & stuck).float()

def push_into_block_penalty(
    env,
    grid_size: float = 6.0,
    num_cells: int = 10,
    obstacle_size: float = 1.0,
    front_cols: int = 3,
    center_rows: int = 4,
    action_term_name: str = "throttle_steer",
    throttle_action_index: tuple[int, int, int, int] = (0, 1, 2, 3),
    scale: float = 1.0,
): 
    grid = observation.get_obstacle_grid_map_scene(
        env,
        grid_size=grid_size,
        num_cells=num_cells,
        obstacle_size=obstacle_size,
        debug_vis=False,
    )
    blocked = _front_blocked_from_grid(grid, num_cells=num_cells, front_cols=front_cols, center_rows=center_rows)

    a = mdp.last_action(env)
    th = a[:, list(throttle_action_index)].mean(dim=1)
    pushing = th > 0.1

    return -scale * (blocked & pushing).float()

def unblocked_bonus(
    env,
    grid_size: float = 6.0,
    num_cells: int = 10,
    obstacle_size: float = 1.0,
    front_cols: int = 3,
    center_rows: int = 4,
    scale: float = 1.0,
) -> torch.Tensor:
    grid = observation.get_obstacle_grid_map_scene(
        env,
        grid_size=grid_size,
        num_cells=num_cells,
        obstacle_size=obstacle_size,
        debug_vis=False,
    )
    blocked = _front_blocked_from_grid(grid, num_cells=num_cells, front_cols=front_cols, center_rows=center_rows)

    if not hasattr(env, "_prev_blocked"):
        env._prev_blocked = blocked.clone()
        return torch.zeros((env.num_envs,), device=env.device)

    prev = env._prev_blocked
    env._prev_blocked = blocked.clone()

    return scale * ((prev == 1) & (blocked == 0)).float()    

def velocity_along_world_dir_reward(env, dir_x: float = 1.0, dir_y: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """Reward forward progress along a fixed world direction. Returns (N,)."""
    robot = env.scene["robot"]
    v = robot.data.root_lin_vel_w[:, :2]  # (N,2)
    d = torch.tensor([dir_x, dir_y], device=env.device, dtype=torch.float32)
    d = d / (torch.norm(d) + 1e-6)
    return scale * (v @ d)  # dot => speed along dir (can be negative if moving opposite)

def heading_align_world_dir_reward(env, dir_x: float = 1.0, dir_y: float = 0.0, scale: float = 1.0) -> torch.Tensor:
    """Reward heading alignment with a fixed world direction. Returns (N,)."""
    robot = env.scene["robot"]
    yaw = robot.data.heading_w
    f = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)  # (N,2)
    d = torch.tensor([dir_x, dir_y], device=env.device, dtype=torch.float32)
    d = d / (torch.norm(d) + 1e-6)
    return scale * (f @ d)  # in [-1, 1]

def heading_align_waypoint_reward(env, command_name: str = "sequential_waypoints", scale: float = 1.0) -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name)
    robot = env.scene["robot"]

    robot_xy = robot.data.root_pos_w[:, :2]
    wp_xy = cmd.pos_command_w[:, :2]

    vec = wp_xy - robot_xy
    desired_yaw = torch.atan2(vec[:, 1], vec[:, 0])

    yaw = robot.data.heading_w
    err = wrap_to_pi(desired_yaw - yaw).abs()
    reward = scale * torch.cos(err)
    return reward

def track_forward_speed_reward(
    env,
    target_speed: float = 2.0,
    std: float = 0.5,
) -> torch.Tensor:
    """Reward tracking a desired forward speed in the robot base frame.

    Uses base x-velocity (forward). Output shape: (num_envs,).
    - target_speed: desired m/s
    - std: tolerance; smaller => sharper peak around target
    """
    v_b = mdp.base_lin_vel(env)[:, 0]  # (N,) forward speed in base frame
    err = v_b - target_speed
    # Gaussian-shaped reward in [0, 1]
    return torch.exp(-0.5 * (err / std) ** 2)

def obstacle_proximity_repulsion_penalty(
    env,
    grid_size: float = 8.0,
    num_cells: int = 10,
    obstacle_size: float = 1.0,
    falloff: float = 1.0,
    max_range: float | None = None,
    front_only: bool = False,
    front_x_min: float = 0.0,
    scale: float = 1.0,
) -> torch.Tensor:
    """Repulsion penalty for being close to occupied grid cells (red cells).

    Penalty = -scale * mean_{occupied cells} exp(-dist / falloff)
    - falloff: larger => penalty spreads farther
    - max_range: if set, ignore occupied cells farther than this distance (in robot frame)
    - front_only: if True, only consider cells with x >= front_x_min (robot forward)
    Returns: (num_envs,)
    """
    occ = observation.get_obstacle_grid_map_scene(
        env,
        grid_size=grid_size,
        num_cells=num_cells,
        obstacle_size=obstacle_size,
        debug_vis=False,
    )  # (N, K), values in {0,1}

    device = env.device
    N, K = occ.shape
    if K != num_cells * num_cells:
        raise ValueError(f"occ has K={K}, expected {num_cells*num_cells}")

    # build (K,2) local cell centers (robot frame), consistent with visualization:
    # x: -half .. +half, y: +half .. -half
    cell_size = grid_size / num_cells
    half = grid_size / 2.0

    x_idx = torch.arange(num_cells, device=device)
    y_idx = torch.arange(num_cells, device=device)
    xx, yy = torch.meshgrid(x_idx, y_idx, indexing="xy")  # (num_cells, num_cells)

    cell_x = (xx + 0.5) * cell_size - half                    # (num_cells, num_cells)
    cell_y = half - (yy + 0.5) * cell_size                    # (num_cells, num_cells)
    centers = torch.stack([cell_x, cell_y], dim=-1).view(-1, 2)  # (K,2)

    # optional masks (K,)
    mask = torch.ones((K,), device=device, dtype=torch.bool)
    if front_only:
        mask &= centers[:, 0] >= front_x_min
    if max_range is not None:
        dist0 = torch.norm(centers, dim=-1)
        mask &= dist0 <= max_range

    # dist from robot center to each cell center (same for all envs) -> (K,)
    dist = torch.norm(centers, dim=-1).clamp(min=1e-6)

    # potential per cell (K,)
    pot = torch.exp(-dist / max(falloff, 1e-6))

    # apply mask
    pot = pot * mask.float()  # (K,)

    # occupied weighting per env (N,K)
    occ_w = occ * pot.unsqueeze(0)  # (N,K)

    # average over occupied cells (avoid dividing by 0)
    occ_count = (occ * mask.float().unsqueeze(0)).sum(dim=1).clamp(min=1.0)  # (N,)
    penalty = occ_w.sum(dim=1) / occ_count  # (N,)

    return -scale * penalty

def flipped_penalty(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), up_z_threshold: float = 0.2, penalty: float = -100.0) -> torch.Tensor:
    """Large penalty if flipped. Returns penalty on flipped envs, else 0."""
    flipped = terminations_user.is_flipped(env, asset_cfg=asset_cfg, up_z_threshold=up_z_threshold)
    return torch.where(flipped, torch.full_like(flipped, float(penalty), dtype=torch.float32), torch.zeros_like(flipped, dtype=torch.float32))