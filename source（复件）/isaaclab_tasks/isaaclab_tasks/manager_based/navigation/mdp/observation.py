from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.envs import ManagerBasedEnv
from isaaclab.utils import math as math_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils

def get_target_rel_pos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """获取目标相对位置 (世界坐标系)"""
    cmd = env.command_manager.get_term("sequential_waypoints")
    result = cmd.pos_command_b
    # print(f"[DEBUG] target_rel_pos shape: {result.shape}")
    # print(f"[DEBUG] target_rel_pos sample: {result[0]}")
    return result

def get_target_heading(env: ManagerBasedRLEnv) -> torch.Tensor:
    """获取目标航向角 (世界坐标系)"""
    cmd = env.command_manager.get_term("sequential_waypoints")
    robot = env.scene["robot"]

    # root_quat: (N,4) wxyz
    root_quat = robot.data.root_quat_w
    root_pos  = robot.data.root_pos_w[:, :3]

    # 1. 定义机器人自身 forward（你已确认是 +X）
    forward_local = torch.tensor([1.0, 0.0, 0.0], device=root_pos.device)
    forward_local = forward_local.expand(env.num_envs, 3)

    # 4. 目标方向（你原来的写法是对的）
    target_dir = cmd.pos_command_w - root_pos
    target_dir[:, 2] = 0.0
    target_dir = math_utils.normalize(target_dir)

    return target_dir


def get_base_heading_vec(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回机器人当前朝向在世界坐标系下的 [cos(yaw), sin(yaw)]。"""
    robot = env.scene["robot"]
    heading = robot.data.heading_w
    return torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)

def get_nearest_obstacle_info(env):
    """
    获取最近障碍物的信息（位置、距离等）
    """
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[:, :2]
    
    if "obstacles" not in env.scene.rigid_object_collections:
        # 如果没有障碍物，返回零向量
        return torch.zeros(env.num_envs, 3, device=env.device)
    
    obstacles = env.scene.rigid_object_collections["obstacles"]
    min_dist = torch.full((env.num_envs,), float('inf'), device=env.device)
    nearest_pos = torch.zeros(env.num_envs, 2, device=env.device)
    
    # 遍历所有障碍物，找到最近的
    for obstacle_name, obstacle_cfg in obstacles.cfg.rigid_objects.items():
        # 获取障碍物位置（静态障碍物使用初始位置）
        obstacle_pos = torch.tensor(
            obstacle_cfg.init_state.pos[:2], 
            device=env.device
        ).unsqueeze(0).repeat(env.num_envs, 1)
        
        # 计算距离
        dist = torch.norm(robot_pos - obstacle_pos, dim=1)
        
        # 更新最近障碍物
        is_closer = dist < min_dist
        min_dist = torch.where(is_closer, dist, min_dist)
        nearest_pos = torch.where(is_closer.unsqueeze(1), obstacle_pos, nearest_pos)
    
    # 返回：最近障碍物的相对位置（x, y, distance）
    relative_pos = nearest_pos - robot_pos
    nearest_info = torch.cat([relative_pos, min_dist.unsqueeze(1)], dim=1)
    
    return nearest_info

def get_contact_sensor_feedback(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns contact force magnitude and contact flag from the robot's body sensor."""
    sensor = env.scene.sensors.get("robot_contact_sensor", None)
    if sensor is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    forces = sensor.data.net_forces_w
    if forces is None or forces.numel() == 0:
        return torch.zeros((env.num_envs, 2), device=env.device)

    # forces shape: (N, B, 3)
    force_magnitude = torch.norm(forces, dim=-1)
    max_force, _ = torch.max(force_magnitude, dim=1)
    contact_flag = (max_force > 1e-3).float()
    return torch.stack([max_force, contact_flag], dim=1)


def get_actuator_load(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Returns applied joint torques for all actuated joints."""
    robot = env.scene["robot"]
    torques = robot.data.applied_torque
    if torques is None:
        return torch.zeros((env.num_envs, 0), device=env.device)
    return torques

def get_imu_data(env: ManagerBasedRLEnv):
    imu = env.scene.sensors.get("IMU_baselink", None)
    data = imu.data
    return torch.cat(
        [data.lin_acc_b, data.ang_vel_b, data.quat_w],
        dim = -1
    )

STATIC_OBSTACLES_XY = torch.tensor([
    # 示例：你用真实数据替换
    [ 0.0,  -6.2473],
    [ 38.0, -6.2473],
    [ 5.0,  -3.0],
    [ 7.0,  12.7527],
    [ 8.0, -5.2473],
    [ 11.0,  -1.7527],
    [ 16.0, 10.7527],
    [ 15.0,  -6.2473],
    [ 17.0,  6.7527],
    [ 20.0, -5.2473],
    [ 12.0,  -11.2473],
    [ 21.0, -12.2473],
    [ 22.0,  15.7527],
    [ 22.0,  7.7527],
    [ 28.0, 10.51739],
    [ 4.0,  8.0],
    [ 27.0, -14.2473],
    [ 29.0,  12.7527],
    [ 27.0,  -9.2473],
    [ 15.0, 14.7527],
    [ 44.04432,  1.16013],
    [ 32.0, -15.2473],
    [ 32.0,  15.7527],
    [ 32.0,  -5.2473],
    [ 38.0, 10.7527],
    [ 0.0,  18.7527],
    [ 5.0, -16.2473],
    [ 11.0,  19.7527],
    [ 39.0,  21.7527],
    [ 39.0, -17.2473],
    [ 26.75, 1.8],          
], dtype=torch.float32)

# STATIC_OBSTACLES_XY = torch.tensor([
#     # 示例：你用真实数据替换
#     [ 0.0,  -6.2473],
#     [ 38.0, -6.2473],
#     [ 5.0,  -3.0],
#     [ 7.0,  12.7527],
#     [ 8.0, -5.2473],
#     # [ 11.0,  1.7527],
#     [ 14.33853, -0.94186],
#     [ 16.0, 10.7527],
#     [ 15.0,  -6.2473],
#     [ 17.0,  6.7527],
#     [ 20.0, -5.2473],
#     [ 12.0,  -11.2473],
#     [ 21.0, -12.2473],
#     [ 22.0,  15.7527],
#     # [ 22.0,  7.7527],
#     [ 22.0, 5.68448],
#     [ 28.0, 10.51739],
#     [ 4.0,  8.0],
#     [ 27.0, -14.2473],
#     [ 29.0,  12.7527],
#     [ 27.0,  -9.2473],
#     [ 15.0, 14.7527],
#     [ 44.04432,  1.16013],
#     [ 32.0, -15.2473],
#     [ 32.0,  15.7527],
#     [ 32.0,  -5.2473],
#     # [ 38.0, 10.7527],
#     [ 41.86662, 8.20356],
#     [ 0.0,  18.7527],
#     [ 5.0, -16.2473],
#     [ 11.0,  19.7527],
#     [ 39.0,  21.7527],
#     [ 39.0, -17.2473],
#     # [ 26.75, 1.8],
#     [ 30.19788, 4.69027],          
# ], dtype=torch.float32)

def _get_env_origins_xy(env: ManagerBasedEnv) -> torch.Tensor:
    """Return per-env origin offsets (N,2). Falls back to zeros."""
    if hasattr(env, "scene") and hasattr(env.scene, "env_origins"):
        return env.scene.env_origins[:, :2]
    return torch.zeros((env.num_envs, 2), device=env.device)

def _get_scene_obstacles_xy(env: ManagerBasedEnv, collection_name: str = "obstacles") -> torch.Tensor:
    """Get obstacle centers in world frame.

    Returns:
      - (N,M,2) if per-env runtime positions are available (preferred)
      - (N,M,2) from init_state + env_origins as fallback
      - empty (0,2) if nothing found
    """
    device = env.device

    # Prefer: asset from env.scene[...] because it's the actual instantiated collection
    asset = None
    try:
        asset = env.scene[collection_name]
    except Exception:
        # fallback to registry if exists (may be cfg only)
        if hasattr(env.scene, "rigid_object_collections") and collection_name in env.scene.rigid_object_collections:
            asset = env.scene.rigid_object_collections[collection_name]
        else:
            return torch.empty((0, 2), device=device)

    # 1) Runtime path: read positions from asset.data (this reflects randomization)
    try:
        data = getattr(asset, "data", None)
        if data is not None:
            # try common field names across IsaacLab versions
            for key in ("root_pos_w", "object_pos_w", "object_positions_w"):
                if hasattr(data, key):
                    pos_w = getattr(data, key)
                    # expected shapes:
                    #   (N, M, 3)  or (M, 3) or (N,3) for single object (unlikely)
                    if isinstance(pos_w, torch.Tensor):
                        if pos_w.ndim == 3:
                            return pos_w[:, :, :2].contiguous()  # (N,M,2)
                        if pos_w.ndim == 2:
                            # assume (M,3) -> broadcast to (N,M,2) later by caller if needed
                            return pos_w[:, :2].contiguous()
    except Exception:
        pass

    # 2) Fallback: use cfg.init_state (static template) + env_origins => (N,M,2)
    try:
        origins_xy = _get_env_origins_xy(env)  # (N,2)

        # asset may be a cfg-like container; try to reach cfg.rigid_objects
        cfg = getattr(asset, "cfg", None)
        if cfg is None or not hasattr(cfg, "rigid_objects"):
            return torch.empty((0, 2), device=device)

        obs_xy_list = []
        for _, obj_cfg in cfg.rigid_objects.items():
            p_local = torch.tensor(obj_cfg.init_state.pos[:2], device=device, dtype=torch.float32)
            obs_xy_list.append(p_local)

        if len(obs_xy_list) == 0:
            return torch.empty((0, 2), device=device)

        obs_xy = torch.stack(obs_xy_list, dim=0)  # (M,2)
        obs_xy = obs_xy.unsqueeze(0) + origins_xy.unsqueeze(1)  # (N,M,2)
        return obs_xy
    except Exception:
        return torch.empty((0, 2), device=device)


def get_obstacle_grid_map_scene(
    env: ManagerBasedEnv,
    grid_size: float = 6.0,
    num_cells: int = 10,
    obstacle_size: float = 1.0,
    debug_vis: bool = False,
    vis_update_interval: int = 2,
    collection_name: str = "obstacles",
) -> torch.Tensor:
    device = env.device
    num_envs = env.num_envs

    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[:, :2]          # (N,2)
    heading = robot.data.heading_w                      # (N,)

    cell = grid_size / num_cells
    half = grid_size / 2.0

    # IMPORTANT: build grid in the same linear index order as visualization:
    # cell_idx = y_idx * num_cells + x_idx   (y-major)
    xs = torch.linspace(cell / 2 - half, half - cell / 2, num_cells, device=device)  # x forward
    ys = torch.linspace(half - cell / 2, cell / 2 - half, num_cells, device=device)  # y left (top->bottom)

    # (K,2) with K=num_cells*num_cells in y-major order
    grid_local_xy = torch.stack(
        [xs.repeat(num_cells), ys.repeat_interleave(num_cells)],
        dim=1,
    )
    K = grid_local_xy.shape[0]

    # yaw-only rotation
    cos_h = torch.cos(heading).unsqueeze(1)  # (N,1)
    sin_h = torch.sin(heading).unsqueeze(1)  # (N,1)

    x = grid_local_xy[:, 0].view(1, K).expand(num_envs, K)
    y = grid_local_xy[:, 1].view(1, K).expand(num_envs, K)

    grid_world_x = cos_h * x - sin_h * y
    grid_world_y = sin_h * x + cos_h * y
    grid_world_xy = torch.stack([grid_world_x, grid_world_y], dim=-1) + robot_pos_w.unsqueeze(1)  # (N,K,2)

    # obstacles
    obs_xy = _get_scene_obstacles_xy(env, collection_name=collection_name)

    if obs_xy.numel() == 0:
        occ = torch.zeros((num_envs, K), device=device, dtype=torch.float32)
    else:
        if obs_xy.ndim == 2:
            obs_xy = obs_xy.unsqueeze(0).expand(num_envs, -1, -1)  # (N,M,2)
        elif obs_xy.ndim != 3:
            raise RuntimeError(f"Unexpected obs_xy shape: {tuple(obs_xy.shape)}")

        # diff = grid_world_xy.unsqueeze(1) - obs_xy.unsqueeze(2)  # (N,M,K,2)
        # half_size = obstacle_size / 2.0
        # inside = (diff[..., 0].abs() <= half_size) & (diff[..., 1].abs() <= half_size)  # (N,M,K)
        # occ = torch.any(inside, dim=1).float()  # (N,K)
        diff = grid_world_xy.unsqueeze(1) - obs_xy.unsqueeze(2)  # (N,M,K,2)

        # AABB overlap between:
        # - cell box: half extent = cell/2
        # - obstacle box: half extent = obstacle_size/2 (+ optional inflate)
        cell_half = 0.5 * cell
        obs_half = 0.5 * obstacle_size

        # overlap if |dx| <= (cell_half + obs_half) and |dy| <= (cell_half + obs_half)
        inside = (diff[..., 0].abs() <= (cell_half + obs_half)) & (diff[..., 1].abs() <= (cell_half + obs_half))
        occ = torch.any(inside, dim=1).float()

    # visualization (unchanged)
    if debug_vis:
        if not hasattr(env, "_grid_vis_frame_count"):
            env._grid_vis_frame_count = 0
        env._grid_vis_frame_count += 1
        if vis_update_interval <= 0:
            vis_update_interval = 1
        if env._grid_vis_frame_count % vis_update_interval == 0:
            _visualize_grid_map(
                env=env,
                grid_map=occ,
                robot_pos=robot.data.root_pos_w[:, :2],
                robot_heading=robot.data.heading_w,
                robot_pos_z=robot.data.root_pos_w[:, 2],
                grid_size=grid_size,
                num_cells=num_cells,
            )

    return occ

def _visualize_grid_map(
    env: ManagerBasedEnv,
    grid_map: torch.Tensor,
    robot_pos: torch.Tensor,
    robot_heading: torch.Tensor,
    robot_pos_z: torch.Tensor,
    grid_size: float,
    num_cells: int,
):
    """可视化栅格地图：只处理 env0，同时显示 occupied(红) 和 free(绿)。"""
    if not hasattr(env, "_grid_map_visualizer"):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ObstacleGridMap",
            markers={
                "occupied": sim_utils.CuboidCfg(
                    size=(grid_size / num_cells * 0.9, grid_size / num_cells * 0.9, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.0),
                        metallic=0.0,
                        roughness=0.5,
                    ),
                ),
                "free": sim_utils.CuboidCfg(
                    size=(grid_size / num_cells * 0.9, grid_size / num_cells * 0.9, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 1.0, 0.0),
                        metallic=0.0,
                        roughness=0.5,
                    ),
                ),
            },
        )
        env._grid_map_visualizer = VisualizationMarkers(marker_cfg)

    device = env.device
    cell_size = grid_size / num_cells
    half_grid = grid_size / 2.0

    env_id = 0
    pos = robot_pos[env_id]  # (2,)
    heading = robot_heading[env_id]
    z_pos = robot_pos_z[env_id].item()

    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    rot_matrix = torch.tensor([[cos_h.item(), -sin_h.item()], [sin_h.item(), cos_h.item()]], device=device)

    orientation = math_utils.quat_from_euler_xyz(
        torch.tensor([0.0], device=device),
        torch.tensor([0.0], device=device),
        heading.unsqueeze(0),
    )[0]  # (4,)

    all_positions = []
    all_orientations = []
    all_indices = []

    for y_idx in range(num_cells):
        for x_idx in range(num_cells):
            cell_idx = y_idx * num_cells + x_idx
            is_occupied = grid_map[env_id, cell_idx] > 0.5

            cell_center_x_robot = (x_idx + 0.5) * cell_size - half_grid
            cell_center_y_robot = half_grid - (y_idx + 0.5) * cell_size

            cell_center_robot = torch.tensor([cell_center_x_robot, cell_center_y_robot], device=device)
            cell_center_world = rot_matrix @ cell_center_robot + pos

            world_pos = torch.tensor(
                [cell_center_world[0].item(), cell_center_world[1].item(), z_pos + 0.1],
                device=device,
            )

            all_positions.append(world_pos)
            all_orientations.append(orientation)
            all_indices.append(0 if is_occupied else 1)  # 0=occupied, 1=free

    # visualize (avoid empty InstanceIndices warning by always publishing full grid)
    positions_tensor = torch.stack(all_positions)  # (num_cells*num_cells, 3)
    orientations_tensor = torch.stack(all_orientations)  # (num_cells*num_cells, 4)
    indices_tensor = torch.tensor(all_indices, device=device, dtype=torch.int32)

    env._grid_map_visualizer.visualize(
        translations=positions_tensor,
        orientations=orientations_tensor,
        marker_indices=indices_tensor,
    )

def desired_world_dir(
    env: ManagerBasedRLEnv,
    x: float = 1.0,
    y: float = 0.0,
):
    d = torch.tensor([x, y], device=env.device, dtype=torch.float32)
    d = d / (torch.norm(d) + 1e-6)
    return d.unsqueeze(0).repeat(env.num_envs, 1)

def base_forward_dir_w(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot = env.scene[asset_cfg.name]
    yaw = robot.data.heading_w
    f = torch.stack([torch.cos(yaw), torch.sin(yaw)], dim=-1)
    return f