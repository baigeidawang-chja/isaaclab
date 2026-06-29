import torch
import numpy as np

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCollection
from isaaclab.utils import math as math_utils
import isaaclab.sim as sim_utils

def generate_random_grid_obstacle_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    grid_rows: int = 5,
    grid_cols: int = 5,
    base_spacing: float = 3.0,
    spacing_variation: float = 1.5,
    x_range: tuple[float, float] = (0.0, 50.0),
    y_range: tuple[float, float] = (-10.0, 10.0),
    z_height: float = 0.5,
    # 新增参数：机器人初始位置范围（用于避开初始位置）
    robot_init_x_range: tuple[float, float] = (-1.5, 1.5),  # 机器人初始X位置范围
    robot_init_y_range: tuple[float, float] = (-1.5, 1.5),  # 机器人初始Y位置范围
    min_distance_from_robot: float = 3.0,  # 障碍物距离机器人初始位置的最小距离（米）
    min_distance_between_obstacles: float = 2.0,  # 障碍物之间的最小距离
    use_random_distribution: bool = True,
    max_sample_tries: int = 1000,
):
    """在网格布局基础上生成随机间距的障碍物位置。
    
    可以避开机器人初始位置，但允许障碍物生成在轨迹的其他部分。
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device)

    asset: RigidObjectCollection = env.scene[asset_cfg.name]
    num_obstacles = asset.num_objects
    
    for env_id in env_ids:
        if isinstance(env_id, torch.Tensor):
            env_id = env_id.item()
        
        positions = []
        x_start, x_end = x_range
        y_start, y_end = y_range
        
        # 计算机器人初始位置的中心和范围（用于排除）
        robot_center_x = (robot_init_x_range[0] + robot_init_x_range[1]) / 2
        robot_center_y = (robot_init_y_range[0] + robot_init_y_range[1]) / 2
        
        if use_random_distribution:
            # 完全随机分布
            for obj_idx in range(num_obstacles):
                valid_position = False
                attempts = 0
                
                while not valid_position and attempts < max_sample_tries:
                    # 随机生成位置
                    x = np.random.uniform(x_start, x_end)
                    y = np.random.uniform(y_start, y_end)
                    
                    # 检查是否距离机器人初始位置足够远（如果min_distance_from_robot > 0）
                    if min_distance_from_robot > 0:
                        dist_from_robot = np.sqrt(
                            (x - robot_center_x)**2 + (y - robot_center_y)**2
                        )
                        
                        if dist_from_robot < min_distance_from_robot:
                            attempts += 1
                            continue
                    
                    # 检查是否与其他已生成的障碍物距离足够远
                    too_close = False
                    for existing_pos in positions:
                        dist = np.sqrt(
                            (x - existing_pos[0])**2 + (y - existing_pos[1])**2
                        )
                        if dist < min_distance_between_obstacles:
                            too_close = True
                            break
                    
                    if not too_close:
                        valid_position = True
                        positions.append([x, y, z_height])
                    else:
                        attempts += 1
                
                # 如果尝试次数过多仍未找到有效位置，使用一个远离机器人的位置
                if not valid_position:
                    # 在远离机器人的区域随机生成
                    if min_distance_from_robot > 0:
                        if robot_center_x < 0:
                            safe_x = np.random.uniform(max(x_start, robot_center_x + min_distance_from_robot), x_end)
                        else:
                            safe_x = np.random.uniform(x_start, min(x_end, robot_center_x - min_distance_from_robot))
                        
                        if robot_center_y < 0:
                            safe_y = np.random.uniform(max(y_start, robot_center_y + min_distance_from_robot), y_end)
                        else:
                            safe_y = np.random.uniform(y_start, min(y_end, robot_center_y - min_distance_from_robot))
                    else:
                        safe_x = np.random.uniform(x_start, x_end)
                        safe_y = np.random.uniform(y_start, y_end)
                    
                    positions.append([safe_x, safe_y, z_height])
        else:
            # 基于网格的分布
            actual_num = min(num_obstacles, grid_rows * grid_cols)
            
            grid_indices = []
            for i in range(grid_rows):
                for j in range(grid_cols):
                    if len(grid_indices) < actual_num:
                        grid_indices.append((i, j))
            
            np.random.shuffle(grid_indices)
            
            for idx, (row, col) in enumerate(grid_indices[:actual_num]):
                valid_position = False
                attempts = 0
                
                while not valid_position and attempts < max_sample_tries:
                    base_x = x_start + (x_end - x_start) * (row / max(1, grid_rows - 1))
                    base_y = y_start + (y_end - y_start) * (col / max(1, grid_cols - 1))
                    
                    spacing_x = base_spacing + np.random.uniform(-spacing_variation, spacing_variation)
                    spacing_y = base_spacing + np.random.uniform(-spacing_variation, spacing_variation)
                    
                    offset_x = np.random.uniform(-1.0, 1.0)
                    offset_y = np.random.uniform(-1.0, 1.0)
                    
                    x = base_x + offset_x
                    y = base_y + offset_y
                    
                    x = np.clip(x, x_start, x_end)
                    y = np.clip(y, y_start, y_end)
                    
                    # 检查距离机器人初始位置（如果min_distance_from_robot > 0）
                    if min_distance_from_robot > 0:
                        dist_from_robot = np.sqrt(
                            (x - robot_center_x)**2 + (y - robot_center_y)**2
                        )
                        
                        if dist_from_robot < min_distance_from_robot:
                            attempts += 1
                            continue
                    
                    too_close = False
                    for existing_pos in positions:
                        dist = np.sqrt(
                            (x - existing_pos[0])**2 + (y - existing_pos[1])**2
                        )
                        if dist < min_distance_between_obstacles:
                            too_close = True
                            break
                    
                    if not too_close:
                        valid_position = True
                        positions.append([x, y, z_height])
                    else:
                        attempts += 1
                
                if not valid_position:
                    # 如果网格位置无效，随机生成一个远离机器人的位置
                    if min_distance_from_robot > 0:
                        safe_x = np.random.uniform(max(x_start, robot_center_x + min_distance_from_robot), x_end)
                        safe_y = np.random.uniform(y_start, y_end)
                    else:
                        safe_x = np.random.uniform(x_start, x_end)
                        safe_y = np.random.uniform(y_start, y_end)
                    positions.append([safe_x, safe_y, z_height])
            
            # 如果障碍物数量超过网格大小，随机生成剩余位置
            for idx in range(actual_num, num_obstacles):
                valid_position = False
                attempts = 0
                
                while not valid_position and attempts < max_sample_tries:
                    x = np.random.uniform(x_start, x_end)
                    y = np.random.uniform(y_start, y_end)
                    
                    # 检查距离机器人初始位置（如果min_distance_from_robot > 0）
                    if min_distance_from_robot > 0:
                        dist_from_robot = np.sqrt(
                            (x - robot_center_x)**2 + (y - robot_center_y)**2
                        )
                        
                        if dist_from_robot < min_distance_from_robot:
                            attempts += 1
                            continue
                    
                    too_close = False
                    for existing_pos in positions:
                        dist = np.sqrt(
                            (x - existing_pos[0])**2 + (y - existing_pos[1])**2
                        )
                        if dist < min_distance_between_obstacles:
                            too_close = True
                            break
                    
                    if not too_close:
                        valid_position = True
                        positions.append([x, y, z_height])
                    else:
                        attempts += 1
                
                if not valid_position:
                    if min_distance_from_robot > 0:
                        safe_x = np.random.uniform(max(x_start, robot_center_x + min_distance_from_robot), x_end)
                        safe_y = np.random.uniform(y_start, y_end)
                    else:
                        safe_x = np.random.uniform(x_start, x_end)
                        safe_y = np.random.uniform(y_start, y_end)
                    positions.append([safe_x, safe_y, z_height])
        
        # 构建所有障碍物的状态张量
        object_states = torch.zeros(1, num_obstacles, 13, device=env.device)
        
        orientation = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=env.device),
            torch.tensor([0.0], device=env.device),
            torch.tensor([0.0], device=env.device)
        )
        
        for obj_idx in range(num_obstacles):
            pos = torch.tensor(positions[obj_idx], device=env.device)
            world_pos = pos + env.scene.env_origins[env_id, 0:3]
            
            object_states[0, obj_idx, 0:3] = world_pos
            object_states[0, obj_idx, 3:7] = orientation[0]
        
        env_id_tensor = torch.tensor([env_id], device=env.device)
        object_ids = torch.arange(num_obstacles, device=env.device)
        
        asset.write_object_state_to_sim(
            object_state=object_states,
            env_ids=env_id_tensor,
            object_ids=object_ids
        )


OBSTACLE_POSITIONS = [
    (2.0, 0.9, 0.5), (4.0, 1.8, 0.5), (6.0, 2.6, 0.5), (8.0, 3.6, 0.5),
    (10.0, 4.6, 0.5), (12.0, 5.6, 0.5), (13.5, 4.6, 0.5), (15.0, 3.7, 0.5),
    (17.0, 2.3, 0.5), (19.5, -0.3, 0.5), (21.0, -1.8, 0.5), (22.5, -2.6, 0.5),
    (24.0, -1.2, 0.5), (25.5, 0.2, 0.5), (26.5, 1.6, 0.5), (28.0, 3.0, 0.5),
    (29.5, 4.8, 0.5), (31.0, 6.2, 0.5), (32.5, 8.0, 0.5), (34.0, 9.2, 0.5),
    (35.5, 8.7, 0.5), (37.0, 8.0, 0.5), (38.5, 7.0, 0.5), (40.0, 6.0, 0.5),
    (41.0, 5.4, 0.5), (42.3, 5.0, 0.5), (44.0, 4.8, 0.5), (45.5, 6.2, 0.5),
    (48.5, 10.6, 0.5), (52.0, 13.3, 0.5),
]

def spawn_global_cylinder_obstacles(
    env,
    env_ids=None,
    radius: float = 0.5,
    height: float = 1.0,
    root_prim_path: str = "/World/Obstacles",
    color=(0.8, 0.2, 0.2),
    static_friction: float = 0.8,
    dynamic_friction: float = 0.8,
    restitution: float = 0.0,
    contact_offset: float = 0.01,
    rest_offset: float = 0.005,
):
    """Spawn global (non-replicated) static cylinder obstacles once.

    Important:
      - These are static colliders (no rigid body), so they will NOT be reset per-env.
      - This avoids CUDA index out of bounds from RigidObjectCollection.reset(env_ids).
    """
    # Ensure root prim exists
    positions = OBSTACLE_POSITIONS

    sim_utils.spawn_from_cfg(root_prim_path, sim_utils.XformCfg())

    physics_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=static_friction,
        dynamic_friction=dynamic_friction,
        restitution=restitution,
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
    )

    for i, pos in enumerate(positions):
        prim_path = f"{root_prim_path}/Obstacle_{i:02d}"

        # Static collider: do NOT set rigid_props
        cylinder_cfg = sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=contact_offset,
                rest_offset=rest_offset,
            ),
            physics_material=physics_material,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                metallic=0.0,
            ),
        )

        sim_utils.spawn_from_cfg(
            prim_path,
            cylinder_cfg,
            translation=pos,
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

def randomize_obstacles_per_env(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,          # NEW: use env.scene[asset_cfg.name]
    x_range: tuple[float, float] = (0.0, 54.0),
    y_range: tuple[float, float] = (-8.0, 10.0),
    z_height: float = 0.5,
    min_dist_between_obstacles: float = 0.4,
    min_dist_from_robot: float = 1.0,
    robot_asset_name: str = "robot",
    max_tries: int = 2000,
) -> None:
    device = env.device

    # normalize env_ids
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=device)
    env_ids = env_ids.to(device=device, dtype=torch.long)

    asset: RigidObjectCollection = env.scene[asset_cfg.name]
    num_obstacles = asset.num_objects

    origins = getattr(env.scene, "env_origins", None)
    if origins is None:
        origins_xy = torch.zeros((env.num_envs, 2), device=device)
        origins_xyz = torch.zeros((env.num_envs, 3), device=device)
    else:
        origins_xy = origins[:, :2].to(device)
        origins_xyz = origins[:, :3].to(device)

    robot = env.scene[robot_asset_name]
    robot_xy_local = robot.data.root_pos_w[:, :2] - origins_xy  # (N,2)

    x0, x1 = float(x_range[0]), float(x_range[1])
    y0, y1 = float(y_range[0]), float(y_range[1])
    min_d2 = float(min_dist_between_obstacles) ** 2
    min_dr2 = float(min_dist_from_robot) ** 2

    # fixed orientation (identity)
    orientation = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)  # (1,4)

    object_ids = torch.arange(num_obstacles, device=device, dtype=torch.long)

    for e in env_ids.tolist():
        placed: list[torch.Tensor] = []
        tries = 0
        while len(placed) < num_obstacles and tries < max_tries:
            tries += 1
            cand = torch.empty((2,), device=device)
            cand[0] = (x1 - x0) * torch.rand((), device=device) + x0
            cand[1] = (y1 - y0) * torch.rand((), device=device) + y0

            # keep away from robot (in env-local frame)
            if torch.sum((cand - robot_xy_local[e]) ** 2) < min_dr2:
                continue

            ok = True
            for p in placed:
                if torch.sum((cand - p) ** 2) < min_d2:
                    ok = False
                    break
            if not ok:
                continue
            placed.append(cand)

        # fill remaining if needed (avoid dead loop)
        while len(placed) < num_obstacles:
            cand = torch.empty((2,), device=device)
            cand[0] = (x1 - x0) * torch.rand((), device=device) + x0
            cand[1] = (y1 - y0) * torch.rand((), device=device) + y0
            placed.append(cand)

        pos_local = torch.stack(placed, dim=0)  # (M,2)

        # build object state: (1, M, 13)
        object_states = torch.zeros((1, num_obstacles, 13), device=device)
        pos_w = torch.zeros((num_obstacles, 3), device=device)
        pos_w[:, :2] = pos_local + origins_xy[e].unsqueeze(0)
        pos_w[:, 2] = float(z_height)

        object_states[0, :, 0:3] = pos_w
        object_states[0, :, 3:7] = orientation.repeat(num_obstacles, 1)

        env_id_tensor = torch.tensor([e], device=device, dtype=torch.long)
        asset.write_object_state_to_sim(
            object_state=object_states,
            env_ids=env_id_tensor,
            object_ids=object_ids,
        )