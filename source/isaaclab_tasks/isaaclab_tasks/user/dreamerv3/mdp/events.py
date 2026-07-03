import torch
import numpy as np

from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import RigidObjectCollection
from isaaclab.utils import math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def _build_dense_path(points: list[tuple[float, float]], step: float | None = 0.2) -> dict[str, torch.Tensor]:
    """Build a polyline path with yaw and arc length."""
    pts = torch.tensor(points, dtype=torch.float32)
    if step is None or step <= 0:
        xy = pts
    else:
        segments = []
        for idx in range(len(pts) - 1):
            p0 = pts[idx]
            p1 = pts[idx + 1]
            delta = p1 - p0
            length = float(torch.norm(delta))
            if length < 1e-6:
                continue
            num = max(2, int(np.ceil(length / step)) + 1)
            alpha = torch.linspace(0.0, 1.0, steps=num, dtype=torch.float32)
            seg = p0.unsqueeze(0) + alpha.unsqueeze(1) * delta.unsqueeze(0)
            if idx > 0:
                seg = seg[1:]
            segments.append(seg)
        xy = torch.cat(segments, dim=0)
    diff = xy[1:] - xy[:-1]
    seg_len = torch.norm(diff, dim=1)
    s = torch.cat([torch.zeros(1, dtype=torch.float32), torch.cumsum(seg_len, dim=0)], dim=0)
    yaw = torch.atan2(diff[:, 1], diff[:, 0])
    yaw = torch.cat([yaw, yaw[-1:]], dim=0)
    curvature = torch.zeros_like(s)
    if yaw.numel() > 2:
        dyaw = math_utils.wrap_to_pi(yaw[1:] - yaw[:-1])
        ds = torch.clamp(s[1:] - s[:-1], min=1e-6)
        curvature[1:] = dyaw / ds
    return {"xy": xy, "yaw": yaw, "s": s, "curvature": curvature}


def _get_path_library(device: torch.device) -> list[dict[str, torch.Tensor]]:
    """Create reference paths for S-curve and Waterland +x traversal tasks."""
    raw_paths = [
        [
            (0.8, 0.0),
            (2.2, -2.1),
            (3.8, -3.2),
            (5.2, -2.8),
            (6.6, -1.1),
            (8.0, 1.1),
            (9.4, 2.8),
            (10.8, 3.2),
            (12.2, 2.1),
            (13.8, 0.0),
        ],
        [
            (-9.2, 0.0),
            (-6.5, 0.0),
            (-3.5, 0.0),
            (-0.5, 0.0),
            (2.5, 0.0),
            (5.5, 0.0),
            (8.5, 0.0),
            (11.5, 0.0),
            (13.2, 0.0),
        ],
    ]
    library = [
        _build_dense_path(raw_paths[0], step=None),
        _build_dense_path(raw_paths[1], step=0.2),
    ]
    for path in library:
        for key, value in path.items():
            path[key] = value.to(device=device)
    return library


def _sample_square_obstacles(
    rng: np.random.Generator,
    path: dict[str, torch.Tensor],
    num_obstacles: int,
    square_x: tuple[float, float],
    square_y: tuple[float, float],
    min_clearance_to_path: float,
    min_clearance_between: float,
    start_clearance: float,
    goal_clearance: float,
    max_sample_tries: int,
) -> list[tuple[float, float]]:
    """Sample dense random cube centers while keeping the S-path traversable."""
    positions: list[tuple[float, float]] = []
    x_min, x_max = square_x
    y_min, y_max = square_y
    path_xy = path["xy"].cpu().numpy()
    start_xy = path_xy[0]
    goal_xy = path_xy[-1]

    attempts = 0
    while len(positions) < num_obstacles and attempts < max_sample_tries:
        attempts += 1
        candidate = np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)], dtype=np.float32)
        if np.min(np.linalg.norm(path_xy - candidate[None, :], axis=1)) < min_clearance_to_path:
            continue
        if np.linalg.norm(candidate - start_xy) < start_clearance:
            continue
        if np.linalg.norm(candidate - goal_xy) < goal_clearance:
            continue
        if any(np.linalg.norm(candidate - np.array(pos, dtype=np.float32)) < min_clearance_between for pos in positions):
            continue
        positions.append((float(candidate[0]), float(candidate[1])))
    return positions


def _ensure_local_nav_buffers(env: ManagerBasedEnv):
    device = env.device
    if hasattr(env, "_local_nav_path_ids"):
        return
    env._local_nav_path_library = _get_path_library(device)
    env._local_nav_path_ids = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._local_nav_start_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._local_nav_progress_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._local_nav_goal_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._local_nav_prev_s = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._local_nav_last_delta_s = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._local_nav_waypoint_advanced = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._local_nav_start_s = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._local_nav_no_progress_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._local_nav_episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._local_nav_collision_flag = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env._local_nav_prev_in_contact = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    env._local_nav_progress_update_step = -1
    env._local_nav_cached_track = None


def _ensure_contact_memory_buffers(
    env: ManagerBasedEnv,
    num_sectors: int = 8,
    decay: float = 0.995,
):
    """Create/refresh per-env contact-memory buffers used by observation labels."""
    device = env.device
    num_sectors = max(4, int(num_sectors))
    # keep runtime config visible to observation side
    env._contact_memory_num_sectors = num_sectors
    env._contact_memory_decay = float(decay)

    needs_init = (
        (not hasattr(env, "_contact_memory_world"))
        or env._contact_memory_world.shape[0] != env.num_envs
        or env._contact_memory_world.shape[1] != num_sectors
    )
    if needs_init:
        env._contact_memory_world = torch.zeros(
            (env.num_envs, num_sectors), dtype=torch.float32, device=device
        )
        robot = env.scene["robot"]
        env._contact_memory_initial_yaw = robot.data.heading_w.detach().clone()
        env._contact_memory_last_update_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=device
        )
        env._contact_memory_last_reset_step = torch.full(
            (env.num_envs,), -1, dtype=torch.long, device=device
        )
    else:
        if (
            not hasattr(env, "_contact_memory_last_update_step")
            or env._contact_memory_last_update_step.shape[0] != env.num_envs
        ):
            env._contact_memory_last_update_step = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=device
            )
        if (
            not hasattr(env, "_contact_memory_last_reset_step")
            or env._contact_memory_last_reset_step.shape[0] != env.num_envs
        ):
            env._contact_memory_last_reset_step = torch.full(
                (env.num_envs,), -1, dtype=torch.long, device=device
            )


def _sample_obstacle_template(
    rng: np.random.Generator,
    corridor_s: tuple[float, float],
    corridor_half_width: float,
) -> list[tuple[float, float, float]]:
    """Return local Frenet-style obstacles as (s_rel, d, radius)."""
    s_min, s_max = corridor_s
    template_id = int(rng.integers(0, 6))
    r = float(rng.uniform(0.18, 0.32))
    if template_id == 0:  # single-side block
        side = -1.0 if rng.random() < 0.5 else 1.0
        return [(float(rng.uniform(s_min + 0.5, s_max - 0.5)), side * float(rng.uniform(0.5, 1.0)), r)]
    if template_id == 1:  # center obstacle
        return [(float(rng.uniform(s_min + 1.0, s_max - 1.0)), float(rng.uniform(-0.2, 0.2)), r)]
    if template_id == 2:  # narrow passage
        s_obs = float(rng.uniform(s_min + 1.0, s_max - 1.0))
        gap_half = float(rng.uniform(0.35, 0.55))
        offset = float(rng.uniform(0.75, 1.1))
        return [(s_obs, offset, r), (s_obs, -offset, r)]
    if template_id == 3:  # bend inside
        return [(float(rng.uniform(s_min + 1.0, s_max - 1.0)), float(rng.uniform(0.4, corridor_half_width - 0.2)), r)]
    if template_id == 4:  # bend outside
        return [(float(rng.uniform(s_min + 1.0, s_max - 1.0)), -float(rng.uniform(0.4, corridor_half_width - 0.2)), r)]
    s0 = float(rng.uniform(s_min + 0.5, s_max - 2.0))
    gap = float(rng.uniform(1.5, 2.5))
    return [
        (s0, float(rng.uniform(-0.4, 0.4)), r),
        (s0 + gap, float(rng.uniform(-0.8, 0.8)), r),
    ]


def _sample_recover_template_obstacles(
    rng: np.random.Generator,
    path: dict[str, torch.Tensor],
    start_idx: int,
    forward_distance_range: tuple[float, float],
    template_set: str = "basic",
) -> list[tuple[float, float]]:
    """Return local obstacle centers for controlled stuck-recovery templates."""
    if template_set == "two_point":
        templates = (
            "front_block",
            "front_left_block",
            "front_right_block",
            "rear_limited",
            "narrow_gap",
            "front_double_offset_gap",
        )
    else:
        templates = ("front_block", "front_left_block", "front_right_block")
    template = rng.choice(templates)
    distance = float(rng.uniform(*forward_distance_range))
    path_xy = path["xy"]
    base_xy = path_xy[start_idx]
    yaw = path["yaw"][start_idx]
    tangent = torch.stack([torch.cos(yaw), torch.sin(yaw)])
    normal = torch.stack([-torch.sin(yaw), torch.cos(yaw)])

    if template == "rear_limited":
        specs = ((distance, 0.0), (-0.45, 0.0))
    elif template == "narrow_gap":
        gap_half = float(rng.uniform(0.28, 0.38))
        specs = ((distance, gap_half + 0.22), (distance, -(gap_half + 0.22)))
    elif template == "front_double_offset_gap":
        gap_side = -1.0 if rng.random() < 0.5 else 1.0
        specs = ((distance, -0.24 * gap_side), (distance + 0.18, 0.38 * gap_side))
    else:
        lateral_offsets = {
            "front_block": (0.0,),
            "front_left_block": (0.28,),
            "front_right_block": (-0.28,),
        }[template]
        specs = tuple((distance, offset) for offset in lateral_offsets)
    positions = []
    for forward, offset in specs:
        xy = base_xy + float(forward) * tangent + float(offset) * normal
        positions.append((float(xy[0].item()), float(xy[1].item())))
    return positions


def _make_two_point_path(
    device: torch.device,
    goal_distance_range: tuple[float, float],
    goal_lateral_range: tuple[float, float],
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    """Create a short per-env start-goal path for local recovery training."""
    goal_x = float(rng.uniform(*goal_distance_range))
    goal_y = float(rng.uniform(*goal_lateral_range))
    path = _build_dense_path([(0.0, 0.0), (goal_x, goal_y)], step=None)
    return {key: value.to(device=device) for key, value in path.items()}


def _obstacles_are_feasible(obstacles: list[tuple[float, float, float]], corridor_half_width: float) -> bool:
    for i, (s_i, d_i, r_i) in enumerate(obstacles):
        if abs(d_i) + r_i > corridor_half_width:
            return False
        for j in range(i + 1, len(obstacles)):
            s_j, d_j, r_j = obstacles[j]
            if (s_i - s_j) ** 2 + (d_i - d_j) ** 2 < (r_i + r_j + 0.2) ** 2:
                return False
    if len(obstacles) >= 2:
        lateral_sorted = sorted(obstacles, key=lambda item: item[1])
        min_gap = min(
            lateral_sorted[idx + 1][1] - lateral_sorted[idx][1] - lateral_sorted[idx][2] - lateral_sorted[idx + 1][2]
            for idx in range(len(lateral_sorted) - 1)
        )
        if min_gap < 0.5:
            return False
    return True


def _path_pose_from_index(path: dict[str, torch.Tensor], index: int) -> tuple[torch.Tensor, torch.Tensor]:
    pos = path["xy"][index]
    yaw = path["yaw"][index]
    return pos, yaw


def _visualize_local_nav_reference(env: ManagerBasedEnv, num_points: int = 16, ds: float = 0.5):
    """Draw the currently sampled local reference points for all environments."""
    if not hasattr(env, "_local_nav_path_ids") or env.num_envs == 0:
        return
    if not hasattr(env, "_local_nav_ref_visualizer"):
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/LocalReferencePath",
            markers={
                "passed_point": sim_utils.SphereCfg(
                    radius=0.07,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.2, 0.9, 0.2),
                        metallic=0.0,
                        roughness=0.4,
                    ),
                ),
                "future_point": sim_utils.SphereCfg(
                    radius=0.08,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.1, 0.8, 1.0),
                        metallic=0.0,
                        roughness=0.4,
                    ),
                ),
                "current_point": sim_utils.SphereCfg(
                    radius=0.1,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.75, 0.1),
                        metallic=0.0,
                        roughness=0.4,
                    ),
                ),
            },
        )
        env._local_nav_ref_visualizer = VisualizationMarkers(marker_cfg)

    points = []
    marker_ids = []
    for env_id in range(env.num_envs):
        path_id = int(env._local_nav_path_ids[env_id].item())
        path = env._local_nav_path_library[path_id]
        current_idx = int(env._local_nav_progress_idx[env_id].item())
        lookbehind = max(1, num_points // 3)
        origin_xy = env.scene.env_origins[env_id, :2]
        for step_idx in range(num_points):
            ref_idx = max(0, min(len(path["s"]) - 1, current_idx - lookbehind + step_idx))
            ref_xy = path["xy"][ref_idx] + origin_xy
            points.append(torch.tensor([ref_xy[0], ref_xy[1], 0.12], device=env.device))
            if ref_idx < current_idx:
                marker_ids.append(0)
            elif ref_idx == current_idx:
                marker_ids.append(2)
            else:
                marker_ids.append(1)

    translations = torch.stack(points, dim=0)
    orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device).repeat(len(points), 1)
    marker_indices = torch.tensor(marker_ids, dtype=torch.int32, device=env.device)
    env._local_nav_ref_visualizer.visualize(
        translations=translations,
        orientations=orientations,
        marker_indices=marker_indices,
    )


def reset_local_nav_task(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    obstacle_asset_cfg: SceneEntityCfg | None = None,
    s_margin_start: float = 3.0,
    s_margin_end: float = 8.0,
    lateral_offset_range: tuple[float, float] = (-0.2, 0.2),
    heading_offset_range: tuple[float, float] = (-0.15, 0.15),
    start_speed_range: tuple[float, float] = (0.0, 0.2),
    max_obstacle_trials: int = 32,
    fixed_path_id: int | None = None,
    path_mode: str = "library",
    goal_distance_range: tuple[float, float] = (1.5, 3.0),
    goal_lateral_range: tuple[float, float] = (-0.3, 0.3),
    start_idx_range: tuple[int, int] | None = None,
    waterland_height_reset: bool = False,
    waterland_x_min: float = -10.0,
    waterland_stage_len: float = 1.5,
    waterland_cycle_len: float = 6.0,
    waterland_high_z: float = 1.0,
    waterland_low_z: float = 0.25,
    root_height_offset: float = 0.2,
    disable_obstacles: bool = False,
    debug_vis: bool = False,
    debug_vis_num_points: int = 16,
    debug_vis_ds: float = 0.5,
    waypoint_reach_thresh: float = 0.45,
    square_x: tuple[float, float] = (0.3, 7.7),
    square_y: tuple[float, float] = (-3.7, 3.7),
    obstacle_path_clearance: float = 0.5,
    obstacle_spacing: float = 0.55,
    obstacle_start_clearance: float = 0.9,
    obstacle_goal_clearance: float = 0.9,
    obstacle_mode: str = "random_square",
    recover_obstacle_distance_range: tuple[float, float] = (0.6, 1.0),
    recover_template_set: str = "basic",
    contact_memory_num_sectors: int = 8,
    contact_memory_decay: float = 0.995,
):
    """Reset robot pose and local task state using a path library and obstacle templates."""
    _ensure_local_nav_buffers(env)
    _ensure_contact_memory_buffers(
        env,
        num_sectors=contact_memory_num_sectors,
        decay=contact_memory_decay,
    )
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)

    robot = env.scene[asset_cfg.name]
    obstacles: RigidObjectCollection | None = None
    num_obstacles = 0
    if not disable_obstacles:
        if obstacle_asset_cfg is None:
            raise ValueError(
                "reset_local_nav_task requires obstacle_asset_cfg when disable_obstacles=False."
            )
        obstacles = env.scene[obstacle_asset_cfg.name]
        num_obstacles = obstacles.num_objects
    device = env.device
    rng = np.random.default_rng()

    env._local_nav_debug_vis = bool(debug_vis)
    env._local_nav_debug_vis_num_points = int(debug_vis_num_points)
    env._local_nav_debug_vis_ds = float(debug_vis_ds)
    env._local_nav_debug_vis_interval = 1
    env._local_nav_waypoint_reach_thresh = float(waypoint_reach_thresh)

    default_root_state = robot.data.default_root_state.clone()
    root_pose = default_root_state[env_ids][:, :7].clone()
    root_vel = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=device)
    contact_init_yaw = torch.zeros((len(env_ids),), dtype=torch.float32, device=device)

    object_states = None
    if not disable_obstacles:
        object_states = torch.zeros((len(env_ids), num_obstacles, 13), dtype=torch.float32, device=device)
        object_states[:, :, 2] = -10.0
        object_states[:, :, 3] = 1.0

    for local_idx, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        if path_mode == "two_point":
            while len(env._local_nav_path_library) <= env_id:
                env._local_nav_path_library.append(_make_two_point_path(device, goal_distance_range, goal_lateral_range, rng))
            env._local_nav_path_library[env_id] = _make_two_point_path(
                device,
                goal_distance_range=goal_distance_range,
                goal_lateral_range=goal_lateral_range,
                rng=rng,
            )
            path_id = env_id
        elif path_mode == "library":
            path_id = 0 if fixed_path_id is None else int(fixed_path_id)
        else:
            raise ValueError(f"Unsupported path_mode: {path_mode}")
        path = env._local_nav_path_library[path_id]
        goal_idx = len(path["s"]) - 1
        start_idx = 0
        if start_idx_range is not None:
            lo = max(0, int(start_idx_range[0]))
            hi = min(goal_idx - 1, int(start_idx_range[1]))
            if hi > lo:
                start_idx = int(rng.integers(lo, hi + 1))
        base_pos_local, base_yaw = _path_pose_from_index(path, start_idx)

        e_y = float(rng.uniform(*lateral_offset_range))
        e_psi = float(rng.uniform(*heading_offset_range))
        speed = float(rng.uniform(*start_speed_range))

        normal = torch.tensor([-torch.sin(base_yaw), torch.cos(base_yaw)], device=device)
        pos_local_xy = base_pos_local + e_y * normal
        yaw = base_yaw + e_psi
        pos_world_xy = pos_local_xy + env.scene.env_origins[env_id, :2]
        pos_z = default_root_state[env_id, 2]
        if waterland_height_reset:
            phase = torch.remainder(pos_local_xy[0] - float(waterland_x_min), float(waterland_cycle_len))
            stage_len = max(float(waterland_stage_len), 1e-6)
            if phase < stage_len:
                terrain_z = float(waterland_high_z)
            elif phase < 2.0 * stage_len:
                r = float((phase - stage_len) / stage_len)
                terrain_z = float(waterland_high_z) + (float(waterland_low_z) - float(waterland_high_z)) * r
            elif phase < 3.0 * stage_len:
                terrain_z = float(waterland_low_z)
            else:
                r = float((phase - 3.0 * stage_len) / stage_len)
                terrain_z = float(waterland_low_z) + (float(waterland_high_z) - float(waterland_low_z)) * r
            pos_z = terrain_z + float(root_height_offset)
        quat = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=device),
            torch.tensor([0.0], device=device),
            yaw.unsqueeze(0),
        )[0]
        root_pose[local_idx, :3] = torch.tensor([pos_world_xy[0], pos_world_xy[1], pos_z], device=device)
        root_pose[local_idx, 3:7] = quat
        root_vel[local_idx, 0] = speed * torch.cos(yaw)
        root_vel[local_idx, 1] = speed * torch.sin(yaw)
        contact_init_yaw[local_idx] = yaw

        env._local_nav_path_ids[env_id] = path_id
        env._local_nav_start_idx[env_id] = start_idx
        env._local_nav_progress_idx[env_id] = min(start_idx + 1, goal_idx)
        env._local_nav_goal_idx[env_id] = goal_idx
        env._local_nav_prev_s[env_id] = path["s"][start_idx]
        env._local_nav_last_delta_s[env_id] = 0.0
        env._local_nav_waypoint_advanced[env_id] = 0.0
        env._local_nav_start_s[env_id] = path["s"][start_idx]
        env._local_nav_no_progress_steps[env_id] = 0
        env._local_nav_episode_steps[env_id] = 0
        env._local_nav_collision_flag[env_id] = False
        env._local_nav_prev_in_contact[env_id] = False
        env._local_nav_progress_update_step = -1
        env._local_nav_cached_track = None
        # Reset contact memory state for this episode.
        env._contact_memory_world[env_id] = 0.0
        env._contact_memory_last_update_step[env_id] = -1
        env._contact_memory_last_reset_step[env_id] = -1

        if not disable_obstacles:
            if obstacle_mode == "recover_template":
                local_obstacles = _sample_recover_template_obstacles(
                    rng=rng,
                    path=path,
                    start_idx=start_idx,
                    forward_distance_range=recover_obstacle_distance_range,
                    template_set=recover_template_set,
                )
            elif obstacle_mode == "random_square":
                local_obstacles = _sample_square_obstacles(
                    rng=rng,
                    path=path,
                    num_obstacles=num_obstacles,
                    square_x=square_x,
                    square_y=square_y,
                    min_clearance_to_path=obstacle_path_clearance,
                    min_clearance_between=obstacle_spacing,
                    start_clearance=obstacle_start_clearance,
                    goal_clearance=obstacle_goal_clearance,
                    max_sample_tries=max_obstacle_trials * max(32, num_obstacles),
                )
            else:
                raise ValueError(f"Unsupported obstacle_mode: {obstacle_mode}")
            for obs_idx, (obs_x, obs_y) in enumerate(local_obstacles[:num_obstacles]):
                obs_xy_world = torch.tensor([obs_x, obs_y], device=device) + env.scene.env_origins[env_id, :2]
                object_states[local_idx, obs_idx, 0:3] = torch.tensor(
                    [obs_xy_world[0], obs_xy_world[1], 0.5],
                    device=device,
                )
                object_states[local_idx, obs_idx, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    # Set per-episode local-world yaw origin.
    env._contact_memory_initial_yaw[env_ids] = contact_init_yaw

    if not disable_obstacles:
        object_ids = torch.arange(num_obstacles, device=device, dtype=torch.long)
        obstacles.write_object_state_to_sim(object_states, env_ids=env_ids, object_ids=object_ids)
    if debug_vis:
        _visualize_local_nav_reference(env, num_points=debug_vis_num_points, ds=debug_vis_ds)


def _ensure_blocked_recovery_buffers(env: ManagerBasedEnv):
    device = env.device
    if hasattr(env, "_blocked_recovery_scenario_id") and env._blocked_recovery_scenario_id.shape[0] == env.num_envs:
        return
    env._blocked_recovery_scenario_id = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._blocked_recovery_start_x = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._blocked_recovery_prev_x = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._blocked_recovery_delta_x = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._blocked_recovery_no_progress_steps = torch.zeros(env.num_envs, dtype=torch.long, device=device)
    env._blocked_recovery_slip_duration = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._blocked_recovery_contact_duration = torch.zeros(env.num_envs, dtype=torch.float32, device=device)
    env._blocked_recovery_energy_proxy = torch.zeros(env.num_envs, dtype=torch.float32, device=device)


def reset_blocked_recovery_task(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    obstacle_asset_cfg: SceneEntityCfg,
    scenario_weights: tuple[float, float, float] = (1.0, 0.0, 0.0),
    start_x_range: tuple[float, float] = (-0.08, 0.08),
    start_y_range: tuple[float, float] = (-0.05, 0.05),
    heading_range: tuple[float, float] = (-0.04, 0.04),
    start_speed_range: tuple[float, float] = (0.0, 0.03),
    root_z: float = 0.18,
    curb_x_range: tuple[float, float] = (0.42, 0.55),
    belly_x_range: tuple[float, float] = (0.30, 0.45),
    success_distance: float = 1.15,
):
    """Reset into blocked recovery primitive-selection scenarios.

    Scenario ids:
    0 curb_momentum_loss, 1 belly_high_center, 2 low_traction_duration.
    First training stage can pass weights=(1,0,0) to use curb only.
    """
    _ensure_blocked_recovery_buffers(env)
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)

    robot = env.scene[asset_cfg.name]
    obstacles: RigidObjectCollection = env.scene[obstacle_asset_cfg.name]
    num_obstacles = obstacles.num_objects
    device = env.device
    rng = np.random.default_rng()

    weights = np.asarray(scenario_weights, dtype=np.float64)
    if weights.size != 3 or np.sum(weights) <= 0:
        raise ValueError("scenario_weights must contain three non-negative values with positive sum.")
    weights = weights / np.sum(weights)

    root_pose = robot.data.default_root_state[env_ids, :7].clone()
    root_vel = torch.zeros((len(env_ids), 6), dtype=torch.float32, device=device)
    object_states = torch.zeros((len(env_ids), num_obstacles, 13), dtype=torch.float32, device=device)
    object_states[:, :, 2] = -10.0
    object_states[:, :, 3] = 1.0

    env_origins = env.scene.env_origins if hasattr(env.scene, "env_origins") else torch.zeros((env.num_envs, 3), device=device)
    for local_idx, env_id_tensor in enumerate(env_ids):
        env_id = int(env_id_tensor.item())
        scenario_id = int(rng.choice(3, p=weights))
        x0 = float(rng.uniform(*start_x_range))
        y0 = float(rng.uniform(*start_y_range))
        yaw = torch.tensor(float(rng.uniform(*heading_range)), device=device)
        speed = float(rng.uniform(*start_speed_range))
        origin = env_origins[env_id]

        quat = math_utils.quat_from_euler_xyz(
            torch.tensor([0.0], device=device),
            torch.tensor([0.0], device=device),
            yaw.unsqueeze(0),
        )[0]
        root_pose[local_idx, :3] = torch.tensor([x0, y0, float(root_z)], device=device) + origin
        root_pose[local_idx, 3:7] = quat
        root_vel[local_idx, 0] = speed * torch.cos(yaw)
        root_vel[local_idx, 1] = speed * torch.sin(yaw)

        env._blocked_recovery_scenario_id[env_id] = scenario_id
        env._blocked_recovery_start_x[env_id] = x0
        env._blocked_recovery_prev_x[env_id] = x0
        env._blocked_recovery_delta_x[env_id] = 0.0
        env._blocked_recovery_no_progress_steps[env_id] = 0
        env._blocked_recovery_slip_duration[env_id] = 0.0
        env._blocked_recovery_contact_duration[env_id] = 0.0
        env._blocked_recovery_energy_proxy[env_id] = 0.0

        if hasattr(env, "_recovery_action_switch_count"):
            env._recovery_action_switch_count[env_id] = 0.0
            env._recovery_invalid_action_count[env_id] = 0.0
            env._recovery_action_switch_pulse[env_id] = 0.0
            env._recovery_invalid_action_pulse[env_id] = 0.0

        # Object 0: curb, Object 1: belly/high-center obstacle. Hide inactive objects.
        if scenario_id == 0 and num_obstacles >= 1:
            curb_x = float(rng.uniform(*curb_x_range))
            object_states[local_idx, 0, 0:3] = torch.tensor([curb_x, 0.0, 0.06], device=device) + origin
            object_states[local_idx, 0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        elif scenario_id == 1 and num_obstacles >= 2:
            belly_x = float(rng.uniform(*belly_x_range))
            object_states[local_idx, 1, 0:3] = torch.tensor([belly_x, 0.0, 0.075], device=device) + origin
            object_states[local_idx, 1, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        # scenario 2 uses proprioceptive slip proxies first; no privileged policy input.

    robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
    object_ids = torch.arange(num_obstacles, device=device, dtype=torch.long)
    obstacles.write_object_state_to_sim(object_states, env_ids=env_ids, object_ids=object_ids)
    env._blocked_recovery_success_distance = float(success_distance)


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
