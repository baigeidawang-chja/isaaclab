from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils import math as math_utils

from .events import _ensure_local_nav_buffers

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


CONTACT_SENSOR_ANGLES_DEG = (0.0, 33.7, 146.3, 180.0, 213.7, 326.3)


def get_contact_sensor_feedback(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return max contact force and binary contact flag."""
    sensor = env.scene.sensors.get("robot_contact_sensor", None)
    if sensor is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    # Observation manager queries shapes during env construction, which can happen
    # before the contact sensor has created its PhysX view. In that case, fall back
    # to zeros and let runtime steps use the real sensor data once available.
    try:
        forces = sensor.data.net_forces_w
    except (AttributeError, RuntimeError):
        return torch.zeros((env.num_envs, 2), device=env.device)

    if forces is None or forces.numel() == 0:
        return torch.zeros((env.num_envs, 2), device=env.device)

    force_mag = torch.norm(forces, dim=-1)
    max_force, _ = torch.max(force_mag, dim=1)
    contact_flag = (max_force > 1e-3).float()
    return torch.stack([max_force, contact_flag], dim=-1)


def _angle_to_sector(angle_rad: torch.Tensor, num_sectors: int) -> torch.Tensor:
    """Map any angle (rad) to sector id [0, num_sectors-1], sector 0 centered at 0 deg."""
    two_pi = 2.0 * math.pi
    angle = torch.remainder(angle_rad, two_pi)
    sector_size = two_pi / float(num_sectors)
    sector_id = torch.floor((angle + sector_size * 0.5) / sector_size).long()
    return sector_id % num_sectors


def _get_contact_now_6(
    env: ManagerBasedRLEnv,
    force_threshold: float = 1e-3,
) -> torch.Tensor:
    """Return current 6-direction contact activation [N, 6] in fixed order.

    Preferred order:
    [front, front_left, rear_left, rear, rear_right, front_right]
    """
    sensor = env.scene.sensors.get("robot_contact_sensor", None)
    if sensor is None:
        return torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)

    try:
        forces = sensor.data.net_forces_w  # [N, B, 3]
    except (AttributeError, RuntimeError):
        return torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)

    if forces is None or forces.numel() == 0:
        return torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)

    force_mag = torch.norm(forces, dim=-1)  # [N, B]
    num_bodies = force_mag.shape[1]
    if hasattr(env, "_contact_sensor_body_indices_6"):
        idx = getattr(env, "_contact_sensor_body_indices_6")
        idx = torch.as_tensor(idx, device=env.device, dtype=torch.long).clamp(0, max(0, num_bodies - 1))
        if idx.numel() >= 6:
            idx = idx[:6]
            selected = force_mag[:, idx]
        else:
            selected = torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)
            if idx.numel() > 0:
                selected[:, : idx.numel()] = force_mag[:, idx]
    else:
        selected = torch.zeros((env.num_envs, 6), dtype=torch.float32, device=env.device)
        take = min(6, num_bodies)
        selected[:, :take] = force_mag[:, :take]
    return (selected > force_threshold).float()


def _ensure_contact_memory_buffers(
    env: ManagerBasedRLEnv,
    num_sectors: int,
):
    """Lazy-create contact-memory buffers on env."""
    device = env.device
    if (not hasattr(env, "_contact_memory_world")) or (env._contact_memory_world.shape[-1] != num_sectors):
        env._contact_memory_world = torch.zeros((env.num_envs, num_sectors), dtype=torch.float32, device=device)
        env._contact_memory_initial_yaw = env.scene["robot"].data.heading_w.detach().clone()
        env._contact_memory_last_update_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
        env._contact_memory_last_reset_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
    else:
        if (
            not hasattr(env, "_contact_memory_last_update_step")
            or env._contact_memory_last_update_step.shape[0] != env.num_envs
        ):
            env._contact_memory_last_update_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
        if (
            not hasattr(env, "_contact_memory_last_reset_step")
            or env._contact_memory_last_reset_step.shape[0] != env.num_envs
        ):
            env._contact_memory_last_reset_step = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)


def _get_contact_memory_current_step(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return per-env episode step used to guard contact-memory updates."""
    device = env.device
    if hasattr(env, "episode_length_buf"):
        return env.episode_length_buf.to(device=device, dtype=torch.long)
    if hasattr(env, "_local_nav_episode_steps"):
        return env._local_nav_episode_steps.to(device=device, dtype=torch.long)
    step = int(getattr(env, "common_step_counter", 0))
    return torch.full((env.num_envs,), step, dtype=torch.long, device=device)


def _reset_contact_memory_if_new_episode(env: ManagerBasedRLEnv):
    """Reset memory for envs that have just started a new episode."""
    step_ctr = _get_contact_memory_current_step(env)
    reset_mask = (step_ctr <= 1) & (
        (env._contact_memory_last_update_step > step_ctr)
        | (env._contact_memory_last_reset_step < 0)
    )
    if reset_mask.any():
        env._contact_memory_world[reset_mask] = 0.0
        env._contact_memory_initial_yaw[reset_mask] = env.scene["robot"].data.heading_w[reset_mask].detach().clone()
        env._contact_memory_last_update_step[reset_mask] = -1
        env._contact_memory_last_reset_step[reset_mask] = step_ctr[reset_mask]


def _update_contact_memory_world_from_6sensors(
    env: ManagerBasedRLEnv,
    contact_now_6: torch.Tensor,   # [N, 6]
    robot_yaw_local: torch.Tensor,  # [N]
    decay: float,
    num_sectors: int,
    update_mask: torch.Tensor,
):
    """Update local-world memory from current 6-direction contacts."""
    if not update_mask.any():
        return
    env._contact_memory_world[update_mask] *= float(decay)
    contact_angles = torch.tensor(CONTACT_SENSOR_ANGLES_DEG, dtype=torch.float32, device=env.device) * (math.pi / 180.0)
    env_ids = torch.arange(env.num_envs, device=env.device)

    for k in range(min(6, contact_now_6.shape[1])):
        active = (contact_now_6[:, k] > 0.5) & update_mask
        if not active.any():
            continue
        angle_world = robot_yaw_local + contact_angles[k]
        sector_world = _angle_to_sector(angle_world, num_sectors)
        env._contact_memory_world[env_ids[active], sector_world[active]] = 1.0


def _world_memory_to_body_memory(
    contact_memory_world: torch.Tensor,  # [N, S]
    robot_yaw_local: torch.Tensor,       # [N]
    num_sectors: int,
) -> torch.Tensor:
    """Rotate local-world memory back into current body frame sectors."""
    device = contact_memory_world.device
    num_envs = contact_memory_world.shape[0]
    body_memory = torch.zeros_like(contact_memory_world)
    env_ids = torch.arange(num_envs, device=device)
    sector_size = 2.0 * math.pi / float(num_sectors)
    world_angles = torch.arange(num_sectors, device=device, dtype=torch.float32) * sector_size

    for ws in range(num_sectors):
        angle_body = world_angles[ws] - robot_yaw_local
        sector_body = _angle_to_sector(angle_body, num_sectors)
        val = contact_memory_world[:, ws]
        body_memory[env_ids, sector_body] = torch.maximum(body_memory[env_ids, sector_body], val)
    return body_memory


def get_contact_memory_body_label(
    env: ManagerBasedRLEnv,
    num_sectors: int = 8,
    decay: float = 0.995,
    force_threshold: float = 1e-3,
) -> torch.Tensor:
    """Return 360-deg body-frame contact memory label [N, num_sectors]."""
    _ensure_contact_memory_buffers(env, num_sectors=num_sectors)
    _reset_contact_memory_if_new_episode(env)

    current_step = _get_contact_memory_current_step(env)
    update_mask = current_step != env._contact_memory_last_update_step
    robot_yaw = env.scene["robot"].data.heading_w
    robot_yaw_local = torch.atan2(
        torch.sin(robot_yaw - env._contact_memory_initial_yaw),
        torch.cos(robot_yaw - env._contact_memory_initial_yaw),
    )
    if update_mask.any():
        contact_now_6 = _get_contact_now_6(env, force_threshold=force_threshold)
        _update_contact_memory_world_from_6sensors(
            env,
            contact_now_6=contact_now_6,
            robot_yaw_local=robot_yaw_local,
            decay=decay,
            num_sectors=num_sectors,
            update_mask=update_mask,
        )
        env._contact_memory_last_update_step[update_mask] = current_step[update_mask]
    return _world_memory_to_body_memory(
        env._contact_memory_world,
        robot_yaw_local=robot_yaw_local,
        num_sectors=num_sectors,
    )


def get_imu_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return compact IMU state: lin acc, ang vel, roll, pitch."""
    imu = env.scene.sensors.get("imu", None)
    if imu is None:
        return torch.zeros((env.num_envs, 8), device=env.device)

    data = imu.data
    quat = data.quat_w
    roll, pitch, _yaw = math_utils.euler_xyz_from_quat(quat)
    return torch.cat(
        [
            data.lin_acc_b,
            data.ang_vel_b,
            roll.unsqueeze(-1),
            pitch.unsqueeze(-1),
        ],
        dim=-1,
    )


def _local_nav_state(env: ManagerBasedRLEnv):
    if not hasattr(env, "_local_nav_path_ids"):
        _ensure_local_nav_buffers(env)
    return (
        env._local_nav_path_library,
        env._local_nav_path_ids,
        env._local_nav_start_idx,
        env._local_nav_progress_idx,
        env._local_nav_goal_idx,
        env._local_nav_prev_s,
        env._local_nav_start_s,
        env._local_nav_no_progress_steps,
    )


def _robot_xy_local(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot_pos_w = env.scene["robot"].data.root_pos_w[:, :2]
    if hasattr(env.scene, "env_origins"):
        return robot_pos_w - env.scene.env_origins[:, :2]
    return robot_pos_w


def _get_obstacle_positions_local(env: ManagerBasedEnv, collection_name: str = "obstacles") -> torch.Tensor:
    """Best-effort runtime obstacle centers in env-local XY, shape (N, M, 2)."""
    device = env.device
    try:
        asset = env.scene[collection_name]
    except Exception:
        return torch.empty((env.num_envs, 0, 2), device=device)

    data = getattr(asset, "data", None)
    if data is not None:
        for key in ("object_pos_w", "root_pos_w", "object_positions_w"):
            if hasattr(data, key):
                pos_w = getattr(data, key)
                if isinstance(pos_w, torch.Tensor):
                    if pos_w.ndim == 3:
                        pos_xy = pos_w[:, :, :2]
                    elif pos_w.ndim == 2:
                        pos_xy = pos_w[:, :2].unsqueeze(0).expand(env.num_envs, -1, -1)
                    else:
                        continue
                    if hasattr(env.scene, "env_origins"):
                        pos_xy = pos_xy - env.scene.env_origins[:, None, :2]
                    return pos_xy.contiguous()
    return torch.empty((env.num_envs, 0, 2), device=device)


def _visualize_local_reference_progress(
    env: ManagerBasedRLEnv,
    num_points: int = 16,
    ds: float = 0.5,
    vis_update_interval: int = 1,
):
    """Refresh per-env local reference colors during stepping."""
    if env.num_envs == 0:
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
    if not hasattr(env, "_local_nav_ref_vis_counter"):
        env._local_nav_ref_vis_counter = 0

    env._local_nav_ref_vis_counter += 1
    if max(1, vis_update_interval) > 1 and env._local_nav_ref_vis_counter % vis_update_interval != 0:
        return

    (
        path_library,
        path_ids,
        _start_idx,
        progress_idx,
        _goal_idx,
        _prev_s,
        _start_s,
        _no_progress_steps,
    ) = _local_nav_state(env)

    points = []
    marker_ids = []
    for env_id in range(env.num_envs):
        path = path_library[int(path_ids[env_id].item())]
        current_idx = int(progress_idx[env_id].item())
        lookbehind = max(1, num_points // 3)
        origin_xy = env.scene.env_origins[env_id, :2] if hasattr(env.scene, "env_origins") else torch.zeros(2, device=env.device)

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


def compute_local_nav_tracking(env: ManagerBasedRLEnv, search_radius: int = 20, update_progress: bool = False):
    """Compute tracking quantities against the current sequential waypoint target."""
    (
        path_library,
        path_ids,
        start_idx,
        progress_idx,
        goal_idx,
        _prev_s,
        start_s,
        _no_progress_steps,
    ) = _local_nav_state(env)
    robot = env.scene["robot"]
    robot_pos = _robot_xy_local(env)
    robot_yaw = robot.data.heading_w
    reach_thresh = float(getattr(env, "_local_nav_waypoint_reach_thresh", 0.45))

    current_idx = progress_idx.clone()
    reached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    s_val = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    lateral_error = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    heading_error = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    curvature = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    target_distance = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    for env_id in range(env.num_envs):
        path = path_library[int(path_ids[env_id].item())]
        goal = int(goal_idx[env_id].item())
        target = int(current_idx[env_id].item())
        target = max(int(start_idx[env_id].item()), min(target, goal))
        prev_idx = max(int(start_idx[env_id].item()), target - 1)

        prev_xy = path["xy"][prev_idx]
        ref_xy = path["xy"][target]
        ref_yaw = path["yaw"][target]
        seg = ref_xy - prev_xy
        seg_len = torch.norm(seg)
        if seg_len > 1e-6:
            tangent = seg / seg_len
            normal = torch.stack([-tangent[1], tangent[0]])
            along = torch.dot(robot_pos[env_id] - prev_xy, tangent)
            along_clamped = torch.clamp(along, min=0.0, max=seg_len)
            s_val[env_id] = path["s"][prev_idx] + along_clamped
            lateral_error[env_id] = torch.dot(robot_pos[env_id] - prev_xy, normal)
        else:
            normal = torch.tensor([-torch.sin(ref_yaw), torch.cos(ref_yaw)], device=env.device)
            s_val[env_id] = path["s"][target]
            lateral_error[env_id] = torch.dot(robot_pos[env_id] - ref_xy, normal)

        heading_error[env_id] = math_utils.wrap_to_pi(robot_yaw[env_id] - ref_yaw)
        curvature[env_id] = path["curvature"][target]
        target_distance[env_id] = torch.norm(robot_pos[env_id] - ref_xy)
        reached[env_id] = target_distance[env_id] <= reach_thresh

    if update_progress:
        can_advance = reached & (current_idx < goal_idx)
        env._local_nav_progress_idx[:] = torch.where(can_advance, current_idx + 1, current_idx)
        current_idx = env._local_nav_progress_idx.clone()
        for env_id in torch.nonzero(can_advance, as_tuple=False).squeeze(-1).tolist():
            path = path_library[int(path_ids[env_id].item())]
            target = int(current_idx[env_id].item())
            prev_idx = max(int(start_idx[env_id].item()), target - 1)
            prev_xy = path["xy"][prev_idx]
            ref_xy = path["xy"][target]
            ref_yaw = path["yaw"][target]
            seg = ref_xy - prev_xy
            seg_len = torch.norm(seg)
            if seg_len > 1e-6:
                tangent = seg / seg_len
                normal = torch.stack([-tangent[1], tangent[0]])
                along = torch.dot(robot_pos[env_id] - prev_xy, tangent)
                along_clamped = torch.clamp(along, min=0.0, max=seg_len)
                s_val[env_id] = path["s"][prev_idx] + along_clamped
                lateral_error[env_id] = torch.dot(robot_pos[env_id] - prev_xy, normal)
            else:
                normal = torch.tensor([-torch.sin(ref_yaw), torch.cos(ref_yaw)], device=env.device)
                s_val[env_id] = path["s"][target]
                lateral_error[env_id] = torch.dot(robot_pos[env_id] - ref_xy, normal)
            heading_error[env_id] = math_utils.wrap_to_pi(robot_yaw[env_id] - ref_yaw)
            curvature[env_id] = path["curvature"][target]
            target_distance[env_id] = torch.norm(robot_pos[env_id] - ref_xy)
        if getattr(env, "_local_nav_debug_vis", False):
            _visualize_local_reference_progress(
                env,
                num_points=getattr(env, "_local_nav_debug_vis_num_points", 16),
                ds=getattr(env, "_local_nav_debug_vis_ds", 0.5),
                vis_update_interval=getattr(env, "_local_nav_debug_vis_interval", 1),
            )

    progress_gain = s_val - start_s
    goal_s = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    for env_id in range(env.num_envs):
        path = path_library[int(path_ids[env_id].item())]
        goal_s[env_id] = path["s"][int(goal_idx[env_id].item())]
    remaining_progress = torch.clamp(goal_s - s_val, min=0.0)
    return {
        "index": current_idx,
        "s": s_val,
        "lateral_error": lateral_error,
        "heading_error": heading_error,
        "progress_gain": progress_gain,
        "remaining_progress": remaining_progress,
        "target_distance": target_distance,
        "curvature": curvature,
        "target_reached": reached.float(),
    }


def get_local_tracking_state(env: ManagerBasedRLEnv) -> torch.Tensor:
    track = compute_local_nav_tracking(env, update_progress=False)
    return torch.stack(
        [
            track["lateral_error"],
            track["heading_error"],
            track["progress_gain"],
            track["target_distance"],
            track["curvature"],
        ],
        dim=-1,
    )


def get_future_waypoint_preview(
    env: ManagerBasedRLEnv,
    num_waypoints: int = 3,
    activate_next_after_dist: float = 0.8,
) -> torch.Tensor:
    """Return sequential waypoint preview in ego frame.

    The first preview point is the current target waypoint, followed by the next
    waypoints in sequence (clamped at the goal index). Each waypoint contributes:
    (x_ego, y_ego, yaw_error, distance).
    """
    (
        path_library,
        path_ids,
        _start_idx,
        progress_idx,
        goal_idx,
        _prev_s,
        _start_s,
        _no_progress_steps,
    ) = _local_nav_state(env)
    robot = env.scene["robot"]
    robot_pos = _robot_xy_local(env)
    robot_yaw = robot.data.heading_w

    previews = []
    for env_id in range(env.num_envs):
        path = path_library[int(path_ids[env_id].item())]
        idx0 = int(progress_idx[env_id].item())
        idxg = int(goal_idx[env_id].item())
        current_wp_xy = path["xy"][idx0]
        dist_to_current = torch.norm(current_wp_xy - robot_pos[env_id])
        cy = torch.cos(robot_yaw[env_id])
        sy = torch.sin(robot_yaw[env_id])
        feat = []
        for k in range(max(1, int(num_waypoints))):
            if k > 0 and dist_to_current > float(activate_next_after_dist):
                # Keep feature dimensionality unchanged while preventing early
                # over-steering toward farther waypoints.
                wp_idx = idx0
            else:
                wp_idx = min(idx0 + k, idxg)
            wp_xy = path["xy"][wp_idx]
            wp_yaw = path["yaw"][wp_idx]
            delta = wp_xy - robot_pos[env_id]
            x_ego = cy * delta[0] + sy * delta[1]
            y_ego = -sy * delta[0] + cy * delta[1]
            yaw_err = math_utils.wrap_to_pi(wp_yaw - robot_yaw[env_id])
            dist = torch.norm(delta)
            feat.extend([x_ego, y_ego, yaw_err, dist])
        previews.append(torch.stack(feat))
    return torch.stack(previews, dim=0)


def get_local_reference_window(
    env: ManagerBasedRLEnv,
    num_points: int = 8,
    ds: float = 0.5,
    debug_vis: bool = False,
    vis_update_interval: int = 1,
) -> torch.Tensor:
    """Return future preview points in ego frame as flattened (x, y, yaw_err)."""
    (
        path_library,
        path_ids,
        start_idx,
        progress_idx,
        _goal_idx,
        _prev_s,
        _start_s,
        _no_progress_steps,
    ) = _local_nav_state(env)
    robot = env.scene["robot"]
    robot_pos = _robot_xy_local(env)
    robot_yaw = robot.data.heading_w

    windows = []
    for env_id in range(env.num_envs):
        path = path_library[int(path_ids[env_id].item())]
        idx = int(progress_idx[env_id].item())
        ref_s0 = float(path["s"][max(int(start_idx[env_id].item()), idx)].item())
        samples = []
        cos_yaw = torch.cos(robot_yaw[env_id])
        sin_yaw = torch.sin(robot_yaw[env_id])
        for step_idx in range(1, num_points + 1):
            target_s = ref_s0 + step_idx * ds
            ref_idx = int(torch.argmin(torch.abs(path["s"] - target_s)).item())
            ref_xy = path["xy"][ref_idx]
            delta = ref_xy - robot_pos[env_id]
            x_ego = cos_yaw * delta[0] + sin_yaw * delta[1]
            y_ego = -sin_yaw * delta[0] + cos_yaw * delta[1]
            yaw_err = math_utils.wrap_to_pi(path["yaw"][ref_idx] - robot_yaw[env_id])
            samples.extend([x_ego, y_ego, yaw_err])
        windows.append(torch.stack(samples))

    if debug_vis:
        _visualize_local_reference_progress(
            env,
            num_points=max(num_points, 16),
            ds=ds,
            vis_update_interval=vis_update_interval,
        )
    return torch.stack(windows, dim=0)


def get_derived_proprio_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Rule-based auxiliary features recommended by the proprio-only design."""
    _ensure_local_nav_buffers(env)
    robot = env.scene["robot"]
    track = compute_local_nav_tracking(env, update_progress=False)

    lin_vel_b = robot.data.root_lin_vel_b[:, :2]
    forward_speed = lin_vel_b[:, 0].abs()
    wheel_speed = robot.data.joint_vel[:, :4].abs().mean(dim=1)
    slip_ratio = (wheel_speed - forward_speed).abs() / (forward_speed + 0.1)

    progress_delta = env._local_nav_last_delta_s

    action = env.action_manager.action if hasattr(env, "action_manager") else torch.zeros((env.num_envs, 2), device=env.device)
    if not hasattr(env, "_local_nav_prev_action_obs"):
        env._local_nav_prev_action_obs = action.detach().clone()
    action_change = torch.norm(action - env._local_nav_prev_action_obs, dim=-1)
    env._local_nav_prev_action_obs = action.detach().clone()

    if not hasattr(env, "_local_nav_prev_heading_error"):
        env._local_nav_prev_heading_error = track["heading_error"].detach().clone()
    yaw_error_rate = track["heading_error"] - env._local_nav_prev_heading_error
    env._local_nav_prev_heading_error = track["heading_error"].detach().clone()

    imu = env.scene.sensors.get("imu", None)
    if imu is not None:
        vibration_level = torch.norm(imu.data.lin_acc_b, dim=-1)
    else:
        vibration_level = torch.zeros(env.num_envs, device=env.device)

    stuck_score = ((progress_delta < 0.002) & (action.abs().sum(dim=-1) > 0.2)).float()

    return torch.stack(
        [
            slip_ratio,
            progress_delta,
            action_change,
            yaw_error_rate,
            vibration_level,
            stuck_score,
        ],
        dim=-1,
    )


def get_next_proprio_target(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Compact proprio target for next-step prediction supervision."""
    robot = env.scene["robot"]
    base_lin_vel = robot.data.root_lin_vel_b[:, :2]
    base_ang_vel = robot.data.root_ang_vel_b[:, 2:3]
    wheel_vel = robot.data.joint_vel[:, :4]
    imu_state = get_imu_state(env)
    contact = get_contact_sensor_feedback(env)
    return torch.cat([base_lin_vel, base_ang_vel, wheel_vel, imu_state, contact], dim=-1)


def get_stuck_label(
    env: ManagerBasedRLEnv,
    progress_threshold: float = 0.002,
    action_threshold: float = 0.2,
    patience: int = 20,
) -> torch.Tensor:
    """Env-side stuck label based on no-progress counter and active control."""
    _ensure_local_nav_buffers(env)
    action = env.action_manager.action if hasattr(env, "action_manager") else torch.zeros((env.num_envs, 2), device=env.device)
    active_action = action.abs().sum(dim=-1) > action_threshold
    little_progress = env._local_nav_last_delta_s < progress_threshold
    persistent = env._local_nav_no_progress_steps >= patience
    stuck = persistent | (little_progress & active_action & (env._local_nav_episode_steps > 10))
    return stuck.float().unsqueeze(-1)


def get_mode_label(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Env-side supervision for track / recover / rejoin."""
    track = compute_local_nav_tracking(env, update_progress=False)
    lateral = track["lateral_error"].abs()
    heading = track["heading_error"].abs()
    stuck = get_stuck_label(env).squeeze(-1) > 0.5
    track_mode = (~stuck) & (lateral < 0.25) & (heading < 0.25)
    recover_mode = stuck
    rejoin_mode = ~(track_mode | recover_mode)
    return torch.stack(
        [track_mode.float(), recover_mode.float(), rejoin_mode.float()],
        dim=-1,
    )


def get_interaction_label(env: ManagerBasedRLEnv, num_dirs: int = 12) -> torch.Tensor:
    """Env-side target with task semantics: passability / trap risk / recovery gain."""
    device = env.device
    window = get_local_reference_window(env, num_points=8, ds=0.5, debug_vis=False)
    pts = window.reshape(env.num_envs, -1, 3)
    ref_xy = pts[..., :2]
    ref_dist = torch.norm(ref_xy, dim=-1).clamp(min=1e-6)
    ref_angle = torch.atan2(ref_xy[..., 1], ref_xy[..., 0])
    ref_bins = ((ref_angle + torch.pi) / (2 * torch.pi) * num_dirs).floor().long().clamp(0, num_dirs - 1)
    ref_score = torch.exp(-ref_dist / 2.0)

    desired = torch.zeros((env.num_envs, num_dirs), device=device)
    for idx in range(num_dirs):
        mask = ref_bins == idx
        desired[:, idx] = torch.where(mask, ref_score, torch.zeros_like(ref_score)).amax(dim=1)

    obstacles = _get_obstacle_positions_local(env)
    obstacle_presence = torch.zeros((env.num_envs, num_dirs), device=device)
    if obstacles.numel() > 0:
        robot_xy = _robot_xy_local(env)
        rel = obstacles - robot_xy[:, None, :]
        dist = torch.norm(rel, dim=-1).clamp(min=1e-6)
        angle = torch.atan2(rel[..., 1], rel[..., 0])
        bins = ((angle + torch.pi) / (2 * torch.pi) * num_dirs).floor().long().clamp(0, num_dirs - 1)
        closeness = torch.exp(-dist / 1.0)
        for idx in range(num_dirs):
            mask = bins == idx
            obstacle_presence[:, idx] = torch.where(mask, closeness, torch.zeros_like(closeness)).amax(dim=1)

    desired_left = torch.roll(desired, shifts=1, dims=1)
    desired_right = torch.roll(desired, shifts=-1, dims=1)
    desired_spread = torch.maximum(desired, torch.maximum(desired_left, desired_right))

    free_space = 1.0 - obstacle_presence
    passability = torch.clamp(0.15 + 0.85 * desired_spread * free_space, 0.0, 1.0)

    stuck = get_stuck_label(env)
    left_block = torch.roll(obstacle_presence, shifts=1, dims=1)
    right_block = torch.roll(obstacle_presence, shifts=-1, dims=1)
    corridor_block = torch.minimum(left_block + right_block, torch.ones_like(obstacle_presence))
    trap_risk = torch.clamp(0.6 * obstacle_presence + 0.4 * corridor_block + 0.5 * stuck, 0.0, 1.0)

    alternative_free = torch.clamp(free_space - desired_spread, 0.0, 1.0)
    recovery_gain = torch.clamp(stuck * (0.7 * alternative_free + 0.3 * free_space), 0.0, 1.0)
    return torch.cat([passability, trap_risk, recovery_gain], dim=-1)
