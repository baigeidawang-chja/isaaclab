from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .medium_state import compute_medium_state


def apply_hydrodynamics(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    medium_mode: str = "linear",
    use_env_origin: bool = True,
    x_start: float = 1.0,
    x_end: float = 3.0,
    terrain_x_min: float = -10.0,
    terrain_x_max: float = 14.0,
    stage_len: float = 1.5,
    cycle_len: float = 6.0,
    high_z: float = 1.0,
    low_z: float = 0.25,
    slope_angle_deg: float = 30.0,
    wheel_threshold: float = 0.55,
    wheel_sharpness: float = 0.15,
    thruster_threshold: float = 0.35,
    thruster_sharpness: float = 0.15,
    drag_min: float = 0.0,
    drag_max: float = 1.0,
    water_z: float = 0.65,
    wheel_submersion_depth: float = 0.15,
    thruster_submersion_depth: float = 0.20,
    body_submersion_depth: float = 0.25,
    front_x: float = 0.23,
    rear_x: float = -0.23,
    left_y: float = 0.14,
    right_y: float = -0.14,
    wheel_z: float = -0.16,
    thruster_x: float = -0.28,
    thruster_y: float = 0.0,
    thruster_z: float = -0.06,
    body_z: float = 0.0,
    contact_force_ref: float = 25.0,
    contact_force_threshold: float = 1e-3,
    use_binary_wheel_contact: bool = False,
    linear_drag: tuple[float, float, float] = (8.0, 14.0, 2.0),
    quadratic_drag: tuple[float, float, float] = (2.0, 4.0, 0.5),
    yaw_damping: float = 1.5,
    thruster_gain: float = 12.0,
    thruster_force_scale: float = 1.0,
    debug: bool = False,
    debug_every: int = 100,
):
    """Apply simple body-frame water drag, yaw damping, and optional thruster force.

    This is a deliberately low-order transition model for fast iteration:
    no buoyancy, no fluid surface mesh, and no CFD. Forces are written into
    Isaac Lab's external wrench buffer and applied by the simulator step.
    """
    asset = env.scene[asset_cfg.name]
    device = asset.device
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=device, dtype=torch.long)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, device=device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    state = compute_medium_state(
        env,
        asset_cfg=SceneEntityCfg(asset_cfg.name),
        medium_mode=medium_mode,
        use_env_origin=use_env_origin,
        x_start=x_start,
        x_end=x_end,
        terrain_x_min=terrain_x_min,
        terrain_x_max=terrain_x_max,
        stage_len=stage_len,
        cycle_len=cycle_len,
        high_z=high_z,
        low_z=low_z,
        slope_angle_deg=slope_angle_deg,
        wheel_threshold=wheel_threshold,
        wheel_sharpness=wheel_sharpness,
        thruster_threshold=thruster_threshold,
        thruster_sharpness=thruster_sharpness,
        drag_min=drag_min,
        drag_max=drag_max,
        water_z=water_z,
        wheel_submersion_depth=wheel_submersion_depth,
        thruster_submersion_depth=thruster_submersion_depth,
        body_submersion_depth=body_submersion_depth,
        front_x=front_x,
        rear_x=rear_x,
        left_y=left_y,
        right_y=right_y,
        wheel_z=wheel_z,
        thruster_x=thruster_x,
        thruster_y=thruster_y,
        thruster_z=thruster_z,
        body_z=body_z,
        contact_force_ref=contact_force_ref,
        contact_force_threshold=contact_force_threshold,
        use_binary_wheel_contact=use_binary_wheel_contact,
    )
    drag_scale = state["drag_scale"][env_ids].unsqueeze(-1)
    eta_thruster = state["eta_thruster"][env_ids]

    vel_b = asset.data.root_lin_vel_b[env_ids]
    yaw_rate = asset.data.root_ang_vel_b[env_ids, 2]
    lin = torch.tensor(linear_drag, device=device, dtype=torch.float32).view(1, 3)
    quad = torch.tensor(quadratic_drag, device=device, dtype=torch.float32).view(1, 3)
    force_b = -drag_scale * (lin * vel_b + quad * vel_b.abs() * vel_b)

    thruster_cmd = getattr(env, "_thruster_cmd", None)
    if thruster_cmd is None:
        thruster_u = torch.zeros(len(env_ids), device=device, dtype=torch.float32)
    else:
        thruster_u = thruster_cmd[env_ids].to(device=device, dtype=torch.float32).reshape(-1)
    thrust = float(thruster_force_scale) * eta_thruster * float(thruster_gain) * thruster_u * thruster_u.abs()
    force_b[:, 0] = force_b[:, 0] + thrust
    force_norm = torch.linalg.norm(force_b, dim=-1)
    if not hasattr(env, "_hydro_force_norm"):
        env._hydro_force_norm = torch.zeros(env.num_envs, device=device, dtype=torch.float32)
    env._hydro_force_norm[env_ids] = force_norm

    torque_b = torch.zeros_like(force_b)
    torque_b[:, 2] = -drag_scale.squeeze(-1) * float(yaw_damping) * yaw_rate

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        num_bodies = asset.num_bodies
    elif isinstance(body_ids, int):
        num_bodies = 1
    else:
        num_bodies = len(body_ids)
    forces = force_b[:, None, :].repeat(1, num_bodies, 1)
    torques = torque_b[:, None, :].repeat(1, num_bodies, 1)
    asset.set_external_force_and_torque(forces, torques, body_ids=body_ids, env_ids=env_ids)

    if debug:
        count = int(getattr(env, "_hydro_debug_count", 0)) + 1
        env._hydro_debug_count = count
        every = max(1, int(debug_every))
        if count % every == 0 and len(env_ids) > 0:
            local_idx = 0
            env_id = int(env_ids[local_idx].item())
            print(
                "[HYDRO DEBUG] "
                f"step={count} env={env_id} "
                f"lambda={state['lambda_medium'][env_id].detach().item():.3f} "
                f"drag_scale={state['drag_scale'][env_id].detach().item():.3f} "
                f"thruster_cmd={thruster_u[local_idx].detach().item():.3f} "
                f"force_norm={force_norm[local_idx].detach().item():.3f}"
            )
