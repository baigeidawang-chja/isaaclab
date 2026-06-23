from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from .medium_state import compute_medium_state


def apply_hydrodynamics(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_start: float = 1.0,
    x_end: float = 3.0,
    linear_drag: tuple[float, float, float] = (8.0, 14.0, 2.0),
    quadratic_drag: tuple[float, float, float] = (2.0, 4.0, 0.5),
    yaw_damping: float = 1.5,
    thruster_gain: float = 12.0,
    thruster_force_scale: float = 1.0,
    **medium_kwargs,
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

    state = compute_medium_state(env, asset_cfg=SceneEntityCfg(asset_cfg.name), x_start=x_start, x_end=x_end, **medium_kwargs)
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
