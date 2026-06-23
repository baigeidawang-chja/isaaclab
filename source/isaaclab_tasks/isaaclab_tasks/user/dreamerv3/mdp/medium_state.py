from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def compute_medium_state(
    env,
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
) -> dict[str, torch.Tensor]:
    """Compute low-order amphibious transition state from robot base x position.

    The returned tensors are privileged simulation state. They are intended for
    labels, dynamics modulation, and diagnostics, not as policy observations.
    """
    asset = env.scene[asset_cfg.name]
    x = asset.data.root_pos_w[:, 0]
    env_origins = getattr(env.scene, "env_origins", None)
    if use_env_origin and env_origins is not None:
        x = x - env_origins[:, 0].to(device=x.device)
    if medium_mode == "linear":
        denom = max(float(x_end) - float(x_start), 1e-6)
        lambda_medium = torch.clamp((x - float(x_start)) / denom, 0.0, 1.0)
        terrain_height = torch.zeros_like(lambda_medium)
        slope_sin = torch.zeros_like(lambda_medium)
        terrain_phase = lambda_medium
    elif medium_mode == "periodic_waterland":
        stage_len = max(float(stage_len), 1e-6)
        cycle_len = max(float(cycle_len), 4.0 * stage_len)
        phase = torch.remainder(x - float(terrain_x_min), cycle_len)
        lambda_medium = torch.zeros_like(phase)
        terrain_height = torch.full_like(phase, float(high_z))
        slope_angle = torch.zeros_like(phase)

        down = (phase >= stage_len) & (phase < 2.0 * stage_len)
        low = (phase >= 2.0 * stage_len) & (phase < 3.0 * stage_len)
        up = phase >= 3.0 * stage_len
        r_down = (phase - stage_len) / stage_len
        r_up = (phase - 3.0 * stage_len) / stage_len

        lambda_medium = torch.where(down, r_down, lambda_medium)
        lambda_medium = torch.where(low, torch.ones_like(lambda_medium), lambda_medium)
        lambda_medium = torch.where(up, 1.0 - r_up, lambda_medium)
        lambda_medium = torch.clamp(lambda_medium, 0.0, 1.0)

        terrain_height = torch.where(
            down,
            float(high_z) + (float(low_z) - float(high_z)) * r_down,
            terrain_height,
        )
        terrain_height = torch.where(low, torch.full_like(terrain_height, float(low_z)), terrain_height)
        terrain_height = torch.where(
            up,
            float(low_z) + (float(high_z) - float(low_z)) * r_up,
            terrain_height,
        )
        slope_rad = torch.deg2rad(torch.tensor(float(slope_angle_deg), device=phase.device, dtype=phase.dtype))
        slope_angle = torch.where(down, -slope_rad.expand_as(slope_angle), slope_angle)
        slope_angle = torch.where(up, slope_rad.expand_as(slope_angle), slope_angle)
        slope_sin = torch.sin(slope_angle)
        terrain_phase = phase / cycle_len
    else:
        raise ValueError(f"Unsupported medium_mode: {medium_mode}")

    eta_wheel = torch.sigmoid((float(wheel_threshold) - lambda_medium) / max(float(wheel_sharpness), 1e-6))
    eta_thruster = torch.sigmoid(
        (lambda_medium - float(thruster_threshold)) / max(float(thruster_sharpness), 1e-6)
    )
    drag_scale = float(drag_min) + (float(drag_max) - float(drag_min)) * lambda_medium

    return {
        "lambda_medium": lambda_medium,
        "eta_wheel": eta_wheel,
        "eta_thruster": eta_thruster,
        "drag_scale": drag_scale,
        "terrain_height": terrain_height,
        "slope_sin": slope_sin,
        "terrain_phase": terrain_phase,
    }


def medium_state_label(
    env,
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
) -> torch.Tensor:
    """Return privileged label [lambda, eta_wheel, eta_thruster, drag_scale]."""
    state = compute_medium_state(
        env,
        asset_cfg=asset_cfg,
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
    )
    return torch.stack(
        [
            state["lambda_medium"],
            state["eta_wheel"],
            state["eta_thruster"],
            state["drag_scale"],
            state["slope_sin"],
            state["terrain_height"],
            state["terrain_phase"],
        ],
        dim=-1,
    )
