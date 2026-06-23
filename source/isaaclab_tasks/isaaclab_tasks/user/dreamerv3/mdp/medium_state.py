from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def compute_medium_state(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    x_start: float = 1.0,
    x_end: float = 3.0,
    wheel_threshold: float = 0.55,
    wheel_sharpness: float = 0.12,
    thruster_threshold: float = 0.35,
    thruster_sharpness: float = 0.12,
    drag_min: float = 0.0,
    drag_max: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Compute low-order amphibious transition state from robot base x position.

    The returned tensors are privileged simulation state. They are intended for
    labels, dynamics modulation, and diagnostics, not as policy observations.
    """
    asset = env.scene[asset_cfg.name]
    x = asset.data.root_pos_w[:, 0]
    denom = max(float(x_end) - float(x_start), 1e-6)
    lambda_medium = torch.clamp((x - float(x_start)) / denom, 0.0, 1.0)

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
    }


def medium_state_label(env, **kwargs) -> torch.Tensor:
    """Return privileged label [lambda, eta_wheel, eta_thruster, drag_scale]."""
    state = compute_medium_state(env, **kwargs)
    return torch.stack(
        [
            state["lambda_medium"],
            state["eta_wheel"],
            state["eta_thruster"],
            state["drag_scale"],
        ],
        dim=-1,
    )
