from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def is_flipped(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), up_z_threshold: float = 0.25) -> torch.Tensor:
    quat = env.scene[asset_cfg.name].data.root_quat_w
    x = quat[:, 1]
    y = quat[:, 2]
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return up_z < float(up_z_threshold)


def out_of_bounds(
    env,
    x_min: float = -20.0,
    x_max: float = 20.0,
    y_min: float = -10.0,
    y_max: float = 10.0,
) -> torch.Tensor:
    pos = env.scene["robot"].data.root_pos_w[:, :2]
    if hasattr(env.scene, "env_origins"):
        pos = pos - env.scene.env_origins[:, :2]
    return (pos[:, 0] < float(x_min)) | (pos[:, 0] > float(x_max)) | (pos[:, 1] < float(y_min)) | (pos[:, 1] > float(y_max))
