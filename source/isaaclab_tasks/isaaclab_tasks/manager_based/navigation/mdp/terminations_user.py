import torch
from isaaclab.managers import SceneEntityCfg

def _get_env_origins_xy(env) -> torch.Tensor:
    """Best-effort fetch of per-env origin offsets (xy). Falls back to zeros."""
    device = env.device if hasattr(env, "device") else None
    num_envs = env.num_envs if hasattr(env, "num_envs") else None

    origins = None
    if hasattr(env, "scene") and hasattr(env.scene, "env_origins"):
        origins = env.scene.env_origins  # (N, 3)
    elif hasattr(env, "env_origins"):
        origins = env.env_origins

    if origins is None:
        if num_envs is None:
            # last resort: infer from robot state later
            raise AttributeError("Cannot find env origins on env/scene.")
        return torch.zeros((num_envs, 2), device=device)

    return origins[:, :2]


def out_of_bounds(
    env,
    x_min: float = -10.0,
    x_max: float = 120.0,
    y_min: float = -20.0,
    y_max: float = 20.0,
    use_env_frame: bool = True,
):
    robot = env.scene["robot"]
    p_w = robot.data.root_pos_w[:, :2]  # (N,2)

    if use_env_frame:
        origins_xy = _get_env_origins_xy(env)  # (N,2)
        p = p_w - origins_xy
    else:
        p = p_w

    x, y = p[:, 0], p[:, 1]
    return (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)


def is_flipped(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), up_z_threshold: float = 0.2) -> torch.Tensor:
    """Terminate when robot is flipped/rolled over.

    Robust rule: compute base "up" vector in world frame and terminate if its z component is too small.
      - upright: up_z ~ 1
      - on side: up_z ~ 0
      - upside down: up_z ~ -1
    """
    asset = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w  # expected (w, x, y, z)
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # up_z for rotating local +Z into world frame: up_z = 1 - 2(x^2 + y^2)
    up_z = 1.0 - 2.0 * (x * x + y * y)

    # flipped if base up has insufficient world-z component
    flipped = up_z < float(up_z_threshold)
    return flipped.to(torch.bool)