from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

from . import recovery_rewards
from .terminations_user import is_flipped


def blocked_recovery_success(env, success_distance: float = 1.15) -> torch.Tensor:
    metrics = recovery_rewards.update_recovery_metrics(env)
    return metrics["progress"] >= float(success_distance)


def blocked_recovery_failure(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_retreat: float = 0.65,
    max_no_progress_steps: int = 120,
    min_episode_steps: int = 20,
    flip_up_z_threshold: float = 0.2,
) -> torch.Tensor:
    metrics = recovery_rewards.update_recovery_metrics(env)
    retreat_fail = metrics["progress"] < -float(max_retreat)
    no_progress = env._blocked_recovery_no_progress_steps >= int(max_no_progress_steps)
    if hasattr(env, "episode_length_buf"):
        no_progress = no_progress & (env.episode_length_buf >= int(min_episode_steps))
    flip_fail = is_flipped(env, asset_cfg=asset_cfg, up_z_threshold=flip_up_z_threshold)
    return retreat_fail | no_progress | flip_fail
