# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg


@configclass
class DreamerSmokeCartpoleEnvCfg(CartpoleEnvCfg):
    """Small, easy-to-learn environment for validating the Dreamer training stack."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.sim.device = "cuda:0"
        self.scene.num_envs = 256
        self.scene.env_spacing = 4.0
        self.episode_length_s = 5.0
        self.observations.policy.enable_corruption = False


@configclass
class DreamerSmokeCartpoleEnvCfg_PLAY(DreamerSmokeCartpoleEnvCfg):
    """Playable variant with fewer environments for quick visual checks."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 4.0
