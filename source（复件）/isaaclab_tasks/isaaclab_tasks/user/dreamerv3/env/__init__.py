# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym


##
# Register Gym environments.
##

gym.register(
    id="Isaac-Navigation-Car-Dreamer-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarRoughEnvCfg",
        "dreamer_cfg_entry_point": f"{__name__}.dreamer_agent_cfg:DreamerAgentCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarRoughEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Simple-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarSimpleEnvCfg",
        "dreamer_cfg_entry_point": f"{__name__}.dreamer_agent_cfg:DreamerAgentCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Simple-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarSimpleEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Recover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarRecoverEnvCfg",
        "dreamer_cfg_entry_point": f"{__name__}.dreamer_agent_cfg:DreamerAgentCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Recover-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarRecoverEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Amphibious-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarAmphibiousEnvCfg",
        "dreamer_cfg_entry_point": f"{__name__}.dreamer_agent_cfg:DreamerAgentCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Amphibious-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarAmphibiousEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Waterland-Amphibious-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarWaterlandAmphibiousEnvCfg",
        "dreamer_cfg_entry_point": f"{__name__}.dreamer_agent_cfg:DreamerAgentCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-Waterland-Amphibious-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarWaterlandAmphibiousEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-AmphibiousTerrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarWaterlandAmphibiousEnvCfg",
        "dreamer_cfg_entry_point": f"{__name__}.dreamer_agent_cfg:DreamerAgentCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-AmphibiousTerrain-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.car_env_cfg:MyCarWaterlandAmphibiousEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-TwoPointRecover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.two_point_recover_env_cfg:MyCarTwoPointRecoverEnvCfg",
        "dreamer_cfg_entry_point": f"{__name__}.dreamer_agent_cfg:DreamerAgentCfg",
    },
)

gym.register(
    id="Isaac-Navigation-Car-Dreamer-TwoPointRecover-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.two_point_recover_env_cfg:MyCarTwoPointRecoverEnvCfg_PLAY",
    },
)

gym.register(
    id="Isaac-Dreamer-Smoke-Cartpole-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.smoke_test_env_cfg:DreamerSmokeCartpoleEnvCfg",
        "dreamer_cfg_entry_point": f"{__name__}.dreamer_agent_cfg:DreamerAgentCfg",
    },
)

gym.register(
    id="Isaac-Dreamer-Smoke-Cartpole-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.smoke_test_env_cfg:DreamerSmokeCartpoleEnvCfg_PLAY",
    },
)
