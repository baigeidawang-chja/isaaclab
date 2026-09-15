"""Car4WD task registrations for traction protective control research."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-TractionProtect-CommandFollow-Car4WD-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.command_follow_env_cfg:CommandFollowEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-TractionProtect-CommandFollow-Car4WD-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.command_follow_env_cfg:CommandFollowEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
