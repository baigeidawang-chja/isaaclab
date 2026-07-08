"""Car4WD user project task registration."""

import gymnasium as gym

from . import agents


gym.register(
    id="Isaac-Project-CommandFollowing-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.command_following_env_cfg:CommandFollowingEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Project-CommandFollowing-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.command_following_env_cfg:CommandFollowingEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)


gym.register(
    id="Isaac-Project-FailureAwareExecution-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.failure_aware_execution_env_cfg:FailureAwareExecutionEnvCfg",
    },
)

gym.register(
    id="Isaac-Project-FailureAwareExecution-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.failure_aware_execution_env_cfg:FailureAwareExecutionEnvCfg_PLAY",
    },
)
