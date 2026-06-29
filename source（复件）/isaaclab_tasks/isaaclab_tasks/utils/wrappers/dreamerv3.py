
"""IsaacLab 环境到 DreamerV3 Agent 的包装器。"""

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IsaacLabDreamerWrapper:
    """
    将 IsaacLab 的 gymnasium 向量化环境包装为 DreamerV3 Agent 所需的接口。

    DreamerV3 Agent 需要:
    - obs_space: dict of SimpleSpace
    - act_space: dict of SimpleSpace
    - obs 中包含 is_first, is_last, is_terminal, reward
    """

    def __init__(self, env, device="cuda:0"):
        self._env = env
        self._device = torch.device(device)
        self._step_count = 0
        self._is_first = True

        # 构建 obs_space / act_space
        self.obs_space = self._build_obs_space()
        self.act_space = self._build_act_space()

    def _build_obs_space(self):
        from isaaclab_tasks.user.dreamerv3.agent import SimpleSpace

        result = {}
        raw = self._env.observation_space
        if isinstance(raw, spaces.Dict):
            for k, v in raw.spaces.items():
                result[k] = self._convert_space(v)
        else:
            result["obs"] = self._convert_space(raw)

        # DreamerV3 需要的额外键
        result["is_first"] = SimpleSpace(dtype=bool, shape=())
        result["is_last"] = SimpleSpace(dtype=bool, shape=())
        result["is_terminal"] = SimpleSpace(dtype=bool, shape=())
        result["reward"] = SimpleSpace(dtype=np.float32, shape=())
        return result

    def _build_act_space(self):
        from isaaclab_tasks.user.dreamerv3.agent import SimpleSpace

        result = {}
        raw = self._env.action_space
        if isinstance(raw, spaces.Dict):
            for k, v in raw.spaces.items():
                result[k] = self._convert_space(v)
        elif isinstance(raw, spaces.Box):
            result["action"] = SimpleSpace(
                dtype=raw.dtype, shape=raw.shape[1:] if raw.shape[0] == self._env.num_envs else raw.shape,
                low=float(raw.low.min()), high=float(raw.high.max()),
                discrete=False)
        elif isinstance(raw, spaces.Discrete):
            result["action"] = SimpleSpace(
                dtype=np.int64, shape=(int(raw.n),), discrete=True)
        else:
            result["action"] = self._convert_space(raw)
        return result

    def _convert_space(self, space):
        from isaaclab_tasks.user.dreamerv3.agent import SimpleSpace

        if isinstance(space, spaces.Box):
            # IsaacLab 向量化环境的 shape 包含 num_envs 维度
            shape = space.shape[1:] if len(space.shape) > 1 else space.shape
            return SimpleSpace(
                dtype=space.dtype, shape=shape,
                low=float(space.low.min()), high=float(space.high.max()),
                discrete=False)
        elif isinstance(space, spaces.Discrete):
            return SimpleSpace(dtype=np.int64, shape=(int(space.n),), discrete=True)
        else:
            shape = getattr(space, "shape", ())
            return SimpleSpace(dtype=np.float32, shape=shape)

    @property
    def num_envs(self):
        return self._env.num_envs

    def reset(self):
        obs_raw, info = self._env.reset()
        obs = self._process_obs(obs_raw, is_first=True)
        self._is_first = False
        return obs, info

    def step(self, action):
        # 从 dict 中提取 action tensor
        if isinstance(action, dict):
            act_tensor = list(action.values())[0]
        else:
            act_tensor = action

        # 确保 action 在正确设备上
        if isinstance(act_tensor, torch.Tensor):
            act_tensor = act_tensor.to(self._device)

        obs_raw, reward, terminated, truncated, info = self._env.step(act_tensor)
        obs = self._process_obs(obs_raw, is_first=False,
                                 reward=reward, terminated=terminated,
                                 truncated=truncated)
        self._step_count += 1
        return obs, reward, terminated, truncated, info

    def _process_obs(self, obs_raw, is_first=False, reward=None,
                      terminated=None, truncated=None):
        """将环境观测转换为 DreamerV3 所需的 dict 格式。"""
        device = self._device
        N = self.num_envs

        if isinstance(obs_raw, dict):
            obs = {k: v.float().to(device) if isinstance(v, torch.Tensor)
                   else torch.tensor(v, dtype=torch.float32, device=device)
                   for k, v in obs_raw.items()}
        elif isinstance(obs_raw, torch.Tensor):
            obs = {"obs": obs_raw.float().to(device)}
        else:
            obs = {"obs": torch.tensor(obs_raw, dtype=torch.float32, device=device)}

        # 添加 DreamerV3 需要的元数据
        if is_first:
            obs["is_first"] = torch.ones(N, dtype=torch.bool, device=device)
        else:
            # is_first 在 done 的下一步为 True
            done = (terminated | truncated) if terminated is not None else torch.zeros(N, dtype=torch.bool, device=device)
            obs["is_first"] = done  # 这些环境在下一步会自动 reset

        obs["is_last"] = (
            (terminated | truncated) if terminated is not None
            else torch.zeros(N, dtype=torch.bool, device=device))
        obs["is_terminal"] = (
            terminated if terminated is not None
            else torch.zeros(N, dtype=torch.bool, device=device))
        obs["reward"] = (
            reward.float().to(device) if reward is not None
            else torch.zeros(N, dtype=torch.float32, device=device))

        return obs

    def close(self):
        self._env.close()
