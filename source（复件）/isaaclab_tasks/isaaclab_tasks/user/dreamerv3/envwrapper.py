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
        self._num_envs = getattr(env, 'num_envs',
                                  getattr(env.unwrapped, 'num_envs', 1))

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
        elif isinstance(raw, spaces.Box):
            result["obs"] = self._convert_space(raw)
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
            # IsaacLab 向量化环境的 Box shape = (num_envs, act_dim)
            shape = raw.shape[1:] if len(raw.shape) > 1 else raw.shape
            result["action"] = SimpleSpace(
                dtype=raw.dtype, shape=shape,
                low=float(raw.low.min()), high=float(raw.high.max()),
                discrete=False)
        elif isinstance(raw, spaces.Discrete):
            result["action"] = SimpleSpace(
                dtype=np.int64, shape=(int(raw.n),), discrete=True)
        elif isinstance(raw, spaces.MultiDiscrete):
            result["action"] = SimpleSpace(
                dtype=np.int64, shape=tuple(raw.nvec),
                discrete=True)
        else:
            result["action"] = self._convert_space(raw)
        return result

    def _convert_space(self, space):
        from isaaclab_tasks.user.dreamerv3.agent import SimpleSpace

        if isinstance(space, spaces.Box):
            # 去掉 num_envs 维度
            shape = space.shape[1:] if (
                len(space.shape) > 1 and space.shape[0] == self._num_envs
            ) else space.shape
            shape = tuple(shape)
            while len(shape) > 1 and shape[0] == 1:
                shape = shape[1:]
                
            return SimpleSpace(
                dtype=space.dtype, shape=shape,
                low=float(space.low.min()), high=float(space.high.max()),
                discrete=False)
        elif isinstance(space, spaces.Discrete):
            return SimpleSpace(
                dtype=np.int64, shape=(int(space.n),), discrete=True)
        elif isinstance(space, spaces.MultiBinary):
            return SimpleSpace(
                dtype=np.int8, shape=space.shape, discrete=True)
        elif isinstance(space, spaces.MultiDiscrete):
            return SimpleSpace(
                dtype=np.int64, shape=tuple(space.nvec), discrete=True)
        else:
            shape = getattr(space, 'shape', ())
            return SimpleSpace(dtype=np.float32, shape=shape)

    @property
    def num_envs(self):
        return self._num_envs

    def reset(self):
        obs_raw, info = self._env.reset()
        obs = self._process_obs(obs_raw, is_first=True)
        self._is_first = False
        return obs, info

    def step(self, action):
        """
        Args:
            action: dict 或 tensor
        Returns:
            obs, reward, terminated, truncated, info
        """
        # 提取 action tensor
        if isinstance(action, dict):
            act_tensor = list(action.values())[0]
        else:
            act_tensor = action

        if isinstance(act_tensor, torch.Tensor):
            act_tensor = act_tensor.to(self._device)

        obs_raw, reward, terminated, truncated, info = self._env.step(act_tensor)

        # 确保 reward 是 1D tensor
        if isinstance(reward, torch.Tensor):
            reward = reward.squeeze()
            if reward.dim() == 0:
                reward = reward.unsqueeze(0)
        else:
            reward = torch.tensor(reward, dtype=torch.float32, device=self._device)
            if reward.dim() == 0:
                reward = reward.unsqueeze(0)

        # 确保 terminated / truncated 是 bool tensor
        def to_bool_tensor(x):
            if isinstance(x, torch.Tensor):
                return x.bool().to(self._device).squeeze()
            return torch.tensor(x, dtype=torch.bool, device=self._device).squeeze()

        terminated = to_bool_tensor(terminated)
        truncated = to_bool_tensor(truncated)
        if terminated.dim() == 0:
            terminated = terminated.unsqueeze(0)
        if truncated.dim() == 0:
            truncated = truncated.unsqueeze(0)

        obs = self._process_obs(obs_raw, is_first=False,
                                 reward=reward, terminated=terminated,
                                 truncated=truncated)
        self._step_count += 1
        return obs, reward, terminated, truncated, info

    def _process_obs(self, obs_raw, is_first=False, reward=None,
                      terminated=None, truncated=None):
        """将原始观测转为 DreamerV3 所需的 dict 格式。"""
        device = self._device
        N = self.num_envs

        # 处理原始观测
        if isinstance(obs_raw, dict):
            obs = {}
            for k, v in obs_raw.items():
                if isinstance(v, torch.Tensor):
                    obs[k] = v.float().to(device)
                elif isinstance(v, np.ndarray):
                    obs[k] = torch.from_numpy(v).float().to(device)
                else:
                    obs[k] = torch.tensor(v, dtype=torch.float32, device=device)
        elif isinstance(obs_raw, torch.Tensor):
            obs = {"obs": obs_raw.float().to(device)}
        elif isinstance(obs_raw, np.ndarray):
            obs = {"obs": torch.from_numpy(obs_raw).float().to(device)}
        else:
            obs = {"obs": torch.tensor(obs_raw, dtype=torch.float32, device=device)}

        # 添加 DreamerV3 元数据
        if is_first:
            obs["is_first"] = torch.ones(N, dtype=torch.bool, device=device)
        elif terminated is not None:
            # done 的环境在下一步自动 reset → is_first=True
            done = terminated | truncated
            obs["is_first"] = done
        else:
            obs["is_first"] = torch.zeros(N, dtype=torch.bool, device=device)

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

    def __getattr__(self, name):
        """代理未定义的属性到内部 env。"""
        return getattr(self._env, name)