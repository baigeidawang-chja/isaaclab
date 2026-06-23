import datetime
import uuid

try:
    import gymnasium as gym
except Exception:
    import gym
import numpy as np
import torch


class IsaacLabVectorBackend:
    """Shared backend that runs one IsaacLab vector env."""

    def __init__(self, env, time_limit_steps=None):
        self._env = env
        self._time_limit_steps = int(time_limit_steps) if time_limit_steps else None
        self._num_envs = int(
            getattr(env.unwrapped, "num_envs", getattr(env, "num_envs", 1))
        )
        if self._num_envs < 1:
            raise ValueError(f"Invalid num_envs: {self._num_envs}")

        self._label_keys = frozenset(
            (
                "stuck_label",
                "mode_label",
                "contact_memory_label",
                "interaction_label",
                "medium_state_label",
            )
        )
        self._label_shapes = {}

        self._action_space = self._build_action_space()
        self._obs_space = self._build_observation_space()

        self._ids = [self._new_id() for _ in range(self._num_envs)]
        self._episode_steps = np.zeros(self._num_envs, dtype=np.int32)
        self._obs_batch = None

        self._gen = 0
        self._pending_actions = {}
        self._step_results = None
        self._step_results_gen = -1

    @property
    def num_envs(self):
        return self._num_envs

    @property
    def ids(self):
        return self._ids

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._obs_space

    def submit_action(self, index, action):
        if index in self._pending_actions:
            raise RuntimeError(f"Action for env index {index} was submitted twice.")
        if isinstance(action, dict):
            if "action" not in action:
                raise KeyError("Expected action dict to contain key 'action'.")
            action = action["action"]
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        self._pending_actions[index] = act
        return self._gen

    def consume_step_result(self, index, generation):
        # If this generation has already been stepped, directly read cached per-env result.
        if self._step_results_gen == generation:
            return self._step_results[index]
        # Otherwise, we should be exactly at the current pending generation.
        if generation != self._gen:
            raise RuntimeError(
                f"Stale generation request: got {generation}, expected {self._gen}."
            )
        if self._step_results_gen != generation:
            self._run_vector_step()
        return self._step_results[index]

    def reset_one(self, index):
        # IsaacLab RL env already auto-resets done envs on step.
        # Here we only renew trajectory id and provide initial transition observation.
        self._ids[index] = self._new_id()
        if self._obs_batch is None:
            self._reset_all()
        obs_i = {k: np.array(v[index], copy=True) for k, v in self._obs_batch.items()}
        obs_i["is_first"] = np.array(True, dtype=np.bool_)
        obs_i["is_terminal"] = np.array(False, dtype=np.bool_)
        obs_i["obs_reward"] = np.array([0.0], dtype=np.float32)
        return obs_i

    def close(self):
        self._env.close()

    def _run_vector_step(self):
        if len(self._pending_actions) != self._num_envs:
            raise RuntimeError(
                f"Vector step requires {self._num_envs} actions, got {len(self._pending_actions)}."
            )

        acts = np.stack([self._pending_actions[i] for i in range(self._num_envs)], axis=0)
        act_tensor = torch.as_tensor(
            acts, device=self._env.unwrapped.device, dtype=torch.float32
        )
        obs_raw, reward, terminated, truncated, info = self._env.step(act_tensor)

        rew = self._to_numpy(reward).reshape(self._num_envs).astype(np.float32)
        term = self._to_numpy(terminated).reshape(self._num_envs).astype(np.bool_)
        trunc = self._to_numpy(truncated).reshape(self._num_envs).astype(np.bool_)
        done = np.logical_or(term, trunc)

        if self._time_limit_steps is not None:
            self._episode_steps += 1
            timeout = self._episode_steps >= self._time_limit_steps
            done = np.logical_or(done, timeout)
        else:
            timeout = np.zeros_like(done)

        is_first = done.copy()
        is_terminal = np.logical_and(done, np.logical_not(timeout))
        discount = np.where(is_terminal, 0.0, 1.0).astype(np.float32)
        self._episode_steps[done] = 0

        obs_batch = self._process_obs(
            obs_raw, reward=rew, is_first=is_first, is_terminal=is_terminal
        )
        self._obs_batch = obs_batch

        out_info = dict(info) if isinstance(info, dict) else {}
        out_info["discount"] = discount
        reward_terms_step = self._collect_reward_terms_step()

        self._step_results = {}
        for i in range(self._num_envs):
            obs_i = {k: np.array(v[i], copy=True) for k, v in obs_batch.items()}
            info_i = {"discount": np.array(discount[i], dtype=np.float32)}
            if reward_terms_step:
                for key, arr in reward_terms_step.items():
                    info_i[f"log_reward/{key}"] = np.array(arr[i], dtype=np.float32)
            self._step_results[i] = (obs_i, float(rew[i]), bool(done[i]), info_i)

        self._pending_actions.clear()
        self._step_results_gen = self._gen
        self._gen += 1

    def _reset_all(self):
        obs_raw, _ = self._env.reset()
        self._ids = [self._new_id() for _ in range(self._num_envs)]
        self._episode_steps[:] = 0
        self._obs_batch = self._process_obs(
            obs_raw,
            reward=np.zeros(self._num_envs, dtype=np.float32),
            is_first=np.ones(self._num_envs, dtype=np.bool_),
            is_terminal=np.zeros(self._num_envs, dtype=np.bool_),
        )
        self._pending_actions.clear()
        self._step_results = None
        self._step_results_gen = -1

    def _build_action_space(self):
        space = self._env.action_space
        if hasattr(space, "low") and hasattr(space, "high"):
            low = np.asarray(space.low, dtype=np.float32)
            high = np.asarray(space.high, dtype=np.float32)
            if low.ndim >= 2:
                low = low[0]
                high = high[0]
            return gym.spaces.Box(low=low, high=high, dtype=np.float32)
        raise TypeError(f"Unsupported action space type: {type(space)}")

    def _build_observation_space(self):
        sample, _ = self._env.reset()
        obs_dim = int(self._flatten_obs(sample).shape[-1])
        label_spaces = {}
        label_shapes = {}
        if isinstance(sample, dict):
            for key, value in sample.items():
                if not self._is_label_key(key):
                    continue
                arr = self._to_numpy(value).reshape(self._num_envs, -1)
                label_shapes[key] = tuple(arr.shape[1:])
        for key, shape in sorted(label_shapes.items()):
            label_spaces[key] = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=shape, dtype=np.float32
            )
        self._label_shapes = label_shapes
        return gym.spaces.Dict(
            {
                "obs": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                ),
                "is_first": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
                "is_terminal": gym.spaces.Box(low=0, high=1, shape=(), dtype=np.bool_),
                **label_spaces,
            }
        )

    def _process_obs(self, obs_raw, reward, is_first, is_terminal):
        vec = self._flatten_obs(obs_raw).astype(np.float32)
        labels = {}
        for key, shape in self._label_shapes.items():
            if isinstance(obs_raw, dict) and key in obs_raw:
                labels[key] = self._to_numpy(obs_raw[key]).reshape(self._num_envs, -1).astype(np.float32)
            else:
                labels[key] = np.zeros((self._num_envs, int(np.prod(shape))), dtype=np.float32)
        return {
            "obs": vec,
            "is_first": np.asarray(is_first, dtype=np.bool_),
            "is_terminal": np.asarray(is_terminal, dtype=np.bool_),
            **labels,
            "obs_reward": np.asarray(reward, dtype=np.float32)[:, None],
        }

    def _flatten_obs(self, obs_raw):
        chunks = []
        if isinstance(obs_raw, dict):
            keys = sorted(obs_raw.keys())
            for key in keys:
                if self._is_label_key(key):
                    # Keep supervision labels separate to avoid
                    # leaking ground-truth directly into the main obs embedding.
                    continue
                arr = self._to_numpy(obs_raw[key])
                if arr.ndim == 1:
                    arr = np.repeat(arr[None, :], self._num_envs, axis=0)
                else:
                    arr = arr.reshape(self._num_envs, -1)
                chunks.append(arr)
        else:
            arr = self._to_numpy(obs_raw)
            if arr.ndim == 1:
                arr = np.repeat(arr[None, :], self._num_envs, axis=0)
            else:
                arr = arr.reshape(self._num_envs, -1)
            chunks.append(arr)
        if not chunks:
            return np.zeros((self._num_envs, 1), dtype=np.float32)
        return np.concatenate(chunks, axis=1)

    def _is_label_key(self, key):
        return key in self._label_keys or key.endswith("_label")

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _new_id():
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        return f"{ts}-{uuid.uuid4().hex}"

    def _collect_reward_terms_step(self):
        """Best-effort per-step weighted reward contributions for each active term.

        Returns:
            Dict[str, np.ndarray]: term -> shape (num_envs,), each value is the
            per-step contribution to env reward return (already multiplied by step_dt).
        """
        try:
            env_unwrapped = self._env.unwrapped
            rm = getattr(env_unwrapped, "reward_manager", None)
            if rm is None:
                return {}
            term_names = list(getattr(rm, "active_terms", []))
            step_reward = getattr(rm, "_step_reward", None)
            if step_reward is None or len(term_names) == 0:
                return {}
            step_reward_np = self._to_numpy(step_reward)
            if step_reward_np.ndim != 2:
                return {}
            dt = float(getattr(env_unwrapped, "step_dt", 1.0))
            # RewardManager stores weighted term divided by dt; multiply back to get
            # per-step return contribution that sums to episode reward component.
            contrib = step_reward_np * dt
            out = {}
            cols = min(contrib.shape[1], len(term_names))
            for j in range(cols):
                out[term_names[j]] = contrib[:, j].astype(np.float32, copy=False)
            return out
        except Exception:
            # Never block training because of diagnostics.
            return {}


class IsaacLabEnvProxy(gym.Env):
    """One env view over one index of IsaacLabVectorBackend."""

    metadata = {}

    def __init__(self, backend: IsaacLabVectorBackend, index: int):
        super().__init__()
        self._backend = backend
        self._index = int(index)
        self.action_space = backend.action_space
        self.observation_space = backend.observation_space

    @property
    def id(self):
        return self._backend.ids[self._index]

    def reset(self):
        return lambda: self._backend.reset_one(self._index)

    def step(self, action):
        gen = self._backend.submit_action(self._index, action)
        return lambda: self._backend.consume_step_result(self._index, gen)

    def close(self):
        # Owned by backend owner.
        return None


def make_isaaclab_env_proxies(env, time_limit_steps=None):
    backend = IsaacLabVectorBackend(env, time_limit_steps=time_limit_steps)
    envs = [IsaacLabEnvProxy(backend, i) for i in range(backend.num_envs)]
    return backend, envs
