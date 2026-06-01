from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch


@dataclass
class Episode:
    obs: torch.Tensor      # (L, obs_dim)
    action: torch.Tensor   # (L, act_dim)
    reward: torch.Tensor   # (L, 1)
    done: torch.Tensor     # (L, 1)
    grid: Optional[torch.Tensor]  # (L, grid_dim) or None


class EpisodeReplay:
    """Episode-based replay with sequence sampling (minimal).

    Stores episodes per env stream. Supports sampling fixed-length sequences.
    """

    def __init__(self, capacity_steps: int, device: torch.device):
        self.capacity_steps = int(capacity_steps)
        self.device = device
        self._episodes: list[Episode] = []
        self._num_steps = 0

        # current building episodes buffers (per env)
        self._cur: Dict[int, Dict[str, list[torch.Tensor]]] = {}

    @property
    def num_steps(self) -> int:
        return self._num_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def reset_stream(self, env_id: int):
        self._cur.pop(int(env_id), None)

    def add_step(
        self,
        env_id: int,
        obs: torch.Tensor,      # (obs_dim,)
        action: torch.Tensor,   # (act_dim,)
        reward: torch.Tensor,   # (1,)
        done: torch.Tensor,     # (1,)
        grid: Optional[torch.Tensor] = None,  # (grid_dim,)
    ):
        eid = int(env_id)
        if eid not in self._cur:
            self._cur[eid] = {"obs": [], "action": [], "reward": [], "done": [], "grid": []}

        buf = self._cur[eid]
        buf["obs"].append(obs.detach())
        buf["action"].append(action.detach())
        buf["reward"].append(reward.detach())
        buf["done"].append(done.detach())
        if grid is not None:
            buf["grid"].append(grid.detach())

        self._num_steps += 1
        self._enforce_capacity()

        if bool(done.item()):
            self._finalize_episode(eid)

    def _finalize_episode(self, env_id: int):
        buf = self._cur.pop(int(env_id), None)
        if buf is None:
            return

        obs = torch.stack(buf["obs"], dim=0).to(self.device)
        action = torch.stack(buf["action"], dim=0).to(self.device)
        reward = torch.stack(buf["reward"], dim=0).to(self.device)
        done = torch.stack(buf["done"], dim=0).to(self.device)

        grid = None
        if len(buf["grid"]) > 0:
            grid = torch.stack(buf["grid"], dim=0).to(self.device)

        ep = Episode(obs=obs, action=action, reward=reward, done=done, grid=grid)
        self._episodes.append(ep)

    def _enforce_capacity(self):
        # crude: drop oldest episodes until under capacity_steps
        while self._num_steps > self.capacity_steps and len(self._episodes) > 0:
            ep0 = self._episodes.pop(0)
            self._num_steps -= int(ep0.obs.shape[0])

    def can_sample(self, batch_size: int, seq_len: int) -> bool:
        if len(self._episodes) == 0:
            return False
        # require at least one episode long enough
        for ep in self._episodes:
            if ep.obs.shape[0] >= seq_len:
                return True
        return False

    def sample_sequences(self, batch_size: int, seq_len: int) -> Episode:
        """Returns an Episode-like object where tensors are (B,T,...)"""
        assert batch_size > 0 and seq_len > 0
        # pick episodes that are long enough
        valid = [ep for ep in self._episodes if ep.obs.shape[0] >= seq_len]
        if len(valid) == 0:
            raise RuntimeError("No episodes long enough to sample from.")

        obs_list, act_list, rew_list, done_list, grid_list = [], [], [], [], []
        have_grid = valid[0].grid is not None

        for _ in range(batch_size):
            ep = valid[torch.randint(0, len(valid), (1,)).item()]
            L = ep.obs.shape[0]
            start = torch.randint(0, L - seq_len + 1, (1,), device=self.device).item()
            end = start + seq_len

            obs_list.append(ep.obs[start:end])
            act_list.append(ep.action[start:end])
            rew_list.append(ep.reward[start:end])
            done_list.append(ep.done[start:end])

            if ep.grid is not None:
                grid_list.append(ep.grid[start:end])

        obs = torch.stack(obs_list, dim=0)
        action = torch.stack(act_list, dim=0)
        reward = torch.stack(rew_list, dim=0)
        done = torch.stack(done_list, dim=0)

        grid = None
        if len(grid_list) > 0:
            grid = torch.stack(grid_list, dim=0)

        return Episode(obs=obs, action=action, reward=reward, done=done, grid=grid)