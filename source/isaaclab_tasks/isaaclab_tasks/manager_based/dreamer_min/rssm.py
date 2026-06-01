from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RSSMState:
    deter: torch.Tensor  # (B, Dd)
    stoch: torch.Tensor  # (B, Ds)
    mean: torch.Tensor   # (B, Ds)
    std: torch.Tensor    # (B, Ds)


class RSSM(nn.Module):
    """Minimal continuous RSSM (Gaussian stoch) suitable for a first working version."""

    def __init__(self, action_dim: int, embed_dim: int, deter_dim: int = 256, stoch_dim: int = 32):
        super().__init__()
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.gru = nn.GRUCell(input_size=stoch_dim + action_dim, hidden_size=deter_dim)

        # prior: p(stoch_t | deter_t)
        # 纯想象，只用action推forward
        self.prior = nn.Sequential(
            nn.Linear(deter_dim, 256),
            nn.ELU(),
            nn.Linear(256, 2 * stoch_dim),
        )

        # posterior: q(stoch_t | deter_t, embed_t)
        # 带观测校正
        self.post = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 2 * stoch_dim),
        )

    def init_state(self, batch: int, device: torch.device) -> RSSMState:
        deter = torch.zeros((batch, self.deter_dim), device=device)
        mean = torch.zeros((batch, self.stoch_dim), device=device)
        std = torch.ones((batch, self.stoch_dim), device=device)
        stoch = torch.zeros((batch, self.stoch_dim), device=device)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    def _stats_to_dist(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, logstd = torch.chunk(stats, 2, dim=-1)
        std = torch.exp(logstd.clamp(-5.0, 2.0))
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mean)
        return mean + std * eps

    def obs_step(self, prev: RSSMState, prev_action: torch.Tensor, embed: torch.Tensor) -> Tuple[RSSMState, RSSMState]:
        """Posterior update using current embed. Returns (post, prior)."""
        x = torch.cat([prev.stoch, prev_action], dim=-1)
        deter = self.gru(x, prev.deter)

        prior_stats = self.prior(deter)
        prior_mean, prior_std = self._stats_to_dist(prior_stats)
        prior_stoch = self._sample(prior_mean, prior_std)
        prior = RSSMState(deter=deter, stoch=prior_stoch, mean=prior_mean, std=prior_std)

        post_stats = self.post(torch.cat([deter, embed], dim=-1))
        post_mean, post_std = self._stats_to_dist(post_stats)
        post_stoch = self._sample(post_mean, post_std)
        post = RSSMState(deter=deter, stoch=post_stoch, mean=post_mean, std=post_std)

        return post, prior

    def img_step(self, prev: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        """Prior rollout without observations."""
        x = torch.cat([prev.stoch, prev_action], dim=-1)
        deter = self.gru(x, prev.deter)
        prior_stats = self.prior(deter)
        mean, std = self._stats_to_dist(prior_stats)
        stoch = self._sample(mean, std)
        return RSSMState(deter=deter, stoch=stoch, mean=mean, std=std)

    @staticmethod
    def kl_div(post: RSSMState, prior: RSSMState) -> torch.Tensor:
        """KL(q||p) for diagonal Gaussians. Returns (B,1)."""
        # KL(N(m1,s1)||N(m2,s2))
        m1, s1 = post.mean, post.std
        m2, s2 = prior.mean, prior.std
        v1 = s1**2
        v2 = s2**2
        kl = torch.log((s2 + 1e-8) / (s1 + 1e-8)) + (v1 + (m1 - m2) ** 2) / (2 * v2 + 1e-8) - 0.5
        kl = kl.sum(dim=-1, keepdim=True)
        return kl

    def feat(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state.deter, state.stoch], dim=-1)