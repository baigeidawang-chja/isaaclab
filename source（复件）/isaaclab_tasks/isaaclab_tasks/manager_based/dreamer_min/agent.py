from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import Actor, Critic
from .rssm import RSSMState
from .utils import symlog


@dataclass
class Losses:
    model: torch.Tensor
    actor: torch.Tensor
    critic: torch.Tensor
    kl: torch.Tensor
    reward: torch.Tensor
    cont: torch.Tensor
    grid: torch.Tensor


def lambda_return(reward, value, cont, gamma=0.99, lam=0.95):
    """Compute lambda-return for imagined trajectories.
    reward: (B,H,1)
    value:  (B,H,1)
    cont:   (B,H,1) in [0,1]
    """
    B, H, _ = reward.shape
    device = reward.device

    returns = torch.zeros((B, H, 1), device=device)
    next_value = value[:, -1]
    acc = next_value
    for t in reversed(range(H)):
        disc = gamma * cont[:, t]
        acc = reward[:, t] + disc * ((1 - lam) * value[:, t] + lam * acc)
        returns[:, t] = acc
    return returns


class DreamerMin(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        act_dim: int,
        hidden=(256, 256),
        actor_ent_coef: float = 1e-3,
    ):
        super().__init__()
        self.actor = Actor(feat_dim=feat_dim, act_dim=act_dim, hidden=hidden)
        self.critic = Critic(feat_dim=feat_dim, hidden=hidden)
        self.critic_target = Critic(feat_dim=feat_dim, hidden=hidden)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_ent_coef = float(actor_ent_coef)

    @torch.no_grad()
    def act(self, feat: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist = self.actor(feat)
        if deterministic:
            a = torch.tanh(dist.mean)
        else:
            a = dist.sample()
        return a

    def actor_critic_loss(
        self,
        feats: torch.Tensor,     # (B,H,feat_dim)
        rewards: torch.Tensor,   # (B,H,1)
        cont: torch.Tensor,      # (B,H,1)
        gamma=0.99,
        lam=0.95,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, D = feats.shape
        flat = feats.reshape(B * H, D)
        v = self.critic(flat).reshape(B, H, 1)
        with torch.no_grad():
            vt = self.critic_target(flat).reshape(B, H, 1)
        # returns based on target critic
        ret = lambda_return(rewards, vt, cont, gamma=gamma, lam=lam)

        # critic loss
        critic_loss = F.mse_loss(v, ret.detach())

        # actor loss: maximize returns (minimize -returns), with entropy bonus
        dist = self.actor(flat)
        ent = dist.entropy().reshape(B, H, 1)
        actor_loss = -(ret.detach()).mean() - self.actor_ent_coef * ent.mean()

        return actor_loss, critic_loss