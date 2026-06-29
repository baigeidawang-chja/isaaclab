from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: Tuple[int, ...] = (256, 256), act=nn.ELU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TanhNormal:
    mean: torch.Tensor
    std: torch.Tensor

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mean)
        y = self.mean + self.std * eps
        return torch.tanh(y)

    def rsample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mean)
        y = self.mean + self.std * eps
        return torch.tanh(y)

    def log_prob(self, action: torch.Tensor) -> torch.Tensor:
        # inverse tanh
        eps = 1e-6
        a = torch.clamp(action, -1 + eps, 1 - eps)
        y = 0.5 * (torch.log1p(a) - torch.log1p(-a))  # atanh
        # normal log prob
        var = self.std.pow(2)
        logp = -0.5 * (((y - self.mean) ** 2) / (var + 1e-8) + 2 * torch.log(self.std + 1e-8) + math.log(2 * math.pi))
        logp = logp.sum(dim=-1, keepdim=True)
        # tanh correction
        logp -= torch.log(1 - a**2 + 1e-6).sum(dim=-1, keepdim=True)
        return logp

    def entropy(self) -> torch.Tensor:
        # approximate: entropy of base normal (ignores tanh squash)
        ent = (0.5 + 0.5 * math.log(2 * math.pi) + torch.log(self.std + 1e-8)).sum(dim=-1, keepdim=True)
        return ent


class Actor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden=(256, 256), min_std=0.1, max_std=1.0):
        super().__init__()
        self.mean_net = MLP(feat_dim, act_dim, hidden=hidden)
        self.logstd_net = MLP(feat_dim, act_dim, hidden=hidden)
        self.min_std = float(min_std)
        self.max_std = float(max_std)

    def forward(self, feat: torch.Tensor) -> TanhNormal:
        mean = self.mean_net(feat)
        logstd = self.logstd_net(feat).clamp(-5.0, 2.0)
        std = torch.exp(logstd).clamp(self.min_std, self.max_std)
        return TanhNormal(mean=mean, std=std)


class Critic(nn.Module):
    def __init__(self, feat_dim: int, hidden=(256, 256)):
        super().__init__()
        self.v = MLP(feat_dim, 1, hidden=hidden)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.v(feat)