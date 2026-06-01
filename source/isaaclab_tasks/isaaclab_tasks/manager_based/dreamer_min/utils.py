from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def to_torch(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.as_tensor(x)
    if device is not None:
        t = t.to(device)
    if dtype is not None:
        t = t.to(dtype=dtype)
    return t


def flatten_any(x: torch.Tensor) -> torch.Tensor:
    """Flatten all but first dimension."""
    if x.ndim <= 1:
        return x
    return x.reshape(x.shape[0], -1)


def obs_extract_policy(obs: Any) -> torch.Tensor:
    """Extract policy tensor from env obs dict.
    Supports:
      - obs["policy"] is Tensor
      - obs is Tensor (fallback)
    """
    if isinstance(obs, dict) and "policy" in obs:
        return obs["policy"]
    if isinstance(obs, torch.Tensor):
        return obs
    raise TypeError(f"Unsupported observation type for policy: {type(obs)}")


def obs_extract_grid(obs: Any, grid_key: str = "obstacle_grid_map") -> Optional[torch.Tensor]:
    """Extract grid target tensor from obs dict.
    Priority:
      1) obs["privileged"][grid_key] if privileged is dict
      2) obs["privileged"] if privileged is tensor (single-term group)
      3) obs["policy"][grid_key] if policy is dict-like (rare in IsaacLab, but keep)
      4) None if not found
    """
    if not isinstance(obs, dict):
        return None

    if "privileged" in obs:
        priv = obs["privileged"]
        if isinstance(priv, dict) and grid_key in priv:
            return priv[grid_key]
        if isinstance(priv, torch.Tensor):
            return priv

    pol = obs.get("policy", None)
    if isinstance(pol, dict) and grid_key in pol:
        return pol[grid_key]

    return None


def soft_update_(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


@dataclass
class TransitionBatch:
    obs: torch.Tensor           # (B,T,obs_dim)
    action: torch.Tensor        # (B,T,act_dim)
    reward: torch.Tensor        # (B,T,1)
    done: torch.Tensor          # (B,T,1)  done at next step boundary
    grid: Optional[torch.Tensor]  # (B,T,grid_dim) float {0,1} or None