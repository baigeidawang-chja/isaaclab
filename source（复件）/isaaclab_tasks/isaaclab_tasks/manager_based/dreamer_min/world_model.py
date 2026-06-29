from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import MLP
from .rssm import RSSM, RSSMState


@dataclass
class WorldModelOutputs:
    post_states: list[RSSMState]
    prior_states: list[RSSMState]
    feats: torch.Tensor           # (B,T,feat_dim)
    reward_pred: torch.Tensor     # (B,T,1)
    cont_logits: torch.Tensor     # (B,T,1)
    grid_logits: Optional[torch.Tensor]  # (B,T,G) or None
    kl: torch.Tensor              # (B,T,1)


class WorldModel(nn.Module):
    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        embed_dim: int = 256,
        deter_dim: int = 256,
        stoch_dim: int = 32,
        hidden=(256, 256),
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.embed_dim = embed_dim

        self.encoder = MLP(obs_dim, embed_dim, hidden=hidden)
        self.rssm = RSSM(action_dim=action_dim, embed_dim=embed_dim, deter_dim=deter_dim, stoch_dim=stoch_dim)

        feat_dim = deter_dim + stoch_dim
        self.reward_head = MLP(feat_dim, 1, hidden=hidden)
        self.cont_head = MLP(feat_dim, 1, hidden=hidden)

        # grid head is optional / lazy-created because grid_dim is unknown at init
        self._grid_head: Optional[nn.Module] = None
        self._grid_dim: Optional[int] = None
        self._feat_dim = feat_dim
        self._hidden = tuple(hidden)

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

    def ensure_grid_head(self, grid_dim: int):
        grid_dim = int(grid_dim)
        if self._grid_head is None or self._grid_dim != grid_dim:
            self._grid_dim = grid_dim
            self._grid_head = MLP(self._feat_dim, grid_dim, hidden=self._hidden).to(next(self.parameters()).device)

    def forward_sequence(
        self,
        obs: torch.Tensor,        # (B,T,obs_dim)
        action: torch.Tensor,     # (B,T,act_dim)
        init_state: Optional[RSSMState] = None,
        predict_grid: bool = False,
    ) -> WorldModelOutputs:
        B, T, _ = obs.shape
        device = obs.device

        if init_state is None:
            state = self.rssm.init_state(B, device=device)
        else:
            state = init_state

        post_states = []
        prior_states = []
        feats = []
        reward_pred = []
        cont_logits = []
        grid_logits = []

        # prev_action for t=0: use zeros
        prev_a = torch.zeros((B, action.shape[-1]), device=device, dtype=obs.dtype)

        for t in range(T):
            embed = self.encoder(obs[:, t])
            post, prior = self.rssm.obs_step(state, prev_a, embed)
            state = post

            feat = self.rssm.feat(post)
            feats.append(feat)
            post_states.append(post)
            prior_states.append(prior)

            reward_pred.append(self.reward_head(feat))
            cont_logits.append(self.cont_head(feat))

            if predict_grid and self._grid_head is not None:
                grid_logits.append(self._grid_head(feat))

            prev_a = action[:, t]

        feats_t = torch.stack(feats, dim=1)
        reward_t = torch.stack(reward_pred, dim=1)
        cont_t = torch.stack(cont_logits, dim=1)
        kl_t = torch.stack([self.rssm.kl_div(p, q) for p, q in zip(post_states, prior_states)], dim=1)

        grid_t = None
        if predict_grid and self._grid_head is not None and len(grid_logits) == T:
            grid_t = torch.stack(grid_logits, dim=1)

        return WorldModelOutputs(
            post_states=post_states,
            prior_states=prior_states,
            feats=feats_t,
            reward_pred=reward_t,
            cont_logits=cont_t,
            grid_logits=grid_t,
            kl=kl_t,
        )

    def imagine(
        self,
        start_state: RSSMState,
        actor,  # Actor module returning distribution
        horizon: int = 15,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, RSSMState]:
        """Imagined rollout from start_state using prior dynamics.
        Returns:
          feats: (B,H,feat_dim)
          rewards: (B,H,1)
          cont: (B,H,1) in [0,1]
          last_state
        """
        B = start_state.deter.shape[0]
        device = start_state.deter.device

        state = start_state
        feats = []
        rews = []
        conts = []

        prev_a = torch.zeros((B, self.action_dim), device=device)

        for _ in range(horizon):
            feat = self.rssm.feat(state)
            dist = actor(feat)
            a = dist.rsample()  # reparam

            state = self.rssm.img_step(state, a)
            feat2 = self.rssm.feat(state)

            r = self.reward_head(feat2)
            c = torch.sigmoid(self.cont_head(feat2))

            feats.append(feat2)
            rews.append(r)
            conts.append(c)
            prev_a = a

        return torch.stack(feats, dim=1), torch.stack(rews, dim=1), torch.stack(conts, dim=1), state