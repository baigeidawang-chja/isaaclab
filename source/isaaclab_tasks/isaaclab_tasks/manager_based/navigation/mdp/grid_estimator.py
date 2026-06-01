"""
Stage-2 Student Grid Estimator.

Takes proprioceptive obs + previous action and outputs:
  - latent_hat  (D-dim): replaces teacher's grid_feat for policy consumption
  - relevance_hat (H*W logits): predicts teacher's decision-relevance map
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class GridEstimatorV2(nn.Module):
    """Student estimator that predicts latent + relevance map from proprioception.

    Architecture:
        input_encoder -> GRU -> split heads:
            head_latent    -> latent_hat  (latent_dim)
            head_relevance -> relevance_hat (grid_cells logits)

    Args:
        proprio_dim: Dimension of non-grid observations (base_vel, ang_vel, gravity, joint_vel, last_action, heading, etc.)
        action_dim: Dimension of the action space.
        latent_dim: Dimension of latent output (must match teacher's grid_feat dim, e.g. 32).
        grid_cells: Total number of grid cells (e.g. 20*20 = 400).
        hidden_dim: GRU hidden dimension.
        encoder_hidden: MLP hidden layer sizes for input encoder.
        num_gru_layers: Number of GRU layers.
    """

    def __init__(
        self,
        proprio_dim: int,
        action_dim: int,
        latent_dim: int = 32,
        grid_cells: int = 400,
        hidden_dim: int = 128,
        encoder_hidden: Tuple[int, ...] = (128, 128),
        num_gru_layers: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_cells = grid_cells
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        # Input encoder: proprio + prev_action -> features
        input_dim = proprio_dim + action_dim
        layers = []
        in_d = input_dim
        for h in encoder_hidden:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ELU())
            in_d = h
        self.input_encoder = nn.Sequential(*layers)

        # GRU for temporal aggregation
        self.gru = nn.GRU(
            input_size=in_d,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
        )

        # Head: latent (for policy consumption)
        self.head_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Head: relevance map (logits, same shape as grid)
        self.head_relevance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, grid_cells),
        )

    def forward(
        self,
        proprio: torch.Tensor,
        prev_action: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            proprio: (B, proprio_dim) current proprioceptive obs (no grid).
            prev_action: (B, action_dim) previous action a_{t-1}.
            hidden: (num_layers, B, hidden_dim) GRU hidden state or None.

        Returns:
            latent_hat: (B, latent_dim)
            relevance_hat: (B, grid_cells) logits (apply sigmoid for prob)
            hidden_new: (num_layers, B, hidden_dim)
        """
        x = torch.cat([proprio, prev_action], dim=-1)  # (B, proprio+act)
        x = self.input_encoder(x)  # (B, encoder_out)
        x = x.unsqueeze(1)  # (B, 1, encoder_out) for GRU

        if hidden is None:
            hidden = torch.zeros(
                self.num_gru_layers, x.shape[0], self.hidden_dim,
                device=x.device, dtype=x.dtype,
            )

        gru_out, hidden_new = self.gru(x, hidden)  # (B, 1, hidden), (L, B, hidden)
        h = gru_out.squeeze(1)  # (B, hidden)

        latent_hat = self.head_latent(h)  # (B, latent_dim)
        relevance_hat = self.head_relevance(h)  # (B, grid_cells)

        return latent_hat, relevance_hat, hidden_new

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            self.num_gru_layers, batch_size, self.hidden_dim,
            device=device,
        )
