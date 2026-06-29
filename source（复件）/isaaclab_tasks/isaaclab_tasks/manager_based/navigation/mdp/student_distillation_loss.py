"""
Stage-2 Student Distillation Loss.

Three-part loss:
  L1 (behavior equivalence):  student latent -> teacher policy head -> action
                               vs teacher action. Core "can it drive?" loss.
  L2 (relevance alignment):   student relevance_hat vs teacher relevance_map.
                               Teaches student "where to look".
  L3 (relevance-weighted policy alignment): Upweight action-matching loss on
                               samples where teacher relevance is high.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class StudentDistillationLoss(nn.Module):
    """Combined loss for stage-2 distillation with relevance maps.

    Args:
        w_behavior: Weight for behavior equivalence loss (L1).
        w_relevance: Weight for relevance alignment loss (L2).
        w_relevance_weighted: Weight for relevance-weighted policy loss (L3).
        relevance_loss_type: "mse", "bce", or "kl" for L2.
        relevance_weight_mode: How to compute per-sample weight from relevance
            for L3. "peak" uses max relevance, "sum" uses total relevance.
    """

    def __init__(
        self,
        w_behavior: float = 1.0,
        w_relevance: float = 0.5,
        w_relevance_weighted: float = 0.3,
        relevance_loss_type: str = "mse",
        relevance_weight_mode: str = "peak",
    ):
        super().__init__()
        self.w_behavior = w_behavior
        self.w_relevance = w_relevance
        self.w_relevance_weighted = w_relevance_weighted
        self.relevance_loss_type = relevance_loss_type
        self.relevance_weight_mode = relevance_weight_mode

    def forward(
        self,
        # Student outputs
        latent_hat: torch.Tensor,       # (B, latent_dim)
        relevance_hat: torch.Tensor,    # (B, grid_cells) logits
        # Teacher outputs (frozen, precomputed)
        teacher_action: torch.Tensor,   # (B, act_dim) deterministic mean
        teacher_relevance: torch.Tensor, # (B, grid_cells) normalized [0,1]
        # Teacher's policy head (frozen) that maps latent -> action
        teacher_policy_head: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute the three-part loss.

        Args:
            latent_hat: Student's predicted latent.
            relevance_hat: Student's predicted relevance logits.
            teacher_action: Teacher's deterministic action mean.
            teacher_relevance: Teacher's relevance map (from RelevanceMapGenerator).
            teacher_policy_head: Frozen teacher's policy head that takes
                (latent) -> (action_mean). This is the "consumer" of the latent.

        Returns:
            Dict with keys: "loss", "L1", "L2", "L3", and individual components.
        """
        B = latent_hat.shape[0]

        # ============================================================
        # L1: Behavior Equivalence Loss (core)
        # student_latent -> teacher_policy_head -> predicted_action
        # vs teacher_action
        # ============================================================
        with torch.no_grad():
            # teacher_policy_head is frozen, but we need grads through latent_hat
            pass
        # We do need grads through latent_hat, so don't use no_grad here
        predicted_action = teacher_policy_head(latent_hat)  # (B, act_dim)
        L1 = F.mse_loss(predicted_action, teacher_action)

        # ============================================================
        # L2: Relevance Alignment Loss
        # student relevance_hat vs teacher relevance_map
        # ============================================================
        relevance_prob = torch.sigmoid(relevance_hat)  # (B, grid_cells)

        if self.relevance_loss_type == "mse":
            L2 = F.mse_loss(relevance_prob, teacher_relevance)
        elif self.relevance_loss_type == "bce":
            L2 = F.binary_cross_entropy_with_logits(
                relevance_hat, teacher_relevance
            )
        elif self.relevance_loss_type == "kl":
            # Treat as distributions: KL(teacher || student)
            # Add small eps for numerical stability
            eps = 1e-8
            teacher_dist = teacher_relevance.clamp(eps, 1 - eps)
            student_dist = relevance_prob.clamp(eps, 1 - eps)
            # Normalize to valid distributions
            teacher_dist = teacher_dist / teacher_dist.sum(dim=-1, keepdim=True)
            student_dist = student_dist / student_dist.sum(dim=-1, keepdim=True)
            L2 = F.kl_div(
                student_dist.log(), teacher_dist, reduction="batchmean"
            )
        else:
            raise ValueError(f"Unknown relevance_loss_type: {self.relevance_loss_type}")

        # ============================================================
        # L3: Relevance-Weighted Policy Alignment
        # Upweight action loss on samples where teacher relevance is high
        # ============================================================
        with torch.no_grad():
            if self.relevance_weight_mode == "peak":
                # Per-sample weight = max relevance across all cells
                sample_weight = teacher_relevance.max(dim=-1).values  # (B,)
            elif self.relevance_weight_mode == "sum":
                # Per-sample weight = total relevance (higher = more obstacles matter)
                sample_weight = teacher_relevance.sum(dim=-1)  # (B,)
                # Normalize to [0, 1] range
                sw_max = sample_weight.max().clamp(min=1e-8)
                sample_weight = sample_weight / sw_max
            else:
                raise ValueError(f"Unknown relevance_weight_mode: {self.relevance_weight_mode}")

            # Ensure weights are positive and meaningful
            sample_weight = sample_weight.clamp(min=0.1)  # minimum weight

        # Per-sample action MSE, weighted by relevance
        per_sample_action_error = (predicted_action - teacher_action).pow(2).mean(dim=-1)  # (B,)
        L3 = (per_sample_action_error * sample_weight).mean()

        # ============================================================
        # Total loss
        # ============================================================
        total_loss = (
            self.w_behavior * L1
            + self.w_relevance * L2
            + self.w_relevance_weighted * L3
        )

        return {
            "loss": total_loss,
            "L1_behavior": L1.detach(),
            "L2_relevance": L2.detach(),
            "L3_weighted": L3.detach(),
        }
