"""
Teacher Decision-Relevance Map Generator.

For each observation sample, perturb each patch of the grid input and measure
the teacher's action change.  The resulting per-patch "relevance score" is
broadcast back to cell resolution to produce a dense relevance map that can
supervise the student.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class RelevanceMapGenerator:
    """Generates teacher decision-relevance maps via patch-level perturbation.

    Given a frozen teacher network, for each sample we:
      1. Run the teacher with the original grid  -> a_base
      2. For each patch p, erase that patch (set to 0) -> a_p
      3. relevance_p = ||a_base - a_p||_2
      4. Normalize across patches (softmax or min-max)
      5. Broadcast patch scores back to cell resolution

    Args:
        teacher_actor: Frozen teacher actor network (obs -> action mean).
        grid_size: Number of cells per side (e.g. 20 for 20x20).
        patch_size: Number of cells per patch side (e.g. 5 for 5x5 patches).
        grid_obs_start_idx: Start index of grid features in the obs vector.
        grid_obs_end_idx: End index (exclusive) of grid features in the obs vector.
        normalize_mode: "softmax" or "minmax".
        erase_value: Value to fill erased patches (0.0 = free space).
    """

    def __init__(
        self,
        teacher_actor: nn.Module,
        grid_size: int = 20,
        patch_size: int = 5,
        grid_obs_start_idx: int = 0,
        grid_obs_end_idx: int = 400,
        normalize_mode: str = "softmax",
        erase_value: float = 0.0,
        temperature: float = 1.0,
    ):
        self.teacher_actor = teacher_actor
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.grid_start = grid_obs_start_idx
        self.grid_end = grid_obs_end_idx
        self.normalize_mode = normalize_mode
        self.erase_value = erase_value
        self.temperature = temperature

        assert grid_size % patch_size == 0, (
            f"grid_size ({grid_size}) must be divisible by patch_size ({patch_size})"
        )
        self.patches_per_side = grid_size // patch_size
        self.num_patches = self.patches_per_side ** 2

        # Precompute patch masks: (num_patches, grid_size, grid_size)
        self._patch_masks = self._build_patch_masks()

    def _build_patch_masks(self) -> torch.Tensor:
        """Build boolean masks for each patch. True = belongs to patch p."""
        masks = torch.zeros(self.num_patches, self.grid_size, self.grid_size, dtype=torch.bool)
        for p in range(self.num_patches):
            row = p // self.patches_per_side
            col = p % self.patches_per_side
            r_start = row * self.patch_size
            r_end = r_start + self.patch_size
            c_start = col * self.patch_size
            c_end = c_start + self.patch_size
            masks[p, r_start:r_end, c_start:c_end] = True
        return masks

    @torch.no_grad()
    def compute_relevance(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute relevance map for a batch of observations.

        Args:
            obs: Full observation tensor, shape (B, obs_dim).
                 Must contain grid features at indices [grid_start:grid_end].

        Returns:
            relevance_map: (B, grid_size*grid_size) dense relevance in [0,1].
            patch_scores:  (B, num_patches) raw (normalized) patch relevance.
        """
        device = obs.device
        B = obs.shape[0]
        patch_masks = self._patch_masks.to(device)  # (P, H, W)

        # --- 1. baseline action ---
        a_base = self.teacher_actor(obs)  # (B, act_dim)

        # --- 2. perturbed actions for each patch ---
        # We batch all patches together: (B * P, obs_dim)
        obs_expanded = obs.unsqueeze(1).expand(B, self.num_patches, -1).reshape(B * self.num_patches, -1).clone()

        # Extract grid portion and reshape to (B*P, H, W)
        grid_flat = obs_expanded[:, self.grid_start:self.grid_end]
        grid_2d = grid_flat.view(B * self.num_patches, self.grid_size, self.grid_size)

        # Apply patch erasure
        # patch_masks: (P, H, W) -> expand to (B*P, H, W)
        masks_expanded = patch_masks.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * self.num_patches, self.grid_size, self.grid_size)
        grid_2d[masks_expanded] = self.erase_value

        # Write back
        obs_expanded[:, self.grid_start:self.grid_end] = grid_2d.view(B * self.num_patches, -1)

        # Forward pass (batched)
        a_perturbed = self.teacher_actor(obs_expanded)  # (B*P, act_dim)
        a_perturbed = a_perturbed.view(B, self.num_patches, -1)  # (B, P, act_dim)

        # --- 3. relevance = L2 distance ---
        a_base_expanded = a_base.unsqueeze(1).expand_as(a_perturbed)  # (B, P, act_dim)
        patch_scores = torch.norm(a_perturbed - a_base_expanded, dim=-1)  # (B, P)

        # --- 4. normalize ---
        if self.normalize_mode == "softmax":
            patch_scores_norm = F.softmax(patch_scores / self.temperature, dim=-1)  # (B, P)
        elif self.normalize_mode == "minmax":
            p_min = patch_scores.min(dim=-1, keepdim=True).values
            p_max = patch_scores.max(dim=-1, keepdim=True).values
            denom = (p_max - p_min).clamp(min=1e-8)
            patch_scores_norm = (patch_scores - p_min) / denom  # (B, P)
        else:
            raise ValueError(f"Unknown normalize_mode: {self.normalize_mode}")

        # --- 5. broadcast to cell resolution ---
        # (B, P) -> (B, H, W)
        relevance_map_2d = torch.zeros(B, self.grid_size, self.grid_size, device=device)
        for p in range(self.num_patches):
            mask_p = patch_masks[p]  # (H, W)
            relevance_map_2d[:, mask_p] = patch_scores_norm[:, p].unsqueeze(-1).expand(
                -1, mask_p.sum().item()
            )

        relevance_map = relevance_map_2d.view(B, -1)  # (B, H*W)

        return relevance_map, patch_scores_norm

    @torch.no_grad()
    def compute_relevance_batched(
        self,
        obs: torch.Tensor,
        max_batch_patches: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Memory-efficient version that processes patches in chunks.

        Use this when B * num_patches is too large for GPU memory.
        """
        device = obs.device
        B = obs.shape[0]
        patch_masks = self._patch_masks.to(device)

        # baseline
        a_base = self.teacher_actor(obs)  # (B, act_dim)

        patch_scores = torch.zeros(B, self.num_patches, device=device)

        # Process patches in chunks
        chunk_size = max(1, max_batch_patches // B)
        for p_start in range(0, self.num_patches, chunk_size):
            p_end = min(p_start + chunk_size, self.num_patches)
            n_p = p_end - p_start

            obs_chunk = obs.unsqueeze(1).expand(B, n_p, -1).reshape(B * n_p, -1).clone()

            grid_flat = obs_chunk[:, self.grid_start:self.grid_end]
            grid_2d = grid_flat.view(B * n_p, self.grid_size, self.grid_size)

            masks_chunk = patch_masks[p_start:p_end].unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n_p, self.grid_size, self.grid_size)
            grid_2d[masks_chunk] = self.erase_value

            obs_chunk[:, self.grid_start:self.grid_end] = grid_2d.view(B * n_p, -1)

            a_p = self.teacher_actor(obs_chunk).view(B, n_p, -1)
            a_base_exp = a_base.unsqueeze(1).expand(B, n_p, -1)
            patch_scores[:, p_start:p_end] = torch.norm(a_p - a_base_exp, dim=-1)

        # normalize
        if self.normalize_mode == "softmax":
            patch_scores_norm = F.softmax(patch_scores / self.temperature, dim=-1)
        else:
            p_min = patch_scores.min(dim=-1, keepdim=True).values
            p_max = patch_scores.max(dim=-1, keepdim=True).values
            patch_scores_norm = (patch_scores - p_min) / (p_max - p_min).clamp(min=1e-8)

        # broadcast
        relevance_map_2d = torch.zeros(B, self.grid_size, self.grid_size, device=device)
        for p in range(self.num_patches):
            mask_p = patch_masks[p]
            relevance_map_2d[:, mask_p] = patch_scores_norm[:, p].unsqueeze(-1).expand(
                -1, mask_p.sum().item()
            )

        return relevance_map_2d.view(B, -1), patch_scores_norm
