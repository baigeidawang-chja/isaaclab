"""Boundary adapter between physical MPC controls and normalized ActionTerm inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CarVWActionAdapter:
    """Apply the existing RateLimitedCarVWAction affine mapping exactly once."""

    scale: tuple[float, float] = (1.5, 0.24)
    offset: tuple[float, float] = (0.0, 0.0)

    def to_normalized(self, physical_control: np.ndarray) -> np.ndarray:
        control = np.asarray(physical_control, dtype=np.float64)
        if control.shape != (2,) or not np.all(np.isfinite(control)):
            raise ValueError("Physical CarVW control must be a finite two-vector.")
        normalized = (control - np.asarray(self.offset)) / np.asarray(self.scale)
        if np.any(normalized < -1.0 - 1.0e-9) or np.any(normalized > 1.0 + 1.0e-9):
            raise ValueError("Physical MPC control exceeds the ActionTerm affine range.")
        return normalized

    def to_physical(self, normalized_action: np.ndarray) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float64)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Normalized CarVW action must be a finite two-vector.")
        return np.clip(action, -1.0, 1.0) * np.asarray(self.scale) + np.asarray(self.offset)
