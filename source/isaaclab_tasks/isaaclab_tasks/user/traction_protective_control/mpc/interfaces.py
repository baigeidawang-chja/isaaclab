"""Physical-unit interfaces shared by nominal MPC backends."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NominalMPCState:
    """Measured MPC state in physical units."""

    heading_error_rad: float
    body_speed_m_s: float
    center_steering_rad: float

    def as_array(self) -> np.ndarray:
        values = np.asarray(
            [self.heading_error_rad, self.body_speed_m_s, self.center_steering_rad],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Nominal MPC state must be finite.")
        return values


@dataclass(frozen=True)
class NominalMPCReference:
    """Planner reference represented as speed and world-heading error."""

    speed_m_s: float
    heading_error_rad: float = 0.0

    def as_array(self) -> np.ndarray:
        values = np.asarray([self.heading_error_rad, self.speed_m_s, 0.0], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("Nominal MPC reference must be finite.")
        return values


@dataclass(frozen=True)
class NominalMPCResult:
    """One nominal MPC solve result; control is always in physical units."""

    control: np.ndarray
    predicted_states: np.ndarray
    predicted_controls: np.ndarray
    success: bool
    status: str
    status_value: int | None
    solve_time_s: float
    solver_run_time_s: float | None
    iterations: int | None
    objective: float | None
    primal_residual: float | None
    dual_residual: float | None
    warm_start_used: bool
    fallback_used: bool
    fallback_reason: str | None
    speed_region_index: int | None
    linearization_speed_m_s: float | None
    linearization_steering_rad: float | None

    @property
    def speed_target_m_s(self) -> float:
        return float(self.control[0])

    @property
    def steering_target_rad(self) -> float:
        return float(self.control[1])


def wrap_to_pi(angle_rad: float) -> float:
    if not math.isfinite(angle_rad):
        raise ValueError("Angle must be finite.")
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))
