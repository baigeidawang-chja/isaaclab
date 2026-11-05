"""Ackermann steering constraints for the nominal Car4WD MPC interface."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_STEERING_CONFIG_PATH = Path(__file__).parent / "config" / "nominal_car4wd_steering.json"


@dataclass(frozen=True)
class AckermannSteeringConstraints:
    """Physical steering bounds derived from Ackermann geometry and wheel-joint limits."""

    wheelbase_m: float
    track_width_m: float
    front_left_min_rad: float
    front_left_max_rad: float
    front_right_min_rad: float
    front_right_max_rad: float
    action_scale_rad: float
    action_offset_rad: float
    normalized_min: float
    normalized_max: float
    safety_margin_rad: float = 0.0
    preserve_existing_ppo_action_mapping: bool = False

    def __post_init__(self) -> None:
        values = (
            self.wheelbase_m,
            self.track_width_m,
            self.front_left_min_rad,
            self.front_left_max_rad,
            self.front_right_min_rad,
            self.front_right_max_rad,
            self.action_scale_rad,
            self.action_offset_rad,
            self.normalized_min,
            self.normalized_max,
            self.safety_margin_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Steering constraint parameters must all be finite.")
        if self.wheelbase_m <= 0.0 or self.track_width_m <= 0.0:
            raise ValueError("Ackermann wheelbase and track width must be positive.")
        if self.action_scale_rad <= 0.0:
            raise ValueError("Steering action scale must be positive.")
        if self.normalized_min >= self.normalized_max:
            raise ValueError("Normalized action minimum must be smaller than its maximum.")
        if not self.front_left_min_rad < 0.0 < self.front_left_max_rad:
            raise ValueError("Front-left steering limits must contain zero.")
        if not self.front_right_min_rad < 0.0 < self.front_right_max_rad:
            raise ValueError("Front-right steering limits must contain zero.")
        if self.safety_margin_rad < 0.0:
            raise ValueError("Steering safety margin cannot be negative.")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AckermannSteeringConstraints":
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported steering constraint schema version.")
        geometry = data["geometry"]
        joint_limits = data["steering_joint_limits_rad"]
        action = data["normalized_action_interface"]
        policy = data["constraint_policy"]
        return cls(
            wheelbase_m=float(geometry["wheelbase_m"]),
            track_width_m=float(geometry["track_width_m"]),
            front_left_min_rad=float(joint_limits["front_left"][0]),
            front_left_max_rad=float(joint_limits["front_left"][1]),
            front_right_min_rad=float(joint_limits["front_right"][0]),
            front_right_max_rad=float(joint_limits["front_right"][1]),
            action_scale_rad=float(action["steering_scale_rad"]),
            action_offset_rad=float(action["steering_offset_rad"]),
            normalized_min=float(action["normalized_min"]),
            normalized_max=float(action["normalized_max"]),
            safety_margin_rad=float(policy["safety_margin_rad"]),
            preserve_existing_ppo_action_mapping=bool(
                policy["preserve_existing_ppo_action_mapping"]
            ),
        )

    @staticmethod
    def _require_positive_limit(angle_rad: float, name: str) -> float:
        if not 0.0 < angle_rad < math.pi / 2.0:
            raise ValueError(f"{name} magnitude must be between zero and pi/2.")
        return angle_rad

    def _center_from_inner_wheel_limit(self, wheel_limit_rad: float) -> float:
        wheel_limit_rad = self._require_positive_limit(wheel_limit_rad, "Inner wheel limit")
        turn_radius_m = self.wheelbase_m / math.tan(wheel_limit_rad) + self.track_width_m / 2.0
        return math.atan(self.wheelbase_m / turn_radius_m)

    def _center_from_outer_wheel_limit(self, wheel_limit_rad: float) -> float:
        wheel_limit_rad = self._require_positive_limit(wheel_limit_rad, "Outer wheel limit")
        turn_radius_m = self.wheelbase_m / math.tan(wheel_limit_rad) - self.track_width_m / 2.0
        if turn_radius_m <= 0.0:
            raise ValueError("Outer wheel limit is outside the conventional Ackermann geometry range.")
        return math.atan(self.wheelbase_m / turn_radius_m)

    @property
    def geometric_center_bounds_rad(self) -> tuple[float, float]:
        """Center-steering bounds imposed by all four directional joint limits."""
        positive_max = min(
            self._center_from_inner_wheel_limit(self.front_left_max_rad),
            self._center_from_outer_wheel_limit(self.front_right_max_rad),
        )
        negative_magnitude = min(
            self._center_from_inner_wheel_limit(-self.front_right_min_rad),
            self._center_from_outer_wheel_limit(-self.front_left_min_rad),
        )
        lower = -negative_magnitude + self.safety_margin_rad
        upper = positive_max - self.safety_margin_rad
        if lower >= upper:
            raise ValueError("Steering safety margin removes the complete feasible interval.")
        return lower, upper

    @property
    def action_center_bounds_rad(self) -> tuple[float, float]:
        lower = self.action_offset_rad + self.action_scale_rad * self.normalized_min
        upper = self.action_offset_rad + self.action_scale_rad * self.normalized_max
        return min(lower, upper), max(lower, upper)

    @property
    def center_bounds_rad(self) -> tuple[float, float]:
        """Effective center-angle bounds shared by geometry and the action interface."""
        geometric_lower, geometric_upper = self.geometric_center_bounds_rad
        action_lower, action_upper = self.action_center_bounds_rad
        lower = max(geometric_lower, action_lower)
        upper = min(geometric_upper, action_upper)
        if lower >= upper:
            raise ValueError("Ackermann geometry and action interface have no feasible intersection.")
        return lower, upper

    @property
    def normalized_bounds(self) -> tuple[float, float]:
        lower, upper = self.center_bounds_rad
        return (
            (lower - self.action_offset_rad) / self.action_scale_rad,
            (upper - self.action_offset_rad) / self.action_scale_rad,
        )

    def wheel_angles_rad(self, center_angle_rad: float) -> tuple[float, float]:
        """Return front-left and front-right angles for a virtual center angle."""
        if not math.isfinite(center_angle_rad):
            raise ValueError("Center steering angle must be finite.")
        if abs(center_angle_rad) <= 1.0e-12:
            return 0.0, 0.0
        turn_radius_m = self.wheelbase_m / math.tan(center_angle_rad)
        left = math.atan(self.wheelbase_m / (turn_radius_m - self.track_width_m / 2.0))
        right = math.atan(self.wheelbase_m / (turn_radius_m + self.track_width_m / 2.0))
        return left, right

    def center_angle_from_wheel_angles_rad(self, left_rad: float, right_rad: float) -> float:
        """Reconstruct an equivalent center angle from measured left/right angles."""
        if not math.isfinite(left_rad) or not math.isfinite(right_rad):
            raise ValueError("Measured steering angles must be finite.")
        left_tangent = math.tan(left_rad)
        right_tangent = math.tan(right_rad)
        left_curvature = left_tangent / (
            self.wheelbase_m + self.track_width_m * left_tangent / 2.0
        )
        right_curvature = right_tangent / (
            self.wheelbase_m - self.track_width_m * right_tangent / 2.0
        )
        equivalent_curvature = 0.5 * (left_curvature + right_curvature)
        return math.atan(self.wheelbase_m * equivalent_curvature)

    def normalized_action(self, center_angle_rad: float) -> float:
        return (center_angle_rad - self.action_offset_rad) / self.action_scale_rad

    def is_feasible(self, center_angle_rad: float, tolerance: float = 1.0e-12) -> bool:
        if not math.isfinite(center_angle_rad):
            return False
        lower, upper = self.center_bounds_rad
        if center_angle_rad < lower - tolerance or center_angle_rad > upper + tolerance:
            return False
        left, right = self.wheel_angles_rad(center_angle_rad)
        return (
            self.front_left_min_rad - tolerance <= left <= self.front_left_max_rad + tolerance
            and self.front_right_min_rad - tolerance <= right <= self.front_right_max_rad + tolerance
        )

    def clip_center_angle(self, center_angle_rad: float) -> float:
        if not math.isfinite(center_angle_rad):
            raise ValueError("Center steering angle must be finite.")
        lower, upper = self.center_bounds_rad
        return min(max(center_angle_rad, lower), upper)

    def summary(self) -> dict[str, Any]:
        lower, upper = self.center_bounds_rad
        normalized_lower, normalized_upper = self.normalized_bounds
        left_at_lower, right_at_lower = self.wheel_angles_rad(lower)
        left_at_upper, right_at_upper = self.wheel_angles_rad(upper)
        return {
            "center_bounds_rad": [lower, upper],
            "center_bounds_deg": [math.degrees(lower), math.degrees(upper)],
            "normalized_bounds": [normalized_lower, normalized_upper],
            "wheel_angles_at_lower_rad": [left_at_lower, right_at_lower],
            "wheel_angles_at_upper_rad": [left_at_upper, right_at_upper],
            "ppo_action_mapping_preserved": self.preserve_existing_ppo_action_mapping,
        }


def load_steering_constraints(
    path: str | Path = DEFAULT_STEERING_CONFIG_PATH,
) -> AckermannSteeringConstraints:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as config_file:
        data = json.load(config_file)
    return AckermannSteeringConstraints.from_mapping(data)
