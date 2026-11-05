"""Tests for the nominal Car4WD Ackermann steering constraint contract."""

import math
import unittest

from isaaclab_tasks.user.traction_protective_control.mpc import load_steering_constraints


class TestAckermannSteeringConstraints(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.constraints = load_steering_constraints()

    def test_geometric_center_bounds_match_current_vehicle(self) -> None:
        lower, upper = self.constraints.geometric_center_bounds_rad

        self.assertTrue(math.isclose(lower, -0.2438743453893955, abs_tol=1.0e-12))
        self.assertTrue(math.isclose(upper, 0.2438743453893955, abs_tol=1.0e-12))

    def test_effective_bounds_match_current_action_interface(self) -> None:
        lower, upper = self.constraints.center_bounds_rad

        self.assertTrue(math.isclose(lower, -0.243, abs_tol=1.0e-12))
        self.assertTrue(math.isclose(upper, 0.243, abs_tol=1.0e-12))

    def test_boundary_wheel_angles_respect_joint_limits(self) -> None:
        lower, upper = self.constraints.center_bounds_rad
        left_lower, right_lower = self.constraints.wheel_angles_rad(lower)
        left_upper, right_upper = self.constraints.wheel_angles_rad(upper)

        self.assertLessEqual(self.constraints.front_left_min_rad, left_lower)
        self.assertLessEqual(left_lower, self.constraints.front_left_max_rad)
        self.assertLessEqual(self.constraints.front_right_min_rad, right_lower)
        self.assertLessEqual(right_lower, self.constraints.front_right_max_rad)
        self.assertLessEqual(self.constraints.front_left_min_rad, left_upper)
        self.assertLessEqual(left_upper, self.constraints.front_left_max_rad)
        self.assertLessEqual(self.constraints.front_right_min_rad, right_upper)
        self.assertLessEqual(right_upper, self.constraints.front_right_max_rad)

    def test_normalized_bounds_use_full_current_action_range(self) -> None:
        lower, upper = self.constraints.normalized_bounds

        self.assertTrue(math.isclose(lower, -1.0, abs_tol=1.0e-12))
        self.assertTrue(math.isclose(upper, 1.0, abs_tol=1.0e-12))
        self.assertFalse(self.constraints.preserve_existing_ppo_action_mapping)

    def test_saturating_command_is_rejected_and_clipped(self) -> None:
        lower, upper = self.constraints.center_bounds_rad

        self.assertTrue(self.constraints.is_feasible(-0.2))
        self.assertTrue(self.constraints.is_feasible(0.2))
        self.assertFalse(self.constraints.is_feasible(-0.25))
        self.assertFalse(self.constraints.is_feasible(0.25))
        self.assertEqual(self.constraints.clip_center_angle(-0.25), lower)
        self.assertEqual(self.constraints.clip_center_angle(0.25), upper)

    def test_ackermann_mapping_is_odd_symmetric(self) -> None:
        positive_left, positive_right = self.constraints.wheel_angles_rad(0.2)
        negative_left, negative_right = self.constraints.wheel_angles_rad(-0.2)

        self.assertTrue(math.isclose(positive_left, -negative_right, abs_tol=1.0e-12))
        self.assertTrue(math.isclose(positive_right, -negative_left, abs_tol=1.0e-12))

    def test_center_angle_reconstruction_inverts_ackermann_mapping(self) -> None:
        for center_angle in (-0.2, -0.08, 0.0, 0.08, 0.2):
            left, right = self.constraints.wheel_angles_rad(center_angle)
            reconstructed = self.constraints.center_angle_from_wheel_angles_rad(left, right)
            self.assertTrue(math.isclose(reconstructed, center_angle, abs_tol=1.0e-12))


if __name__ == "__main__":
    unittest.main()
