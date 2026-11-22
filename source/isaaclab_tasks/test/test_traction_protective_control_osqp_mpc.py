"""Pure-Python regression tests for the identified-model OSQP nominal MPC."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from isaaclab_tasks.user.traction_protective_control.mpc import (
    CarVWActionAdapter,
    NominalMPC,
    NominalMPCReference,
    NominalMPCState,
)


class TestOSQPNominalMPC(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = NominalMPC()

    def test_equilibrium_at_rest_is_held(self) -> None:
        result = self.controller.solve(NominalMPCState(0.0, 0.0, 0.0), NominalMPCReference(0.0))

        self.assertTrue(result.success, result.fallback_reason)
        np.testing.assert_allclose(result.control, [0.0, 0.0], atol=1.0e-7)

    def test_speed_step_respects_first_rate_limit(self) -> None:
        result = self.controller.solve(NominalMPCState(0.0, 0.0, 0.0), NominalMPCReference(0.8))

        self.assertTrue(result.success, result.fallback_reason)
        self.assertGreater(result.speed_target_m_s, 0.0)
        self.assertLessEqual(result.speed_target_m_s, 0.1 + 1.0e-6)
        self.assertAlmostEqual(result.steering_target_rad, 0.0, places=6)

    def test_heading_error_uses_symmetric_steering_direction(self) -> None:
        positive = self.controller.solve(
            NominalMPCState(0.25, 0.8, 0.0), NominalMPCReference(0.8), [0.6, 0.0]
        )
        self.controller.reset([0.6, 0.0])
        negative = self.controller.solve(
            NominalMPCState(-0.25, 0.8, 0.0), NominalMPCReference(0.8), [0.6, 0.0]
        )

        self.assertTrue(positive.success and negative.success)
        self.assertGreater(positive.steering_target_rad, 0.0)
        self.assertLess(negative.steering_target_rad, 0.0)
        self.assertAlmostEqual(
            positive.steering_target_rad, -negative.steering_target_rad, delta=2.0e-5
        )

    def test_all_predicted_controls_obey_absolute_and_rate_constraints(self) -> None:
        previous = np.asarray([0.4, -0.05])
        result = self.controller.solve(
            NominalMPCState(1.2, 0.4, -0.05), NominalMPCReference(1.2), previous
        )

        self.assertTrue(result.success, result.fallback_reason)
        lower = np.asarray(self.controller.config.control_lower)
        upper = np.asarray(self.controller.config.control_upper)
        self.assertTrue(np.all(result.predicted_controls >= lower - 2.0e-5))
        self.assertTrue(np.all(result.predicted_controls <= upper + 2.0e-5))
        differences = np.vstack([result.predicted_controls[0] - previous, np.diff(result.predicted_controls, axis=0)])
        max_change = np.asarray(self.controller.config.control_rate_per_s) * self.controller.control_period_s
        self.assertTrue(np.all(np.abs(differences) <= max_change + 2.0e-5))

    def test_first_prediction_matches_frozen_linear_model(self) -> None:
        state = NominalMPCState(0.1, 0.6, 0.02)
        previous = np.asarray([0.5, 0.01])
        result = self.controller.solve(state, NominalMPCReference(0.8), previous)

        self.assertTrue(result.success, result.fallback_reason)
        speed_model = self.controller.model.speed.regions[result.speed_region_index].model
        a, b, c = self.controller.linearized_dynamics(
            speed_model,
            result.linearization_speed_m_s,
            result.linearization_steering_rad,
        )
        expected = a @ state.as_array() + b @ result.predicted_controls[0] + c
        np.testing.assert_allclose(result.predicted_states[1], expected, atol=2.0e-5)

    def test_second_solve_uses_shifted_warm_start(self) -> None:
        state = NominalMPCState(0.1, 0.5, 0.0)
        reference = NominalMPCReference(0.8)
        first = self.controller.solve(state, reference, [0.4, 0.0])
        second = self.controller.solve(state, reference, first.control)

        self.assertTrue(first.success and second.success)
        self.assertFalse(first.warm_start_used)
        self.assertTrue(second.warm_start_used)

    def test_nonfinite_input_uses_rate_limited_fallback(self) -> None:
        result = self.controller.solve([float("nan"), 0.5, 0.0], [0.8, 0.0], [0.4, 0.1])

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.status, "invalid_input")
        np.testing.assert_allclose(result.control, [0.3, 0.0], atol=1.0e-12)

    def test_solver_infeasible_status_uses_fallback(self) -> None:
        fake = SimpleNamespace(
            x=None,
            info=SimpleNamespace(
                status="primal infeasible",
                status_val=-3,
                run_time=0.001,
                iter=25,
                obj_val=1.0e30,
                pri_res=1.0,
                dua_res=1.0,
            ),
        )
        with patch.object(self.controller, "_run_osqp", return_value=fake):
            result = self.controller.solve(
                NominalMPCState(0.0, 0.4, 0.0), NominalMPCReference(0.8), [0.4, 0.05]
            )

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.status, "primal infeasible")
        np.testing.assert_allclose(result.control, [0.3, 0.0], atol=1.0e-12)

    def test_solver_exception_uses_fallback(self) -> None:
        with patch.object(self.controller, "_run_osqp", side_effect=RuntimeError("backend failed")):
            result = self.controller.solve(
                NominalMPCState(0.0, 0.4, 0.0), NominalMPCReference(0.8), [0.4, 0.05]
            )

        self.assertFalse(result.success)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.status, "solver_exception")
        self.assertIn("RuntimeError", result.fallback_reason)
        np.testing.assert_allclose(result.control, [0.3, 0.0], atol=1.0e-12)

    def test_action_adapter_is_exact_inverse_in_physical_range(self) -> None:
        adapter = CarVWActionAdapter()
        physical = np.asarray([1.2, -0.22])

        normalized = adapter.to_normalized(physical)
        np.testing.assert_allclose(adapter.to_physical(normalized), physical, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
