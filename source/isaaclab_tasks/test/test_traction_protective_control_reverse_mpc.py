"""Signed-speed contracts, tested without Isaac Sim or a policy."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

from isaaclab_tasks.user.traction_protective_control.mpc import (
    CarVWActionAdapter,
    DEFAULT_NOMINAL_MODEL_CONFIG_PATH,
    NominalMPC,
    NominalMPCReference,
    NominalMPCState,
    load_nominal_vehicle_model,
)


class TestBidirectionalMPC(unittest.TestCase):
    def setUp(self):
        self.controller = NominalMPC()

    def test_reverse_departure_obeys_rate_limit(self):
        result = self.controller.solve([0.0, 0.0, 0.0], [-0.5, 0.0], [0.0, 0.0])
        self.assertTrue(result.success, result.fallback_reason)
        self.assertLess(result.control[0], 0.0)
        self.assertGreaterEqual(result.control[0], -0.100002)
        self.assertLess(result.linearization_speed_m_s, 0.0)

    def test_reverse_heading_requires_opposite_steering(self):
        for speed in (-0.5, 0.5):
            for error in (-0.2, 0.2):
                with self.subTest(speed=speed, error=error):
                    result = self.controller.solve([error, speed, 0.0], [speed, 0.0], [speed, 0.0])
                    self.assertTrue(result.success, result.fallback_reason)
                    self.assertGreater(result.control[1] * speed * error, 0.0)
                    self.assertLess(abs(result.predicted_states[-1, 0]), abs(error))

    def test_braking_direction_follows_motion_not_new_reference(self):
        for speed, target in ((0.4, -0.5), (-0.4, 0.5)):
            result = self.controller.solve([0.1, speed, 0.0], [target, 0.0], [speed, 0.0])
            self.assertTrue(result.success, result.fallback_reason)
            self.assertGreater(result.linearization_speed_m_s * speed, 0.0)
            self.assertLessEqual(abs(result.control[0] - speed), 0.100002)

    def test_reverse_prediction_and_constraints(self):
        state = NominalMPCState(0.15, -0.45, -0.03)
        previous = np.array([-0.5, -0.02])
        result = self.controller.solve(state, NominalMPCReference(-0.5), previous)
        self.assertTrue(result.success, result.fallback_reason)
        model = self.controller.model.speed.regions[result.speed_region_index].model
        a, b, c = self.controller.linearized_dynamics(
            model, result.linearization_speed_m_s, result.linearization_steering_rad
        )
        for step, control in enumerate(result.predicted_controls):
            np.testing.assert_allclose(result.predicted_states[step + 1],
                                       a @ result.predicted_states[step] + b @ control + c, atol=2e-5)
        controls = result.predicted_controls
        self.assertTrue(np.all(controls >= np.array(self.controller.config.control_lower) - 2e-5))
        self.assertTrue(np.all(controls <= np.array(self.controller.config.control_upper) + 2e-5))
        increments = np.diff(np.vstack([previous, controls]), axis=0)
        self.assertTrue(np.all(np.abs(increments) <= 0.10002))

    def test_zero_reference_at_rest_does_not_invent_turn_authority(self):
        result = self.controller.solve([0.2, 0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
        self.assertTrue(result.success, result.fallback_reason)
        self.assertEqual(result.linearization_speed_m_s, 0.0)
        np.testing.assert_allclose(result.control, [0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(result.predicted_states[:, 0], 0.2, atol=1e-6)

    def test_reference_bounds_are_distinct_from_actuator_bounds(self):
        self.assertEqual(self.controller.config.reference_speed_bounds, (-0.5, 1.2))
        self.assertEqual(self.controller.config.control_lower[0], -1.5)
        self.assertEqual(self.controller.config.control_upper[0], 1.5)
        for target in (-0.51, 1.21):
            result = self.controller.solve([0.0, 0.0, 0.0], [target, 0.0])
            self.assertFalse(result.success)
            self.assertEqual(result.status, "invalid_input")

    def test_reverse_invalid_input_fallback_brakes_toward_zero(self):
        result = self.controller.solve([np.nan, -0.5, 0.0], [-0.5, 0.0], [-0.5, -0.1])
        self.assertTrue(result.fallback_used)
        np.testing.assert_allclose(result.control, [-0.4, 0.0], atol=1e-12)

    def test_default_model_has_independent_validation_certificate(self):
        path = Path(DEFAULT_NOMINAL_MODEL_CONFIG_PATH)
        model = json.loads(path.read_text())
        validation = json.loads(path.with_name("nominal_car4wd_bidirectional_validation.json").read_text())
        self.assertEqual(validation["parameters_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertIs(validation["parameters_refit_or_selected_on_validation"], False)
        self.assertIs(model["provenance"]["validation_used_for_parameter_selection"], False)
        self.assertNotEqual(validation["validation_csv"], model["identification"]["source_csv"])
        self.assertNotEqual(validation["validation_sha256"], model["identification"]["source_sha256"])
        self.assertEqual(self.controller.config.model_config_path.resolve(), path.resolve())

    def test_old_forward_model_cannot_silently_enable_reverse(self):
        old = Path(DEFAULT_NOMINAL_MODEL_CONFIG_PATH).with_name("nominal_car4wd_model.json")
        with self.assertRaisesRegex(ValueError, "loaded identified model"):
            NominalMPC(model=load_nominal_vehicle_model(old))

    def test_config_cannot_exceed_identified_or_action_bounds(self):
        with self.assertRaises(ValueError):
            replace(self.controller.config, control_lower=(-1.6, -0.22))

    def test_signed_action_mapping_is_not_applied_twice(self):
        adapter = CarVWActionAdapter()
        for speed in (-1.5, -0.5, 0.0, 1.2, 1.5):
            physical = np.array([speed, -0.22])
            normalized = adapter.to_normalized(physical)
            self.assertAlmostEqual(normalized[0], speed / 1.5)
            np.testing.assert_allclose(adapter.to_physical(normalized), physical, atol=1e-12)

    def test_model_rollout_crosses_zero_in_both_directions(self):
        state = np.array([0.0, 0.5, 0.0])
        previous = np.array([0.55, 0.0])
        for target in (-0.5, 0.5):
            for _ in range(80):
                result = self.controller.solve(state, [target, 0.0], previous)
                self.assertTrue(result.success, result.fallback_reason)
                model = self.controller.model
                state = np.array([
                    state[0] - model.sample_time_s * model.yaw_rate_rad_s(state[1], state[2]),
                    model.speed.next_output(state[1], float(result.control[0])),
                    model.steering.next_output(state[2], float(result.control[1])),
                ])
                previous = result.control
            self.assertAlmostEqual(state[1], target, delta=0.02)


if __name__ == "__main__":
    unittest.main()
