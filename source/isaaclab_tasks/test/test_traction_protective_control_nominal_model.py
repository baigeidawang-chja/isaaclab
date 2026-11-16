"""Tests for the identified nominal Car4WD model implementation."""

import math
import unittest

from isaaclab_tasks.user.traction_protective_control.mpc.nominal_model import (
    FirstOrderDelayModel,
    FirstOrderRegion,
    GainScheduledFirstOrderModel,
    NominalCar4WDModel,
    load_nominal_vehicle_model,
)


class TestFirstOrderDelayModel(unittest.TestCase):
    def test_constant_input_converges_to_dc_gain(self) -> None:
        model = FirstOrderDelayModel(sample_time_s=0.05, delay_steps=0, a=0.8, b=0.3)
        output = model.simulate([2.0] * 200, initial_output=0.0)

        self.assertAlmostEqual(output[-1], model.dc_gain * 2.0, places=8)
        self.assertAlmostEqual(model.dc_gain, 1.5)

    def test_delay_uses_supplied_input_history(self) -> None:
        model = FirstOrderDelayModel(sample_time_s=0.05, delay_steps=2, a=0.5, b=0.5)
        output = model.simulate([1.0, 1.0, 1.0], initial_output=0.0, input_history=[0.0, 0.0])

        self.assertEqual(output, [0.0, 0.0, 0.5])


class TestGainScheduledFirstOrderModel(unittest.TestCase):
    def test_region_selects_local_linear_model(self) -> None:
        low = FirstOrderDelayModel(0.05, 0, 0.5, 0.75)
        high = FirstOrderDelayModel(0.05, 0, 0.8, 0.2)
        model = GainScheduledFirstOrderModel(
            (
                FirstOrderRegion(0.0, 0.5, low),
                FirstOrderRegion(0.5, 1.0, high),
            )
        )

        self.assertIs(model.model_for(0.25), low)
        self.assertIs(model.model_for(0.75), high)
        self.assertAlmostEqual(model.next_output(0.0, 0.25), 0.1875)
        with self.assertRaises(ValueError):
            model.model_for(1.1)


class TestNominalCar4WDModel(unittest.TestCase):
    def test_ackermann_yaw_rate_is_odd_in_steering(self) -> None:
        first_order = FirstOrderDelayModel(0.05, 0, 0.8, 0.2)
        model = NominalCar4WDModel(0.05, 0.407, first_order, first_order, yaw_gain=0.95)

        positive = model.yaw_rate_rad_s(0.8, 0.2)
        negative = model.yaw_rate_rad_s(0.8, -0.2)
        self.assertTrue(math.isclose(positive, -negative, abs_tol=1.0e-12))

    def test_frozen_identified_model_loads(self) -> None:
        model = load_nominal_vehicle_model()

        self.assertIsInstance(model.speed, GainScheduledFirstOrderModel)
        self.assertEqual(model.speed.delay_steps, 0)
        self.assertEqual(model.steering.delay_steps, 0)
        self.assertAlmostEqual(model.yaw_gain, 0.9413673453849162)


if __name__ == "__main__":
    unittest.main()
