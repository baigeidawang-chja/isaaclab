"""Metric unit fixtures only, not simulated experimental evidence."""

import unittest

import numpy as np
import pandas as pd

from analyze_nominal_mpc_longitudinal import confirmed_crossing, last_window, stage_metrics, WHEELS


def fixture(speed=0.3, count=160):
    frame = pd.DataFrame({
        "step_dt_s": np.full(count, 0.05), "command_time_s": np.arange(count) * 0.05,
        "sample_time_s": (np.arange(count) + 1) * 0.05, "stage_index": np.zeros(count),
        "desired_speed_m_s": np.full(count, speed), "desired_heading_rad": np.zeros(count),
        "actual_body_vx_m_s": np.full(count, speed), "actual_heading_rad": np.zeros(count),
        "actual_heading_error_rad": np.zeros(count), "pre_body_speed_m_s": np.full(count, speed),
        "previous_executed_speed_m_s": np.full(count, speed), "pre_heading_rad": np.zeros(count),
        "executed_speed_m_s": np.full(count, speed), "mpc_speed_target_m_s": np.full(count, speed),
        "predicted_next_body_speed_m_s": np.full(count, speed),
        "predicted_next_heading_error_rad": np.zeros(count), "predicted_next_center_steering_rad": np.zeros(count),
        "actual_equivalent_center_steering_rad": np.zeros(count),
    })
    for wheel in WHEELS:
        for prefix in ("target_wheel_speed", "actual_wheel_speed"):
            frame[f"{prefix}_{wheel}_rad_s"] = speed / 0.065
        for prefix in ("computed_wheel_torque", "applied_wheel_torque"):
            frame[f"{prefix}_{wheel}_nm"] = 0.0
    for side in ("front_left", "front_right"):
        frame[f"actual_steering_{side}_rad"] = 0.0
    return frame


class MetricsTest(unittest.TestCase):
    def test_open_left_steady_window(self):
        frame = fixture()
        frame["sample_time_s"] += 0.1
        tail = last_window(frame)
        self.assertEqual(len(tail), 20)
        self.assertAlmostEqual(tail.sample_time_s.iloc[0], 7.15)
        self.assertAlmostEqual(tail.sample_time_s.iloc[-1], 8.1)

    def test_three_samples_not_single_noise_spike(self):
        frame = fixture(count=8)
        frame["actual_body_vx_m_s"] = [-.1, .03, -.01, .02, .021, .025, .03, .04]
        self.assertAlmostEqual(confirmed_crossing(frame, "actual_body_vx_m_s", 1, "sample_time_s"), .25)
        self.assertIsNone(confirmed_crossing(frame, "actual_body_vx_m_s", -1, "sample_time_s"))

    def test_clock_difference(self):
        frame = fixture(count=8)
        frame["executed_speed_m_s"] = [-.1, .1, .2, .3, .3, .3, .3, .3]
        frame["actual_body_vx_m_s"] = [-.1, -.01, .03, .1, .2, .3, .3, .3]
        a = confirmed_crossing(frame, "executed_speed_m_s", 1, "command_time_s")
        b = confirmed_crossing(frame, "actual_body_vx_m_s", 1, "sample_time_s")
        self.assertAlmostEqual(a, .05)
        self.assertAlmostEqual(b - a, .1)

    def test_settling_requires_remainder_in_band(self):
        frame = fixture()
        frame.loc[150, "actual_body_vx_m_s"] = 0.4
        metrics = stage_metrics(frame, None, 160, -1.2, 1.23)
        self.assertAlmostEqual(metrics["speed_settling_time_s"], 7.6)
        self.assertEqual(metrics["trailing_in_band_samples"], 9)
        self.assertAlmostEqual(metrics["trailing_in_band_span_s"], .4)
        self.assertIsNone(metrics["speed_overshoot_m_s"])

    def test_unsettled_and_missing_projection(self):
        frame = fixture()
        frame.loc[159, "actual_body_vx_m_s"] = 0.4
        metrics = stage_metrics(frame, None, 160, -1.2, 1.23)
        self.assertIsNone(metrics["speed_settling_time_s"])
        self.assertFalse(metrics["world_projection_available"])
        self.assertIsNone(metrics["projected_speed_rmse_m_s"])

    def test_reverse_projection_signed_and_downstep_overshoot(self):
        frame = fixture(-0.5)
        frame["actual_world_vx_m_s"] = -.5
        frame["actual_world_vy_m_s"] = 0.0
        frame.loc[0, "actual_body_vx_m_s"] = -.6
        metrics = stage_metrics(frame, 0.5, 160, -1.2, 1.23)
        self.assertEqual(metrics["projected_speed_mae_m_s"], 0)
        self.assertAlmostEqual(metrics["speed_overshoot_m_s"], .1)

    def test_stop_drift_is_distance_not_speed(self):
        frame = fixture(0)
        frame["actual_world_x_m"] = frame.sample_time_s * .001
        frame["pre_world_x_m"] = frame.command_time_s * .001
        frame["actual_world_y_m"] = frame["pre_world_y_m"] = 0
        metrics = stage_metrics(frame, .5, 160, -1.2, 1.23)
        self.assertAlmostEqual(metrics["stop_tail_signed_world_x_drift_m"], .001)
        self.assertAlmostEqual(metrics["stop_tail_path_length_m"], .001)
        self.assertIsNone(metrics["actual_crossing_time_s"])


if __name__ == "__main__":
    unittest.main()
