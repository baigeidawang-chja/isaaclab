"""Summarize recorded nominal MPC closed-loop runs without Isaac Sim."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean


def _rmse(values: list[float]) -> float:
    return math.sqrt(mean(value * value for value in values))


def _settling_time(rows: list[dict], key: str, tolerance: float) -> float | None:
    errors = [abs(float(row[key])) for row in rows]
    start = float(rows[0]["command_time_s"])
    for index, error in enumerate(errors):
        if error <= tolerance and all(value <= tolerance for value in errors[index:]):
            return float(rows[index]["sample_time_s"]) - start
    return None


def _read_run(run_dir: Path) -> tuple[dict, dict, dict, list[dict]]:
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    model_path = Path(metadata["model_config"])
    config_path = Path(metadata["mpc_config"])
    steering_path = Path(metadata["steering_config"])
    for label, path in (("model", model_path), ("mpc", config_path), ("steering", steering_path)):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata[f"{label}_config_sha256"]:
            raise ValueError(f"{label} config changed since {run_dir} was recorded")
    if "robot_usd_sha256" in metadata:
        robot_usd_path = Path(metadata["robot_usd_path"])
        if hashlib.sha256(robot_usd_path.read_bytes()).hexdigest() != metadata["robot_usd_sha256"]:
            raise ValueError(f"Robot USD changed since {run_dir} was recorded")
    model = json.loads(model_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with (run_dir / "steps.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows or len(rows) != metadata["steps"]:
        raise ValueError(f"Missing or incomplete step records in {run_dir}")
    required = (
        "command_time_s", "sample_time_s", "desired_speed_m_s", "desired_heading_rad",
        "actual_body_vx_m_s", "actual_heading_rad", "actual_heading_error_rad",
        "actual_equivalent_center_steering_rad", "predicted_next_body_speed_m_s",
        "predicted_next_heading_error_rad", "predicted_next_center_steering_rad",
        "mpc_speed_target_m_s", "physical_target_speed_m_s", "executed_speed_m_s",
        "mpc_steering_target_rad", "physical_target_steering_rad", "executed_steering_rad",
        "solve_time_s",
    )
    for row in rows:
        if not all(math.isfinite(float(row[key])) for key in required):
            raise ValueError(f"Non-finite MPC record in {run_dir}")
    return metadata, model, config, rows


def _stage_metrics(
    stage_rows: list[dict], previous_command: tuple[float, float] | None,
    speed_tolerance: float, heading_tolerance: float, steady_window_s: float,
    speed_upper_bound: float, nominal_max_speed: float,
    speed_lower_bound: float, nominal_min_speed: float,
) -> dict:
    first = stage_rows[0]
    speed = float(first["desired_speed_m_s"])
    heading = float(first["desired_heading_rad"])
    speed_error = [speed - float(row["actual_body_vx_m_s"]) for row in stage_rows]
    heading_error = [float(row["actual_heading_error_rad"]) for row in stage_rows]
    end = float(stage_rows[-1]["sample_time_s"])
    steady = [row for row in stage_rows if float(row["sample_time_s"]) > end - steady_window_s]
    speed_step = None if previous_command is None else speed - previous_command[0]
    heading_step = None if previous_command is None else math.atan2(
        math.sin(heading - previous_command[1]), math.cos(heading - previous_command[1])
    )

    def overshoot(actual_key: str, target: float, step: float | None, tolerance: float) -> float | None:
        if step is None or abs(step) <= tolerance:
            return None
        direction = 1.0 if step > 0.0 else -1.0
        return max(0.0, max(direction * (float(row[actual_key]) - target) for row in stage_rows))

    def prediction_rmse(predicted: str, actual: str) -> float:
        return _rmse([float(row[predicted]) - float(row[actual]) for row in stage_rows])

    wheel_error = [
        abs(float(row[f"target_wheel_speed_{wheel}_rad_s"]) - float(row[f"actual_wheel_speed_{wheel}_rad_s"]))
        for row in steady
        for wheel in ("front_left", "front_right", "rear_left", "rear_right")
    ]
    wheel_names = ("front_left", "front_right", "rear_left", "rear_right")
    torque_logged = "applied_wheel_torque_front_left_nm" in first
    torque_pairs = [
        (float(row[f"computed_wheel_torque_{wheel}_nm"]), float(row[f"applied_wheel_torque_{wheel}_nm"]))
        for row in steady for wheel in wheel_names
    ] if torque_logged else []
    negative_wheel_fraction = mean(
        float(row[f"actual_wheel_speed_{wheel}_rad_s"]) < 0.0
        for row in steady for wheel in wheel_names
    )
    return {
        "stage_index": int(first["stage_index"]),
        "samples": len(stage_rows),
        "desired_speed_m_s": speed,
        "desired_heading_rad": heading,
        "speed_mae_m_s": mean(map(abs, speed_error)),
        "speed_rmse_m_s": _rmse(speed_error),
        "heading_mae_rad": mean(map(abs, heading_error)),
        "heading_rmse_rad": _rmse(heading_error),
        "steady_speed_bias_m_s": mean(speed - float(row["actual_body_vx_m_s"]) for row in steady),
        "steady_heading_bias_rad": mean(float(row["actual_heading_error_rad"]) for row in steady),
        "speed_overshoot_m_s": overshoot("actual_body_vx_m_s", speed, speed_step, speed_tolerance),
        "heading_overshoot_rad": overshoot("actual_heading_rad", heading, heading_step, heading_tolerance),
        "speed_settling_time_s": _settling_time(stage_rows, "speed_error", speed_tolerance),
        "heading_settling_time_s": _settling_time(stage_rows, "actual_heading_error_rad", heading_tolerance),
        "one_step_speed_rmse_m_s": prediction_rmse("predicted_next_body_speed_m_s", "actual_body_vx_m_s"),
        "one_step_heading_rmse_rad": prediction_rmse("predicted_next_heading_error_rad", "actual_heading_error_rad"),
        "one_step_steering_rmse_rad": prediction_rmse("predicted_next_center_steering_rad", "actual_equivalent_center_steering_rad"),
        "steady_wheel_target_error_rad_s": mean(wheel_error),
        "steady_negative_wheel_speed_fraction": negative_wheel_fraction,
        "steady_applied_wheel_torque_abs_nm": mean(abs(applied) for _, applied in torque_pairs) if torque_pairs else None,
        "steady_wheel_torque_clip_fraction": mean(
            abs(computed) > abs(applied) + 1.0e-4 for computed, applied in torque_pairs
        ) if torque_pairs else None,
        "speed_target_upper_fraction": mean(
            float(row["mpc_speed_target_m_s"]) >= speed_upper_bound - 1.0e-4 for row in stage_rows
        ),
        "speed_target_lower_fraction": mean(
            float(row["mpc_speed_target_m_s"]) <= speed_lower_bound + 1.0e-4 for row in stage_rows
        ),
        "nominal_min_steady_speed_m_s": nominal_min_speed,
        "reference_below_nominal_min": speed < nominal_min_speed - 1.0e-6,
        "nominal_max_steady_speed_m_s": nominal_max_speed,
        "reference_above_nominal_max": speed > nominal_max_speed + 1.0e-6,
    }


def analyze(run_dir: Path, speed_tolerance: float, heading_tolerance: float, steady_window_s: float) -> dict:
    metadata, model, config, rows = _read_run(run_dir)
    speed_lower, speed_upper = map(float, config["constraints"]["effective_mpc_control_bounds"]["speed_target_m_s"])
    regions = model["dynamics"]["longitudinal_speed"]["regions"]
    nominal_max = max(
        float(region["dc_gain"]) * min(speed_upper, float(region["input_range_m_s"][1]))
        for region in regions if float(region["input_range_m_s"][0]) <= speed_upper
    )
    nominal_min = min(
        float(region["dc_gain"]) * max(speed_lower, float(region["input_range_m_s"][0]))
        for region in regions if float(region["input_range_m_s"][1]) >= speed_lower
    )
    stages = []
    previous_command = None
    for index in sorted({int(row["stage_index"]) for row in rows}):
        stage_rows = [row for row in rows if int(row["stage_index"]) == index]
        for row in stage_rows:
            row["speed_error"] = float(row["desired_speed_m_s"]) - float(row["actual_body_vx_m_s"])
        stage = _stage_metrics(stage_rows, previous_command, speed_tolerance, heading_tolerance,
                               steady_window_s, speed_upper, nominal_max, speed_lower, nominal_min)
        stages.append(stage)
        previous_command = (stage["desired_speed_m_s"], stage["desired_heading_rad"])

    summary = {
        "profile": metadata["profile"],
        "model_sha256": metadata["model_config_sha256"],
        "samples": len(rows),
        "solver_status_counts": dict(Counter(row["solver_status"] for row in rows)),
        "fallback_count": sum(int(row["fallback_used"]) for row in rows),
        "termination_reason": metadata["termination_reason"],
        "mean_solve_time_ms": 1000.0 * mean(float(row["solve_time_s"]) for row in rows),
        "max_solve_time_ms": 1000.0 * max(float(row["solve_time_s"]) for row in rows),
        "control_period_ms": 1000.0 * float(metadata["step_dt_s"]),
        "max_target_mapping_error": max(
            abs(float(row[f"mpc_{axis}_target_{unit}"]) - float(row[f"physical_target_{axis}_{unit}"]))
            for row in rows for axis, unit in (("speed", "m_s"), ("steering", "rad"))
        ),
        "max_rate_limiter_difference": max(
            abs(float(row[f"physical_target_{axis}_{unit}"]) - float(row[f"executed_{axis}_{unit}"]))
            for row in rows for axis, unit in (("speed", "m_s"), ("steering", "rad"))
        ),
        "stages": stages,
    }
    (run_dir / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "stage_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(stages[0]))
        writer.writeheader()
        writer.writerows(stages)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--speed-tolerance", type=float, default=0.05)
    parser.add_argument("--heading-tolerance", type=float, default=0.05)
    parser.add_argument("--steady-window-s", type=float, default=1.0)
    args = parser.parse_args()
    if min(args.speed_tolerance, args.heading_tolerance, args.steady_window_s) <= 0.0:
        parser.error("Tolerances and steady window must be positive")
    for directory in args.run_dirs:
        result = analyze(directory.expanduser().resolve(), args.speed_tolerance,
                         args.heading_tolerance, args.steady_window_s)
        print(f"{directory}: {result['samples']} steps, {result['fallback_count']} fallbacks, "
              f"max solve {result['max_solve_time_ms']:.2f} ms")


if __name__ == "__main__":
    main()
