"""Check and print the nominal MPC Ackermann steering constraint contract."""

from __future__ import annotations

import argparse
import json

from isaaclab_tasks.user.traction_protective_control.mpc import (
    DEFAULT_STEERING_CONFIG_PATH,
    load_steering_constraints,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_STEERING_CONFIG_PATH))
    parser.add_argument(
        "--center-angle",
        action="append",
        type=float,
        dest="center_angles",
        help="Additional virtual center angle in radians to check.",
    )
    args = parser.parse_args()

    constraints = load_steering_constraints(args.config)
    lower, upper = constraints.center_bounds_rad
    samples = [0.0, -0.2, 0.2, lower, upper, -0.25, 0.25]
    if args.center_angles:
        samples.extend(args.center_angles)

    result = constraints.summary()
    result["config"] = args.config
    result["samples"] = []
    for center_angle in samples:
        left, right = constraints.wheel_angles_rad(center_angle)
        result["samples"].append(
            {
                "center_angle_rad": center_angle,
                "normalized_action": constraints.normalized_action(center_angle),
                "front_left_rad": left,
                "front_right_rad": right,
                "feasible": constraints.is_feasible(center_angle),
            }
        )

    if not constraints.is_feasible(lower) or not constraints.is_feasible(upper):
        raise RuntimeError("Derived steering boundary is not feasible.")
    if constraints.is_feasible(-0.25) or constraints.is_feasible(0.25):
        raise RuntimeError("Known joint-saturating steering command was accepted.")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
