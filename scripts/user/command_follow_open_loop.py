"""Run policy-free steering symmetry diagnostics for the Car4WD CommandFollow task."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-TractionProtect-CommandFollow-Car4WD-v0")
parser.add_argument("--speed", type=float, default=0.8, help="Physical forward-speed command in m/s.")
parser.add_argument("--steer", type=float, default=0.25, help="Physical steering magnitude in rad.")
parser.add_argument("--duration", type=float, default=8.0, help="Duration of each test case in simulation seconds.")
parser.add_argument("--warmup", type=float, default=3.0, help="Initial interval excluded from steady-state metrics.")
parser.add_argument("--output", default="/tmp/command_follow_open_loop.csv", help="Per-step CSV output path.")
parser.add_argument("--disable_fabric", action="store_true", help="Use USD I/O instead of Fabric.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
import json
import math
from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


STEERING_JOINTS = {
    "steer_right": "joint_front_right_steer",
    "steer_left": "joint_front_left_steer",
}
WHEEL_JOINTS = {
    "wheel_front_right": "joint_front_right_wheel_link_wheel",
    "wheel_front_left": "joint_front_left_wheel_link_wheel",
    "wheel_rear_right": "joint_back_right_wheel_link_wheel",
    "wheel_rear_left": "joint_back_left_wheel_link_wheel",
}


def _wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _single_joint_ids(robot, joint_names: dict[str, str]) -> dict[str, int]:
    return {
        label: robot.find_joints([joint_name], preserve_order=True)[0][0]
        for label, joint_name in joint_names.items()
    }


def _mean(rows: list[dict[str, float]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def main():
    if args_cli.warmup >= args_cli.duration:
        raise ValueError("--warmup must be smaller than --duration")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1, use_fabric=not args_cli.disable_fabric)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = max(20.0, args_cli.duration + 2.0)
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.planner_command.resampling_time_range = (100.0, 100.0)
    env_cfg.commands.planner_command.desired_speed_range = (args_cli.speed, args_cli.speed)
    env_cfg.commands.planner_command.desired_heading_range = (0.0, 0.0)
    pose_range = env_cfg.events.reset_base.params["pose_range"]
    pose_range["x"] = (0.0, 0.0)
    pose_range["y"] = (0.0, 0.0)
    pose_range["yaw"] = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("throttle_steer")
    command_term = base_env.command_manager.get_term("planner_command")

    steering_ids = _single_joint_ids(robot, STEERING_JOINTS)
    wheel_ids = _single_joint_ids(robot, WHEEL_JOINTS)
    print("\n=== Runtime joint mapping ===")
    print(f"all_joint_names={robot.joint_names}")
    print(f"action steering ids={action_term._steering_ids}, names={action_term._steering_names}")
    print(f"action wheel ids={action_term._wheel_ids}, names={action_term._wheel_names}")
    print("Ackermann steering columns=[left, right]")
    print("Ackermann wheel columns=[rear_left, rear_right, front_left, front_right]")

    scale = action_term._scale.detach().cpu().tolist()
    offset = action_term._offset.detach().cpu().tolist()
    raw_speed = (args_cli.speed - offset[0]) / scale[0]
    raw_steer_magnitude = args_cli.steer / scale[1]
    if abs(raw_speed) > 1.0 or abs(raw_steer_magnitude) > 1.0:
        raise ValueError("Requested physical command exceeds the configured normalized action range")

    step_dt = float(base_env.step_dt)
    num_steps = math.ceil(args_cli.duration / step_dt)
    rows: list[dict[str, float | str | int]] = []
    summaries: dict[str, dict[str, float]] = {}

    for case_name, steer_sign in (("zero", 0.0), ("positive", 1.0), ("negative", -1.0)):
        with torch.inference_mode():
            env.reset()
        previous_heading = float(robot.data.heading_w[0].item())
        unwrapped_heading = previous_heading
        case_rows: list[dict[str, float | str | int]] = []
        raw_action = torch.tensor(
            [[raw_speed, steer_sign * raw_steer_magnitude]], device=base_env.device, dtype=torch.float32
        )

        for step in range(num_steps):
            with torch.inference_mode():
                _, _, terminated, truncated, _ = env.step(raw_action)

                heading = float(robot.data.heading_w[0].item())
                unwrapped_heading += _wrap_to_pi(heading - previous_heading)
                previous_heading = heading
                processed = action_term.processed_actions[0]
                ack_left, ack_right, ack_wheels = action_term._calculate_ackermann_angles_and_velocities(
                    target_steering_angle=processed[1:2], target_velocity=processed[0:1]
                )

                row: dict[str, float | str | int] = {
                    "case": case_name,
                    "step": step + 1,
                    "time_s": (step + 1) * step_dt,
                    "raw_speed": float(raw_action[0, 0].item()),
                    "raw_steering": float(raw_action[0, 1].item()),
                    "target_speed": float(action_term.target_actions[0, 0].item()),
                    "target_steering": float(action_term.target_actions[0, 1].item()),
                    "executed_speed": float(processed[0].item()),
                    "executed_steering": float(processed[1].item()),
                    "ackermann_steer_left": float(ack_left[0].item()),
                    "ackermann_steer_right": float(ack_right[0].item()),
                    "ackermann_wheel_rear_left": float(ack_wheels[0, 0].item()),
                    "ackermann_wheel_rear_right": float(ack_wheels[0, 1].item()),
                    "ackermann_wheel_front_left": float(ack_wheels[0, 2].item()),
                    "ackermann_wheel_front_right": float(ack_wheels[0, 3].item()),
                    "heading": heading,
                    "heading_unwrapped": unwrapped_heading,
                    "heading_error": float(command_term.heading_error[0].item()),
                    "yaw_rate": float(robot.data.root_ang_vel_b[0, 2].item()),
                    "terminated": int(terminated[0].item()),
                    "truncated": int(truncated[0].item()),
                }
                for label, joint_id in steering_ids.items():
                    row[f"target_{label}"] = float(robot.data.joint_pos_target[0, joint_id].item())
                    row[f"actual_{label}"] = float(robot.data.joint_pos[0, joint_id].item())
                for label, joint_id in wheel_ids.items():
                    row[f"target_{label}"] = float(robot.data.joint_vel_target[0, joint_id].item())
                    row[f"actual_{label}"] = float(robot.data.joint_vel[0, joint_id].item())
                rows.append(row)
                case_rows.append(row)

            if bool(terminated[0] or truncated[0]):
                raise RuntimeError(f"Case {case_name!r} reset unexpectedly at step {step + 1}")

        steady_rows = [row for row in case_rows if float(row["time_s"]) >= args_cli.warmup]
        steady_duration = float(steady_rows[-1]["time_s"]) - float(steady_rows[0]["time_s"])
        heading_rate = (
            float(steady_rows[-1]["heading_unwrapped"]) - float(steady_rows[0]["heading_unwrapped"])
        ) / steady_duration
        summaries[case_name] = {
            "heading_rate_rad_s": heading_rate,
            "mean_yaw_rate_rad_s": _mean(steady_rows, "yaw_rate"),
            "mean_heading_error_rad": _mean(steady_rows, "heading_error"),
            "mean_executed_steering_rad": _mean(steady_rows, "executed_steering"),
            "mean_actual_steer_left_rad": _mean(steady_rows, "actual_steer_left"),
            "mean_actual_steer_right_rad": _mean(steady_rows, "actual_steer_right"),
            "mean_actual_wheel_front_left_rad_s": _mean(steady_rows, "actual_wheel_front_left"),
            "mean_actual_wheel_front_right_rad_s": _mean(steady_rows, "actual_wheel_front_right"),
            "mean_actual_wheel_rear_left_rad_s": _mean(steady_rows, "actual_wheel_rear_left"),
            "mean_actual_wheel_rear_right_rad_s": _mean(steady_rows, "actual_wheel_rear_right"),
        }

    positive_yaw = summaries["positive"]["mean_yaw_rate_rad_s"]
    negative_yaw = summaries["negative"]["mean_yaw_rate_rad_s"]
    summaries["symmetry"] = {
        "yaw_rate_odd_bias_rad_s": 0.5 * (positive_yaw + negative_yaw),
        "yaw_rate_abs_ratio_pos_over_neg": abs(positive_yaw) / max(abs(negative_yaw), 1.0e-9),
    }

    output_path = Path(args_cli.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    print("\n=== Open-loop summary ===")
    print(json.dumps(summaries, indent=2))
    print(f"CSV: {output_path}")
    print(f"Summary: {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
