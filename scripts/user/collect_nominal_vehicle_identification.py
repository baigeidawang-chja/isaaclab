"""Collect policy-independent nominal Car4WD identification and validation CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_PATH = (
    REPO_ROOT
    / "source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/mpc/config"
    / "nominal_vehicle_identification_protocol.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_identification"
)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-TractionProtect-CommandFollow-Car4WD-v0")
parser.add_argument("--protocol", default=str(DEFAULT_PROTOCOL_PATH))
parser.add_argument("--output_dir", default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
from datetime import datetime
import hashlib
import json
import math
from typing import Any

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.user.traction_protective_control.mpc import (
    DEFAULT_STEERING_CONFIG_PATH,
    load_steering_constraints,
)
from isaaclab_tasks.utils import parse_env_cfg


STEERING_JOINTS = {
    "front_left": "joint_front_left_steer",
    "front_right": "joint_front_right_steer",
}
WHEEL_JOINTS = {
    "front_left": "joint_front_left_wheel_link_wheel",
    "front_right": "joint_front_right_wheel_link_wheel",
    "rear_left": "joint_back_left_wheel_link_wheel",
    "rear_right": "joint_back_right_wheel_link_wheel",
}


def _wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _joint_ids(robot, names: dict[str, str]) -> dict[str, int]:
    return {
        label: robot.find_joints([joint_name], preserve_order=True)[0][0]
        for label, joint_name in names.items()
    }


def _build_profiles(protocol: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    profiles: dict[str, list[dict[str, Any]]] = {}
    for split in ("identification", "validation"):
        split_cfg = protocol[split]
        speed_cfg = split_cfg["speed_step"]
        split_profiles = [
            {
                "name": f"{split}_speed_step",
                "type": "speed_step",
                "segments": [
                    {
                        "duration_s": float(speed_cfg["segment_duration_s"]),
                        "speed_m_s": float(speed),
                        "steering_rad": float(speed_cfg["steering_rad"]),
                    }
                    for speed in speed_cfg["speed_levels_m_s"]
                ],
            }
        ]
        steering_cfg = split_cfg["steering_step"]
        for speed in steering_cfg["fixed_speeds_m_s"]:
            speed = float(speed)
            split_profiles.append(
                {
                    "name": f"{split}_steering_step_{speed:.2f}mps",
                    "type": "steering_step",
                    "segments": [
                        {
                            "duration_s": float(steering_cfg["segment_duration_s"]),
                            "speed_m_s": speed,
                            "steering_rad": float(steering),
                        }
                        for steering in steering_cfg["steering_levels_rad"]
                    ],
                }
            )
        profiles[split] = split_profiles
    return profiles


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No samples collected for {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    protocol_path = Path(args_cli.protocol).expanduser().resolve()
    protocol = _load_json(protocol_path)
    if protocol.get("schema_version") != 1:
        raise ValueError("Unsupported identification protocol schema version.")
    profiles = _build_profiles(protocol)
    constraints = load_steering_constraints()

    for split_profiles in profiles.values():
        for profile in split_profiles:
            for segment in profile["segments"]:
                if not constraints.is_feasible(segment["steering_rad"]):
                    raise ValueError(
                        f"Profile {profile['name']} requests infeasible steering "
                        f"{segment['steering_rad']} rad."
                    )

    longest_profile_s = max(
        sum(segment["duration_s"] for segment in profile["segments"])
        for split_profiles in profiles.values()
        for profile in split_profiles
    )
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = longest_profile_s + 2.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.planner_command.debug_vis = False
    env_cfg.terminations.out_of_bounds.params = {
        "x_min": -2000.0,
        "x_max": 2000.0,
        "y_min": -2000.0,
        "y_max": 2000.0,
    }
    pose_range = env_cfg.events.reset_base.params["pose_range"]
    pose_range.update({"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    action_term = base_env.action_manager.get_term("throttle_steer")
    steering_ids = _joint_ids(robot, STEERING_JOINTS)
    wheel_ids = _joint_ids(robot, WHEEL_JOINTS)
    step_dt = float(base_env.step_dt)

    action_scale = action_term._scale.detach().cpu().tolist()
    action_offset = action_term._offset.detach().cpu().tolist()
    config_tolerance = 1.0e-6
    if not math.isclose(action_scale[1], constraints.action_scale_rad, abs_tol=config_tolerance):
        raise RuntimeError("Steering constraint action scale does not match the environment action term.")
    if not math.isclose(action_offset[1], constraints.action_offset_rad, abs_tol=config_tolerance):
        raise RuntimeError("Steering constraint action offset does not match the environment action term.")
    if not math.isclose(action_term.cfg.base_length, constraints.wheelbase_m, abs_tol=config_tolerance):
        raise RuntimeError("Steering constraint wheelbase does not match the environment action term.")
    if not math.isclose(action_term.cfg.base_width, constraints.track_width_m, abs_tol=config_tolerance):
        raise RuntimeError("Steering constraint track width does not match the environment action term.")

    collected: dict[str, list[dict[str, Any]]] = {"identification": [], "validation": []}
    global_step = 0

    try:
        for split, split_profiles in profiles.items():
            for run_index, profile in enumerate(split_profiles):
                with torch.inference_mode():
                    env.reset()
                previous_heading = float(robot.data.heading_w[0].item())
                unwrapped_heading = previous_heading
                previous_body_speed = float(robot.data.root_lin_vel_b[0, 0].item())
                run_step = 0

                for segment_index, segment in enumerate(profile["segments"]):
                    requested_speed = float(segment["speed_m_s"])
                    requested_steering = float(segment["steering_rad"])
                    segment_steps_float = float(segment["duration_s"]) / step_dt
                    segment_steps = round(segment_steps_float)
                    if segment_steps < 1 or not math.isclose(
                        segment_steps_float, segment_steps, abs_tol=1.0e-9
                    ):
                        raise ValueError("Every segment duration must be an integer multiple of step_dt.")

                    raw_speed = (requested_speed - action_offset[0]) / action_scale[0]
                    raw_steering = (requested_steering - action_offset[1]) / action_scale[1]
                    if abs(raw_speed) > 1.0 or abs(raw_steering) > 1.0:
                        raise ValueError(f"Profile {profile['name']} exceeds the normalized action range.")
                    raw_action = torch.tensor(
                        [[raw_speed, raw_steering]],
                        device=base_env.device,
                        dtype=torch.float32,
                    )

                    for segment_step in range(segment_steps):
                        with torch.inference_mode():
                            _, _, terminated, truncated, _ = env.step(raw_action)

                        heading = float(robot.data.heading_w[0].item())
                        unwrapped_heading += _wrap_to_pi(heading - previous_heading)
                        previous_heading = heading
                        body_speed = float(robot.data.root_lin_vel_b[0, 0].item())
                        longitudinal_acceleration = (body_speed - previous_body_speed) / step_dt
                        previous_body_speed = body_speed

                        actual_left = float(robot.data.joint_pos[0, steering_ids["front_left"]].item())
                        actual_right = float(robot.data.joint_pos[0, steering_ids["front_right"]].item())
                        actual_center = constraints.center_angle_from_wheel_angles_rad(
                            actual_left, actual_right
                        )
                        target_action = action_term.target_actions[0]
                        executed_action = action_term.processed_actions[0]
                        raw_applied = action_term.raw_actions[0]

                        reasons = [
                            name
                            for name in base_env.termination_manager.active_terms
                            if bool(base_env.termination_manager.get_term(name)[0].item())
                        ]
                        terminated_value = bool(terminated[0].item())
                        truncated_value = bool(truncated[0].item())
                        row: dict[str, Any] = {
                            "data_source": "isaac_sim_policy_bypassed",
                            "split": split,
                            "profile_type": profile["type"],
                            "profile_name": profile["name"],
                            "run_index": run_index,
                            "segment_index": segment_index,
                            "segment_step": segment_step,
                            "global_step": global_step,
                            "run_time_s": (run_step + 1) * step_dt,
                            "segment_time_s": (segment_step + 1) * step_dt,
                            "step_dt_s": step_dt,
                            "requested_target_speed_m_s": requested_speed,
                            "requested_target_steering_rad": requested_steering,
                            "raw_action_speed": float(raw_applied[0].item()),
                            "raw_action_steering": float(raw_applied[1].item()),
                            "physical_target_speed_m_s": float(target_action[0].item()),
                            "physical_target_steering_rad": float(target_action[1].item()),
                            "executed_speed_m_s": float(executed_action[0].item()),
                            "executed_steering_rad": float(executed_action[1].item()),
                            "actual_body_vx_m_s": body_speed,
                            "actual_body_vy_m_s": float(robot.data.root_lin_vel_b[0, 1].item()),
                            "actual_world_vx_m_s": float(robot.data.root_lin_vel_w[0, 0].item()),
                            "actual_world_vy_m_s": float(robot.data.root_lin_vel_w[0, 1].item()),
                            "actual_longitudinal_accel_m_s2": longitudinal_acceleration,
                            "actual_heading_rad": heading,
                            "actual_heading_unwrapped_rad": unwrapped_heading,
                            "actual_yaw_rate_rad_s": float(robot.data.root_ang_vel_b[0, 2].item()),
                            "actual_position_x_m": float(robot.data.root_pos_w[0, 0].item()),
                            "actual_position_y_m": float(robot.data.root_pos_w[0, 1].item()),
                            "actual_steering_front_left_rad": actual_left,
                            "actual_steering_front_right_rad": actual_right,
                            "actual_equivalent_center_steering_rad": actual_center,
                            "terminated": int(terminated_value),
                            "truncated": int(truncated_value),
                            "termination_reason": "|".join(reasons),
                        }
                        for side, joint_id in steering_ids.items():
                            row[f"target_steering_{side}_rad"] = float(
                                robot.data.joint_pos_target[0, joint_id].item()
                            )
                        for wheel, joint_id in wheel_ids.items():
                            row[f"target_wheel_speed_{wheel}_rad_s"] = float(
                                robot.data.joint_vel_target[0, joint_id].item()
                            )
                            row[f"actual_wheel_speed_{wheel}_rad_s"] = float(
                                robot.data.joint_vel[0, joint_id].item()
                            )
                        collected[split].append(row)

                        if terminated_value or truncated_value:
                            raise RuntimeError(
                                f"Profile {profile['name']} reset at step {run_step + 1}: {reasons}"
                            )
                        run_step += 1
                        global_step += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(args_cli.output_dir).expanduser().resolve()
            if args_cli.output_dir
            else DEFAULT_OUTPUT_ROOT / timestamp
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_csv(output_dir / "identification.csv", collected["identification"])
        _write_csv(output_dir / "validation.csv", collected["validation"])
        manifest = {
            "schema_version": 1,
            "data_source": "Isaac Sim open-loop ManagerBasedRLEnv with the policy bypassed",
            "task": args_cli.task,
            "seed": args_cli.seed,
            "step_dt_s": step_dt,
            "protocol_path": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
            "steering_config_path": str(DEFAULT_STEERING_CONFIG_PATH),
            "steering_config_sha256": _sha256(DEFAULT_STEERING_CONFIG_PATH),
            "steering_constraint": constraints.summary(),
            "identification_rows": len(collected["identification"]),
            "validation_rows": len(collected["validation"]),
            "identification_profiles": [profile["name"] for profile in profiles["identification"]],
            "validation_profiles": [profile["name"] for profile in profiles["validation"]],
            "policy_used": False,
            "ppo_task_modified": False,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        (output_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2), encoding="utf-8"
        )
        print(f"[INFO] Identification CSV: {output_dir / 'identification.csv'}")
        print(f"[INFO] Validation CSV: {output_dir / 'validation.csv'}")
        print(f"[INFO] Manifest: {output_dir / 'manifest.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
