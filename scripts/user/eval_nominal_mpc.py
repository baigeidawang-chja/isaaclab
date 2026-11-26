"""Run the nominal OSQP MPC in one Isaac Lab environment and record the full control chain."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


DEFAULT_OUTPUT_ROOT = Path(
    "/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_nominal_mpc"
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-TractionProtect-CommandFollow-Car4WD-v0")
parser.add_argument("--profile", choices=("fixed", "speed_step", "heading_step", "joint_step"), default="fixed")
parser.add_argument("--duration", type=float, default=2.0)
parser.add_argument("--segment_duration", type=float, default=2.0)
parser.add_argument("--speed", type=float, default=0.6)
parser.add_argument(
    "--physical-speed-override", type=float, default=None,
    help="Diagnostic only: replace the MPC speed output in m/s while keeping its steering output.",
)
parser.add_argument("--heading", type=float, default=0.0)
parser.add_argument("--speed_sequence", default="0.3,0.5,0.8,0.6")
parser.add_argument("--heading_sequence", default="0.0,0.2,-0.2,0.0")
parser.add_argument("--output_dir", default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import inspect
import sys
import json
import math
from typing import Any

import gymnasium as gym
import numpy as np
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.user.traction_protective_control.mpc import (
    CarVWActionAdapter,
    DEFAULT_NOMINAL_MODEL_CONFIG_PATH,
    DEFAULT_OSQP_MPC_CONFIG_PATH,
    DEFAULT_STEERING_CONFIG_PATH,
    NominalMPC,
    NominalMPCReference,
    NominalMPCState,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _joint_ids(robot, names: dict[str, str]) -> dict[str, int]:
    return {
        label: robot.find_joints([joint_name], preserve_order=True)[0][0]
        for label, joint_name in names.items()
    }


def _parse_sequence(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Command sequences must contain finite comma-separated values.")
    return values


def _build_profile() -> list[tuple[float, float]]:
    speeds = _parse_sequence(args_cli.speed_sequence)
    headings = _parse_sequence(args_cli.heading_sequence)
    if args_cli.profile == "fixed":
        return [(args_cli.speed, args_cli.heading)]
    if args_cli.profile == "speed_step":
        return [(speed, args_cli.heading) for speed in speeds]
    if args_cli.profile == "heading_step":
        return [(args_cli.speed, heading) for heading in headings]
    if len(speeds) != len(headings):
        raise ValueError("joint_step requires equal speed and heading sequence lengths.")
    return list(zip(speeds, headings))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Nominal MPC smoke test produced no samples.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    profile = _build_profile()
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = args_cli.duration + 2.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.commands.planner_command.debug_vis = not args_cli.headless
    env_cfg.commands.planner_command.resampling_time_range = (args_cli.duration + 10.0,) * 2
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
    robot_usd_path = Path(robot.cfg.spawn.usd_path).expanduser().resolve()
    action_term = base_env.action_manager.get_term("throttle_steer")
    planner_command = base_env.command_manager.get_term("planner_command")
    steering_ids = _joint_ids(robot, STEERING_JOINTS)
    wheel_ids = _joint_ids(robot, WHEEL_JOINTS)
    steering_constraints = load_steering_constraints()
    controller = NominalMPC()
    lower, upper = controller.config.reference_speed_bounds
    if any(not math.isfinite(speed) or not lower <= speed <= upper
           or not math.isfinite(heading) for speed, heading in profile):
        env.close()
        raise ValueError(f"Planner speed references must be within [{lower}, {upper}] m/s.")
    if action_term.cfg.no_reverse:
        env.close()
        raise RuntimeError("Bidirectional MPC requires an ActionTerm with no_reverse=False.")
    step_dt = float(base_env.step_dt)

    action_scale = tuple(float(value) for value in action_term._scale.detach().cpu().tolist())
    action_offset = tuple(float(value) for value in action_term._offset.detach().cpu().tolist())
    adapter = CarVWActionAdapter(scale=action_scale, offset=action_offset)
    tolerance = 1.0e-6
    if not math.isclose(step_dt, controller.control_period_s, abs_tol=tolerance):
        raise RuntimeError("Environment step_dt does not match identified model/MPC dt.")
    expected_scale = (1.5, 0.24)
    expected_offset = (0.0, 0.0)
    speed_override = args_cli.physical_speed_override
    if speed_override is not None and (
        not math.isfinite(speed_override)
        or not action_offset[0] - action_scale[0] <= speed_override <= action_offset[0] + action_scale[0]
        or not controller.config.control_lower[0] <= speed_override <= controller.config.control_upper[0]
    ):
        raise ValueError("Physical speed override is outside the ActionTerm/MPC input range.")
    scale_matches = all(
        math.isclose(actual, expected, abs_tol=tolerance)
        for actual, expected in zip(action_scale, expected_scale)
    )
    offset_matches = all(
        math.isclose(actual, expected, abs_tol=tolerance)
        for actual, expected in zip(action_offset, expected_offset)
    )
    if not scale_matches or not offset_matches:
        raise RuntimeError(f"Unexpected RateLimitedCarVWAction mapping: {action_scale}, {action_offset}.")
    if not math.isclose(action_term.cfg.max_v_rate, controller.config.control_rate_per_s[0], abs_tol=tolerance):
        raise RuntimeError("MPC speed-rate constraint does not match RateLimitedCarVWAction.")
    if not math.isclose(action_term.cfg.max_steer_rate, controller.config.control_rate_per_s[1], abs_tol=tolerance):
        raise RuntimeError("MPC steering-rate constraint does not match RateLimitedCarVWAction.")
    geometry_pairs = (
        (action_term.cfg.base_length, steering_constraints.wheelbase_m, "wheelbase"),
        (action_term.cfg.base_width, steering_constraints.track_width_m, "track width"),
        (action_term.cfg.wheel_radius, steering_constraints.wheel_radius_m, "wheel radius"),
        (controller.model.wheelbase_m, steering_constraints.wheelbase_m, "model wheelbase"),
    )
    for actual, expected, name in geometry_pairs:
        if not math.isclose(actual, expected, abs_tol=tolerance):
            raise RuntimeError(f"MPC {name} mismatch: {actual} != {expected}.")

    segment_steps = max(1, round(args_cli.segment_duration / step_dt))
    total_steps = max(1, math.ceil(args_cli.duration / step_dt))
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    try:
        with torch.inference_mode():
            env.reset()
        controller.reset([0.0, 0.0])
        initial_state = {
            "root_state_w": robot.data.root_state_w[0].detach().cpu().tolist(),
            "joint_names": robot.joint_names,
            "joint_pos_rad": robot.data.joint_pos[0].detach().cpu().tolist(),
            "joint_vel_rad_s": robot.data.joint_vel[0].detach().cpu().tolist(),
            "executed_action": action_term.processed_actions[0].detach().cpu().tolist(),
        }

        for step in range(total_steps):
            stage_index = min(step // segment_steps, len(profile) - 1)
            desired_speed, desired_heading = profile[stage_index]
            with torch.inference_mode():
                planner_command.command[0, 0] = desired_speed
                planner_command.command[0, 1] = desired_heading
            heading = float(robot.data.heading_w[0].item())
            pre_position = robot.data.root_pos_w[0].detach().cpu().tolist()
            body_speed = float(robot.data.root_lin_vel_b[0, 0].item())
            actual_left = float(robot.data.joint_pos[0, steering_ids["front_left"]].item())
            actual_right = float(robot.data.joint_pos[0, steering_ids["front_right"]].item())
            actual_center = steering_constraints.center_angle_from_wheel_angles_rad(
                actual_left, actual_right
            )
            heading_error = _wrap_to_pi(desired_heading - heading)
            previous_executed = action_term.processed_actions[0].detach().cpu().numpy().astype(np.float64)
            result = controller.solve(
                NominalMPCState(heading_error, body_speed, actual_center),
                NominalMPCReference(desired_speed),
                previous_executed,
            )
            if speed_override is not None and result.fallback_used:
                raise RuntimeError("Steering MPC fell back; paired diagnostic cannot continue.")
            selected_control = result.control.copy()
            if speed_override is not None:
                selected_control[0] = speed_override
            normalized = adapter.to_normalized(selected_control)
            raw_action = torch.tensor(
                normalized[None, :], device=base_env.device, dtype=torch.float32
            )

            with torch.inference_mode():
                _, _, terminated, truncated, _ = env.step(raw_action)

            terminated_value = bool(terminated[0].item())
            truncated_value = bool(truncated[0].item())
            reasons = [
                name
                for name in base_env.termination_manager.active_terms
                if bool(base_env.termination_manager.get_term(name)[0].item())
            ]
            post_heading = float(robot.data.heading_w[0].item())
            post_left = float(robot.data.joint_pos[0, steering_ids["front_left"]].item())
            post_right = float(robot.data.joint_pos[0, steering_ids["front_right"]].item())
            physical_target = action_term.target_actions[0]
            executed = action_term.processed_actions[0]
            row: dict[str, Any] = {
                "data_source": "isaac_sim_nominal_mpc_policy_bypassed",
                "environment_desired_speed_m_s": float(planner_command.command[0, 0].item()),
                "environment_desired_heading_rad": float(planner_command.command[0, 1].item()),
                "task": args_cli.task,
                "profile": args_cli.profile,
                "step": step,
                "stage_index": stage_index,
                "command_time_s": step * step_dt,
                "sample_time_s": (step + 1) * step_dt,
                "step_dt_s": step_dt,
                "desired_speed_m_s": desired_speed,
                "desired_heading_rad": desired_heading,
                "pre_heading_error_rad": heading_error,
                "pre_heading_rad": heading,
                "pre_world_x_m": pre_position[0],
                "pre_world_y_m": pre_position[1],
                "previous_executed_speed_m_s": float(previous_executed[0]),
                "previous_executed_steering_rad": float(previous_executed[1]),
                "pre_body_speed_m_s": body_speed,
                "pre_equivalent_center_steering_rad": actual_center,
                "mpc_speed_target_m_s": result.speed_target_m_s,
                "selected_speed_target_m_s": float(selected_control[0]),
                "mpc_steering_target_rad": result.steering_target_rad,
                "normalized_action_speed": float(normalized[0]),
                "normalized_action_steering": float(normalized[1]),
                "physical_target_speed_m_s": float(physical_target[0].item()),
                "physical_target_steering_rad": float(physical_target[1].item()),
                "executed_speed_m_s": float(executed[0].item()),
                "executed_steering_rad": float(executed[1].item()),
                "actual_body_vx_m_s": float(robot.data.root_lin_vel_b[0, 0].item()),
                "actual_body_vy_m_s": float(robot.data.root_lin_vel_b[0, 1].item()),
                "actual_world_vx_m_s": float(robot.data.root_lin_vel_w[0, 0].item()),
                "actual_world_vy_m_s": float(robot.data.root_lin_vel_w[0, 1].item()),
                "actual_world_x_m": float(robot.data.root_pos_w[0, 0].item()),
                "actual_world_y_m": float(robot.data.root_pos_w[0, 1].item()),
                "actual_heading_rad": post_heading,
                "actual_heading_error_rad": _wrap_to_pi(desired_heading - post_heading),
                "actual_yaw_rate_rad_s": float(robot.data.root_ang_vel_b[0, 2].item()),
                "actual_steering_front_left_rad": post_left,
                "actual_steering_front_right_rad": post_right,
                "actual_equivalent_center_steering_rad": (
                    steering_constraints.center_angle_from_wheel_angles_rad(post_left, post_right)
                ),
                "solver_success": int(result.success),
                "solver_status": result.status,
                "solver_status_value": result.status_value,
                "solve_time_s": result.solve_time_s,
                "solver_run_time_s": result.solver_run_time_s,
                "solver_iterations": result.iterations,
                "solver_objective": result.objective,
                "solver_primal_residual": result.primal_residual,
                "solver_dual_residual": result.dual_residual,
                "warm_start_used": int(result.warm_start_used),
                "fallback_used": int(result.fallback_used),
                "fallback_reason": result.fallback_reason or "",
                "speed_region_index": result.speed_region_index,
                "linearization_speed_m_s": result.linearization_speed_m_s,
                "linearization_steering_rad": result.linearization_steering_rad,
                "predicted_next_heading_error_rad": float(result.predicted_states[1, 0]),
                "predicted_next_body_speed_m_s": float(result.predicted_states[1, 1]),
                "predicted_next_center_steering_rad": float(result.predicted_states[1, 2]),
                "terminated": int(terminated_value),
                "post_step_state_reset": int(terminated_value or truncated_value),
                "response_sample_valid": int(not (terminated_value or truncated_value)),
                "truncated": int(truncated_value),
                "termination_reason": "|".join(reasons),
            }
            row["actual_projected_speed_m_s"] = (
                row["actual_world_vx_m_s"] * math.cos(desired_heading)
                + row["actual_world_vy_m_s"] * math.sin(desired_heading)
            )
            for side, joint_id in steering_ids.items():
                row[f"target_steering_{side}_rad"] = float(robot.data.joint_pos_target[0, joint_id].item())
            for wheel, joint_id in wheel_ids.items():
                row[f"target_wheel_speed_{wheel}_rad_s"] = float(robot.data.joint_vel_target[0, joint_id].item())
                row[f"actual_wheel_speed_{wheel}_rad_s"] = float(robot.data.joint_vel[0, joint_id].item())
                row[f"computed_wheel_torque_{wheel}_nm"] = float(robot.data.computed_torque[0, joint_id].item())
                row[f"applied_wheel_torque_{wheel}_nm"] = float(robot.data.applied_torque[0, joint_id].item())
            rows.append(row)
            predictions.append(
                {
                    "step": step,
                    "mpc_prediction_matches_selected_speed": speed_override is None,
                    "predicted_states": result.predicted_states.tolist(),
                    "predicted_controls": result.predicted_controls.tolist(),
                }
            )
            if terminated_value or truncated_value:
                break

        if rows and not rows[-1]["termination_reason"]:
            rows[-1]["termination_reason"] = "evaluation_horizon"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(args_cli.output_dir).expanduser().resolve()
            if args_cli.output_dir
            else DEFAULT_OUTPUT_ROOT / args_cli.profile / timestamp
        )
        output_dir.mkdir(parents=True, exist_ok=False)
        _write_csv(output_dir / "steps.csv", rows)
        with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as file:
            for prediction in predictions:
                file.write(json.dumps(prediction) + "\n")
        # Keep each run analyzable after future parameter/config changes.
        config_snapshots = {}
        for name, source in (
            ("model", DEFAULT_NOMINAL_MODEL_CONFIG_PATH),
            ("mpc", DEFAULT_OSQP_MPC_CONFIG_PATH),
            ("steering", DEFAULT_STEERING_CONFIG_PATH),
        ):
            destination = output_dir / f"{name}_config.json"
            destination.write_bytes(source.read_bytes())
            config_snapshots[name] = destination
        metadata = {
            "schema_version": 2,
            "argv": sys.argv,
            "duration_s": args_cli.duration,
            "segment_duration_s": args_cli.segment_duration,
            "expected_steps": total_steps,
            "final_stage_index": rows[-1]["stage_index"],
            "initial_state": initial_state,
            "wheel_joint_mapping": {
                label: {"name": WHEEL_JOINTS[label], "id": joint_id}
                for label, joint_id in wheel_ids.items()
            },
            "steering_joint_mapping": {
                label: {"name": STEERING_JOINTS[label], "id": joint_id}
                for label, joint_id in steering_ids.items()
            },
            "action_wheel_ids": action_term._wheel_ids,
            "action_wheel_names": action_term._wheel_names,
            "action_steering_ids": action_term._steering_ids,
            "action_steering_names": action_term._steering_names,
            "task": args_cli.task,
            "profile": args_cli.profile,
            "longitudinal_output_mode": "nominal_mpc" if speed_override is None else "fixed_physical_speed_target",
            "physical_speed_override_m_s": speed_override,
            "prediction_role": "executed_mpc_plan" if speed_override is None else "counterfactual_mpc_plan",
            "commands": profile,
            "seed": args_cli.seed,
            "step_dt_s": step_dt,
            "steps": len(rows),
            "solver_status_counts": dict(Counter(row["solver_status"] for row in rows)),
            "fallback_count": sum(row["fallback_used"] for row in rows),
            "termination_reason": rows[-1]["termination_reason"],
            "policy_used": False,
            "performance_claim": False,
            "action_scale": action_scale,
            "action_offset": action_offset,
            "robot_usd_path": str(robot_usd_path),
            "robot_usd_sha256": _sha256(robot_usd_path),
            "model_config": str(config_snapshots["model"]),
            "model_config_source": str(DEFAULT_NOMINAL_MODEL_CONFIG_PATH),
            "model_config_sha256": _sha256(DEFAULT_NOMINAL_MODEL_CONFIG_PATH),
            "mpc_config": str(config_snapshots["mpc"]),
            "mpc_config_source": str(DEFAULT_OSQP_MPC_CONFIG_PATH),
            "mpc_config_sha256": _sha256(DEFAULT_OSQP_MPC_CONFIG_PATH),
            "steering_config": str(config_snapshots["steering"]),
            "steering_config_source": str(DEFAULT_STEERING_CONFIG_PATH),
            "steering_config_sha256": _sha256(DEFAULT_STEERING_CONFIG_PATH),
        }
        runtime = {
            "python": sys.executable,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("isaaclab", "osqp", "skrl", "torch", "numpy", "scipy")},
            "osqp_native_version": __import__("osqp").OSQP().version(),
            "osqp_module": __import__("osqp").__file__,
            "isaac_sim_version": (Path(__file__).resolve().parents[2] / "_isaac_sim/VERSION").read_text().strip(),
            "environment": base_env.cfg.to_dict(),
            "action_class_file": inspect.getfile(type(action_term)),
            "apply_actions_file": inspect.getfile(action_term.apply_actions),
            "apply_actions_sha256": _sha256(Path(inspect.getfile(action_term.apply_actions))),
            "control_dt_s": step_dt,
            "torque_semantics": "Last physics substep actuator computed/applied torque; not an interval mean or contact-force measurement.",
        }
        (output_dir / "runtime_environment.json").write_text(
            json.dumps(runtime, indent=2, default=str), encoding="utf-8"
        )
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[INFO] MPC step CSV: {output_dir / 'steps.csv'}")
        print(f"[INFO] MPC predictions: {output_dir / 'predictions.jsonl'}")
        print(f"[INFO] MPC metadata: {output_dir / 'metadata.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        sys.stdout.flush()
        simulation_app.close()
