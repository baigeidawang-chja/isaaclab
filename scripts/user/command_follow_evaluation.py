"""Shared checkpoint evaluation and CSV reporting for Car4WD CommandFollow tasks."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path(
    "/media/chja/CE54D158C95990271/IsaacLabTrainingData/traction_protective_control_evaluations"
)

STEERING_JOINTS = {
    "left": "joint_front_left_steer",
    "right": "joint_front_right_steer",
}
WHEEL_JOINTS = {
    "front_left": "joint_front_left_wheel_link_wheel",
    "front_right": "joint_front_right_wheel_link_wheel",
    "rear_left": "joint_back_left_wheel_link_wheel",
    "rear_right": "joint_back_right_wheel_link_wheel",
}


def _wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _joint_ids(robot, names: dict[str, str]) -> dict[str, int]:
    return {
        label: robot.find_joints([joint_name], preserve_order=True)[0][0]
        for label, joint_name in names.items()
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _rmse(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def _settling_time(
    times: list[float], errors: list[float], stage_start_s: float, tolerance: float
) -> float | None:
    for index, error in enumerate(errors):
        if abs(error) <= tolerance and all(abs(value) <= tolerance for value in errors[index:]):
            return max(0.0, times[index] - stage_start_s)
    return None


def _response_metrics(
    rows: list[dict[str, Any]],
    *,
    value_key: str,
    error_key: str,
    target: float,
    initial_actual: float,
    previous_target: float | None,
    tolerance: float,
    steady_state_window_s: float,
    angular: bool,
) -> dict[str, float | None]:
    values = [float(row[value_key]) for row in rows]
    errors = [float(row[error_key]) for row in rows]
    times = [float(row["sample_time_s"]) for row in rows]
    stage_start_s = float(rows[0]["command_time_s"])
    stage_end_s = float(rows[-1]["sample_time_s"])
    steady_start_s = max(stage_start_s, stage_end_s - steady_state_window_s)
    steady_errors = [error for time_s, error in zip(times, errors) if time_s >= steady_start_s]

    if angular:
        command_step = _wrap_to_pi(target - (initial_actual if previous_target is None else previous_target))
        target_errors = [_wrap_to_pi(value - target) for value in values]
    else:
        command_step = target - (initial_actual if previous_target is None else previous_target)
        target_errors = [value - target for value in values]

    overshoot = None
    overshoot_percent = None
    if abs(command_step) > tolerance:
        direction = 1.0 if command_step > 0.0 else -1.0
        overshoot = max(0.0, max(direction * value for value in target_errors))
        overshoot_percent = 100.0 * overshoot / abs(command_step)

    steady_error = _mean(steady_errors)
    return {
        "mae": _mean([abs(value) for value in errors]),
        "rmse": _rmse(errors),
        "steady_state_error": steady_error,
        "steady_state_abs_error": abs(steady_error),
        "overshoot": overshoot,
        "overshoot_percent": overshoot_percent,
        "settling_time_s": _settling_time(times, errors, stage_start_s, tolerance),
    }


def _stage_summaries(
    rows: list[dict[str, Any]],
    *,
    speed_tolerance: float,
    heading_tolerance: float,
    steady_state_window_s: float,
) -> list[dict[str, Any]]:
    summaries = []
    previous_final_speed = None
    previous_final_heading = None
    previous_target_speed = None
    previous_target_heading = None
    stage_ids = sorted({int(row["stage_index"]) for row in rows})

    for stage_index in stage_ids:
        stage_rows = [
            row
            for row in rows
            if int(row["stage_index"]) == stage_index and not bool(row["post_step_state_reset"])
        ]
        if not stage_rows:
            continue

        target_speed = float(stage_rows[0]["desired_speed"])
        target_heading = float(stage_rows[0]["desired_heading"])
        initial_speed = (
            float(stage_rows[0]["actual_speed"])
            if previous_final_speed is None
            else previous_final_speed
        )
        initial_heading = (
            float(stage_rows[0]["actual_heading"])
            if previous_final_heading is None
            else previous_final_heading
        )
        speed = _response_metrics(
            stage_rows,
            value_key="actual_speed",
            error_key="speed_error",
            target=target_speed,
            initial_actual=initial_speed,
            previous_target=previous_target_speed,
            tolerance=speed_tolerance,
            steady_state_window_s=steady_state_window_s,
            angular=False,
        )
        heading = _response_metrics(
            stage_rows,
            value_key="actual_heading",
            error_key="heading_error",
            target=target_heading,
            initial_actual=initial_heading,
            previous_target=previous_target_heading,
            tolerance=heading_tolerance,
            steady_state_window_s=steady_state_window_s,
            angular=True,
        )
        termination_reasons = [row["termination_reason"] for row in stage_rows if row["termination_reason"]]
        summary: dict[str, Any] = {
            "stage_index": stage_index,
            "start_time_s": stage_rows[0]["command_time_s"],
            "end_time_s": stage_rows[-1]["sample_time_s"],
            "duration_s": float(stage_rows[-1]["sample_time_s"]) - float(stage_rows[0]["command_time_s"]),
            "sample_count": len(stage_rows),
            "desired_speed": target_speed,
            "desired_heading": target_heading,
            "initial_actual_speed": initial_speed,
            "initial_actual_heading": initial_heading,
            "previous_desired_speed": previous_target_speed,
            "previous_desired_heading": previous_target_heading,
            "termination_reason": "|".join(sorted(set(termination_reasons))),
        }
        summary.update({f"speed_{key}": value for key, value in speed.items()})
        summary.update({f"heading_{key}": value for key, value in heading.items()})
        summaries.append(summary)
        previous_final_speed = float(stage_rows[-1]["actual_speed"])
        previous_final_heading = float(stage_rows[-1]["actual_heading"])
        previous_target_speed = target_speed
        previous_target_heading = target_heading

    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path.name}.")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(default_task: str, evaluation_name: str) -> None:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default=default_task)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--duration", type=float, default=None, help="Evaluation duration in seconds.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed_tolerance", type=float, default=0.05)
    parser.add_argument("--heading_tolerance", type=float, default=0.05)
    parser.add_argument("--steady_state_window", type=float, default=1.0)
    parser.add_argument("--disable_fabric", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from skrl.utils.runner.torch import Runner

    from isaaclab_rl.skrl import SkrlVecEnvWrapper
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

    try:
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=1,
            use_fabric=not args_cli.disable_fabric,
        )
        env_cfg.seed = args_cli.seed
        env_cfg.scene.num_envs = 1
        env_cfg.observations.policy.enable_corruption = False
        env_cfg.commands.planner_command.debug_vis = not args_cli.headless

        experiment_cfg = load_cfg_from_registry(args_cli.task, "skrl_cfg_entry_point")
        experiment_cfg["seed"] = args_cli.seed
        experiment_cfg["agent"]["experiment"]["wandb"] = False
        experiment_cfg["agent"]["experiment"]["write_interval"] = 0
        experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
        experiment_cfg["trainer"]["close_environment_at_exit"] = False

        raw_env = gym.make(args_cli.task, cfg=env_cfg)
        base_env = raw_env.unwrapped
        robot = base_env.scene["robot"]
        command_term = base_env.command_manager.get_term("planner_command")
        action_term = base_env.action_manager.get_term("throttle_steer")
        steering_ids = _joint_ids(robot, STEERING_JOINTS)
        wheel_ids = _joint_ids(robot, WHEEL_JOINTS)

        env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
        runner = Runner(env, experiment_cfg)
        checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
        print(f"[INFO] Loading model checkpoint from: {checkpoint_path}")
        runner.agent.load(str(checkpoint_path))
        runner.agent.set_running_mode("eval")

        with torch.inference_mode():
            observation, _ = env.reset()

        step_dt = float(base_env.step_dt)
        default_steps = max(1, int(base_env.max_episode_length) - 1)
        max_steps = (
            min(default_steps, max(1, math.ceil(args_cli.duration / step_dt)))
            if args_cli.duration is not None
            else default_steps
        )
        rows: list[dict[str, Any]] = []
        stage_index = 0
        previous_command = None

        for step in range(max_steps):
            desired_speed = float(command_term.desired_speed[0].item())
            desired_heading = float(command_term.desired_heading[0].item())
            command = (desired_speed, desired_heading)
            if previous_command is not None and (
                not math.isclose(command[0], previous_command[0], abs_tol=1.0e-6)
                or abs(_wrap_to_pi(command[1] - previous_command[1])) > 1.0e-6
            ):
                stage_index += 1
            previous_command = command

            with torch.inference_mode():
                outputs = runner.agent.act(observation, timestep=0, timesteps=0)
                actions = outputs[-1].get("mean_actions", outputs[0])
                observation, _, terminated, truncated, _ = env.step(actions)

            terminated_value = bool(terminated[0].item())
            truncated_value = bool(truncated[0].item())
            reasons = [
                name
                for name in base_env.termination_manager.active_terms
                if bool(base_env.termination_manager.get_term(name)[0].item())
            ]
            termination_reason = "|".join(reasons)
            actual_heading = float(robot.data.heading_w[0].item())
            plan_direction = torch.tensor(
                [math.cos(desired_heading), math.sin(desired_heading)],
                device=base_env.device,
            )
            actual_speed = float(torch.dot(robot.data.root_lin_vel_w[0, :2], plan_direction).item())
            raw_action = action_term.raw_actions[0]
            physical_target = action_term.target_actions[0]
            executed_action = action_term.processed_actions[0]

            row: dict[str, Any] = {
                "evaluation": evaluation_name,
                "task": args_cli.task,
                "step": step,
                "command_time_s": step * step_dt,
                "sample_time_s": (step + 1) * step_dt,
                "stage_index": stage_index,
                "desired_speed": desired_speed,
                "desired_heading": desired_heading,
                "actual_speed": actual_speed,
                "actual_body_forward_speed": float(robot.data.root_lin_vel_b[0, 0].item()),
                "actual_world_velocity_x": float(robot.data.root_lin_vel_w[0, 0].item()),
                "actual_world_velocity_y": float(robot.data.root_lin_vel_w[0, 1].item()),
                "actual_heading": actual_heading,
                "speed_error": actual_speed - desired_speed,
                "heading_error": _wrap_to_pi(actual_heading - desired_heading),
                "raw_action_speed": float(raw_action[0].item()),
                "raw_action_steering": float(raw_action[1].item()),
                "physical_target_speed": float(physical_target[0].item()),
                "physical_target_steering": float(physical_target[1].item()),
                "executed_speed": float(executed_action[0].item()),
                "executed_steering": float(executed_action[1].item()),
                "yaw_rate": float(robot.data.root_ang_vel_b[0, 2].item()),
                "terminated": int(terminated_value),
                "truncated": int(truncated_value),
                "termination_reason": termination_reason,
                "post_step_state_reset": int(terminated_value or truncated_value),
            }
            for side, joint_id in steering_ids.items():
                row[f"target_steering_{side}"] = float(robot.data.joint_pos_target[0, joint_id].item())
                row[f"actual_steering_{side}"] = float(robot.data.joint_pos[0, joint_id].item())
            for wheel, joint_id in wheel_ids.items():
                row[f"target_wheel_speed_{wheel}"] = float(robot.data.joint_vel_target[0, joint_id].item())
                row[f"actual_wheel_speed_{wheel}"] = float(robot.data.joint_vel[0, joint_id].item())
            rows.append(row)

            if terminated_value or truncated_value:
                break

        if rows and not rows[-1]["termination_reason"]:
            rows[-1]["termination_reason"] = "evaluation_horizon"

        summaries = _stage_summaries(
            rows,
            speed_tolerance=args_cli.speed_tolerance,
            heading_tolerance=args_cli.heading_tolerance,
            steady_state_window_s=args_cli.steady_state_window,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path(args_cli.output_dir).expanduser().resolve()
            if args_cli.output_dir
            else DEFAULT_OUTPUT_ROOT / evaluation_name / timestamp
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(output_dir / "steps.csv", rows)
        _write_csv(output_dir / "stages.csv", summaries)
        metadata = {
            "evaluation": evaluation_name,
            "task": args_cli.task,
            "checkpoint": str(checkpoint_path),
            "seed": args_cli.seed,
            "step_dt": step_dt,
            "steps": len(rows),
            "stages": len(summaries),
            "speed_tolerance": args_cli.speed_tolerance,
            "heading_tolerance": args_cli.heading_tolerance,
            "steady_state_window_s": args_cli.steady_state_window,
            "termination_reason": rows[-1]["termination_reason"],
        }
        (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[INFO] Step CSV: {output_dir / 'steps.csv'}")
        print(f"[INFO] Stage CSV: {output_dir / 'stages.csv'}")
        print(f"[INFO] Metadata: {output_dir / 'metadata.json'}")
        raw_env.close()
    finally:
        simulation_app.close()
