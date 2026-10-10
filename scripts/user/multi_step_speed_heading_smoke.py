"""Validate synchronized speed and heading steps and checkpoint interface dimensions."""

import argparse
import math

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    default="Isaac-TractionProtect-CommandFollow-MultiStepSpeedHeading-Car4WD-Play-v0",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _wrapped_difference(lhs: float, rhs: float) -> float:
    return math.atan2(math.sin(lhs - rhs), math.cos(lhs - rhs))


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.enable_corruption = False

    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    command_term = base_env.command_manager.get_term("planner_command")
    expected_speeds = list(command_term.cfg.speed_levels)
    expected_offsets = list(command_term.cfg.heading_offsets)

    with torch.inference_mode():
        observation, _ = env.reset()

    policy_observation = observation["policy"]
    action_dim = base_env.action_manager.total_action_dim
    zero_action = torch.zeros((1, action_dim), device=base_env.device)
    base_heading = float(command_term.desired_heading[0].item()) - expected_offsets[0]
    observed_speeds = [float(command_term.desired_speed[0].item())]
    observed_headings = [float(command_term.desired_heading[0].item())]
    transition_times = [0.0]

    num_steps = math.ceil(base_env.max_episode_length) - 1
    for step in range(num_steps):
        with torch.inference_mode():
            _, _, terminated, truncated, _ = env.step(zero_action)

        if bool((terminated[0] | truncated[0]).item()):
            raise RuntimeError(f"Episode ended before the complete profile at step {step + 1}.")

        speed = float(command_term.desired_speed[0].item())
        heading = float(command_term.desired_heading[0].item())
        speed_changed = not math.isclose(speed, observed_speeds[-1], abs_tol=1.0e-6)
        heading_changed = abs(_wrapped_difference(heading, observed_headings[-1])) > 1.0e-6
        if speed_changed or heading_changed:
            observed_speeds.append(speed)
            observed_headings.append(heading)
            transition_times.append((step + 1) * float(base_env.step_dt))

    if len(observed_speeds) != len(expected_speeds) or not all(
        math.isclose(actual, expected, abs_tol=1.0e-6)
        for actual, expected in zip(observed_speeds, expected_speeds)
    ):
        raise AssertionError(f"Unexpected speed profile: {observed_speeds}; expected {expected_speeds}.")

    expected_headings = [
        math.atan2(math.sin(base_heading + offset), math.cos(base_heading + offset))
        for offset in expected_offsets
    ]
    if len(observed_headings) != len(expected_headings) or not all(
        abs(_wrapped_difference(actual, expected)) <= 1.0e-6
        for actual, expected in zip(observed_headings, expected_headings)
    ):
        raise AssertionError(f"Unexpected heading profile: {observed_headings}; expected {expected_headings}.")

    print(f"policy_observation_shape={tuple(policy_observation.shape)}")
    print(f"action_dim={action_dim}")
    print(f"base_heading_rad={base_heading:+.6f}")
    print(f"speed_levels_m_s={observed_speeds}")
    print(f"heading_levels_rad={observed_headings}")
    print(f"transition_times_s={transition_times}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
