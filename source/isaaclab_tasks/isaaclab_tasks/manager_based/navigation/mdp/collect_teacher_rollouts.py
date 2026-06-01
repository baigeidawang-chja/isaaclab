"""
Collect teacher rollout data for Stage-2 distillation.

Runs the teacher policy in the environment and saves (proprio, grid, action, prev_action)
tuples. The grid observation is the ground-truth occupancy from the scene.

Output: .pt files, each containing one episode's data.
"""

from __future__ import annotations

import os
import argparse
import torch
from typing import Dict


def split_obs(obs: torch.Tensor, grid_start: int, grid_end: int):
    """Split full observation into proprio and grid components.

    Args:
        obs: (B, obs_dim) full observation.
        grid_start: Start index of grid features.
        grid_end: End index of grid features.

    Returns:
        proprio: (B, proprio_dim) everything except grid.
        grid: (B, grid_cells) grid features.
    """
    proprio = torch.cat([obs[:, :grid_start], obs[:, grid_end:]], dim=-1)
    grid = obs[:, grid_start:grid_end]
    return proprio, grid


def collect_rollouts(
    env,
    teacher_policy,
    num_episodes: int,
    output_dir: str,
    grid_start: int,
    grid_end: int,
    max_steps: int = 2000,
    device: str = "cuda:0",
):
    """Collect rollout data from teacher policy.

    Args:
        env: Isaac Lab environment instance.
        teacher_policy: Trained teacher policy (obs -> action).
        num_episodes: Number of episodes to collect.
        output_dir: Directory to save .pt files.
        grid_start: Start index of grid in obs.
        grid_end: End index of grid in obs.
        max_steps: Maximum steps per episode.
        device: Torch device.
    """
    os.makedirs(output_dir, exist_ok=True)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        obs_tensor = obs["policy"]  # Adjust key to your env

        proprios = []
        grids = []
        actions = []
        prev_actions = []

        prev_action = torch.zeros(obs_tensor.shape[0], env.action_space.shape[-1], device=device)

        for step in range(max_steps):
            proprio, grid = split_obs(obs_tensor, grid_start, grid_end)

            with torch.no_grad():
                action = teacher_policy(obs_tensor)

            proprios.append(proprio.cpu())
            grids.append(grid.cpu())
            actions.append(action.cpu())
            prev_actions.append(prev_action.cpu())

            obs, _, terminated, truncated, _ = env.step(action)
            obs_tensor = obs["policy"]
            prev_action = action.clone()

            if terminated.any() or truncated.any():
                break

        # Stack and save
        data = {
            "proprio": torch.stack(proprios, dim=1).squeeze(0),      # (T, proprio_dim)
            "grid": torch.stack(grids, dim=1).squeeze(0),            # (T, grid_cells)
            "action": torch.stack(actions, dim=1).squeeze(0),        # (T, act_dim)
            "prev_action": torch.stack(prev_actions, dim=1).squeeze(0),  # (T, act_dim)
        }

        save_path = os.path.join(output_dir, f"episode_{ep:04d}.pt")
        torch.save(data, save_path)
        print(f"[Collect] Episode {ep+1}/{num_episodes} saved ({data['proprio'].shape[0]} steps)")

    print(f"[Collect] Done. {num_episodes} episodes saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./teacher_rollouts")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--grid_start", type=int, default=0)
    parser.add_argument("--grid_end", type=int, default=400)
    parser.add_argument("--max_steps", type=int, default=2000)

    args = parser.parse_args()

    # ===== Adapt the following to your setup =====
    # from your_env_module import create_env
    # from your_policy_module import load_teacher

    # env = create_env(...)
    # teacher = load_teacher(args.teacher_ckpt)
    # collect_rollouts(env, teacher, args.num_episodes, args.output_dir,
    #                  args.grid_start, args.grid_end, args.max_steps)

    print("Please adapt the __main__ block to your environment and teacher loading code.")
