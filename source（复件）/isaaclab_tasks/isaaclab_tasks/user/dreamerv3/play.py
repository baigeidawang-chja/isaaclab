"""DreamerV3 checkpoint playback / evaluation script for Isaac Lab."""

import argparse
import pathlib
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="DreamerV3 checkpoint player")
parser.add_argument("--task", type=str, required=True, help="Isaac Lab task name")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to a saved checkpoint .pt file")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
parser.add_argument("--max_steps", type=int, default=5000, help="Max environment steps per episode")
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="Fallback Dreamer config names from configs.yaml, comma-separated. If omitted, try checkpoint run dir first.",
)
parser.add_argument(
    "--task_config",
    type=str,
    default=None,
    help="Optional YAML override for env config, same meaning as in main.py.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()
sys.argv = [sys.argv[0]] + unknown

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.user.dreamerv3.env  # noqa: F401
from isaaclab.utils.io import dump_yaml  # noqa: F401
from isaaclab_tasks.user.dreamerv3.agent import Agent
from isaaclab_tasks.user.dreamerv3.envwrapper import IsaacLabDreamerWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry, parse_env_cfg


def deep_update(base: dict, update: dict) -> dict:
    result = base.copy()
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


def load_local_dreamer_config(config_names: str) -> dict:
    import ruamel.yaml as yaml

    config_path = pathlib.Path(__file__).parent / "configs.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        all_configs = yaml.YAML(typ="safe").load(f)

    config = dict(all_configs["defaults"])
    for name in config_names.split(","):
        name = name.strip()
        if name and name != "defaults" and name in all_configs:
            config = deep_update(config, all_configs[name])
    return config


def load_dreamer_cfg_for_checkpoint(checkpoint_path: pathlib.Path) -> dict:
    import ruamel.yaml as yaml

    run_dir = checkpoint_path.parent.parent
    cfg_path = run_dir / "dreamer_cfg.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.YAML(typ="safe").load(f)
    if args_cli.config:
        return load_local_dreamer_config(args_cli.config)
    return load_local_dreamer_config("defaults")


def resolve_env_cfg(task_name: str, num_envs: int):
    if args_cli.task_config:
        env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
        if args_cli.task_config.endswith(".yaml"):
            import yaml

            with open(args_cli.task_config, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f)
            env_cfg.from_dict(overrides)
        env_cfg.sim.device = args_cli.device
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"):
            env_cfg.scene.num_envs = num_envs
        return env_cfg
    return parse_env_cfg(task_name, device=args_cli.device, num_envs=num_envs)


def initialize_lazy_modules(agent, env: IsaacLabDreamerWrapper, device: torch.device):
    carry = agent.init_carry(args_cli.num_envs, device)
    obs, _ = env.reset()
    with torch.no_grad():
        carry, _, _ = agent.policy(carry, obs, mode="eval")
    return carry


def _infer_checkpoint_obs_dim(state_dict: dict) -> int | None:
    weight = state_dict.get("enc.mlp_layers.0.linear.weight")
    if isinstance(weight, torch.Tensor) and weight.ndim == 2:
        return int(weight.shape[1])
    return None


def play():
    device = torch.device(args_cli.device)
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    ckpt_path = pathlib.Path(args_cli.checkpoint).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    dreamer_cfg = load_dreamer_cfg_for_checkpoint(ckpt_path)
    agent_cfg = dreamer_cfg.get("agent", {})
    agent_cls = Agent
    env_cfg = resolve_env_cfg(args_cli.task, args_cli.num_envs)

    print(f"[PLAY] Task: {args_cli.task}")
    print(f"[PLAY] Checkpoint: {ckpt_path}")
    print(f"[PLAY] Device: {device}")
    print(f"[PLAY] Num envs: {args_cli.num_envs}")
    print("[PLAY] Agent kind: dreamer")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = IsaacLabDreamerWrapper(env, device=device)
    agent = agent_cls(env.obs_space, env.act_space, agent_cfg).to(device)

    initialize_lazy_modules(agent, env, device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["agent"] if isinstance(ckpt, dict) and "agent" in ckpt else ckpt
    current_obs_dim = None
    if isinstance(env.obs_space, dict) and "policy" in env.obs_space:
        current_obs_dim = int(np.prod(env.obs_space["policy"].shape))
    ckpt_obs_dim = _infer_checkpoint_obs_dim(state_dict)
    if ckpt_obs_dim is not None and current_obs_dim is not None and ckpt_obs_dim != current_obs_dim:
        raise RuntimeError(
            "Checkpoint is incompatible with the current environment observation layout: "
            f"checkpoint expects policy obs dim {ckpt_obs_dim}, current task provides {current_obs_dim}. "
            "This usually means the task/observation config changed after training. "
            "Use a checkpoint trained on the current env version or retrain."
        )
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[PLAY] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[PLAY] Unexpected keys: {len(unexpected)}")

    agent.eval()
    carry = agent.init_carry(args_cli.num_envs, device)
    obs, _ = env.reset()

    episode_returns = torch.zeros(args_cli.num_envs, device=device)
    episode_lengths = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    completed_returns = []
    completed_lengths = []

    start_time = time.time()
    total_steps = 0

    while simulation_app.is_running() and len(completed_returns) < args_cli.num_episodes:
        with torch.no_grad():
            carry, action, _ = agent.policy(carry, obs, mode="eval")

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated | truncated

        if reward.dim() > 1:
            reward = reward.squeeze(-1)
        if done.dim() > 1:
            done = done.squeeze(-1)

        episode_returns += reward
        episode_lengths += 1
        total_steps += args_cli.num_envs

        if done.any():
            done_mask = done.bool()
            for i in range(args_cli.num_envs):
                if done_mask[i] and len(completed_returns) < args_cli.num_episodes:
                    ret = float(episode_returns[i].item())
                    length = int(episode_lengths[i].item())
                    completed_returns.append(ret)
                    completed_lengths.append(length)
                    print(
                        f"[PLAY] Episode {len(completed_returns):>3d}/{args_cli.num_episodes} | "
                        f"return={ret:>9.3f} | length={length:>5d}"
                    )
            episode_returns[done_mask] = 0
            episode_lengths[done_mask] = 0

        obs = next_obs

        if args_cli.max_steps > 0 and (episode_lengths >= args_cli.max_steps).any():
            over_mask = episode_lengths >= args_cli.max_steps
            for i in range(args_cli.num_envs):
                if over_mask[i] and len(completed_returns) < args_cli.num_episodes:
                    ret = float(episode_returns[i].item())
                    length = int(episode_lengths[i].item())
                    completed_returns.append(ret)
                    completed_lengths.append(length)
                    print(
                        f"[PLAY] Episode {len(completed_returns):>3d}/{args_cli.num_episodes} | "
                        f"return={ret:>9.3f} | length={length:>5d} | truncated=max_steps"
                    )
            episode_returns[over_mask] = 0
            episode_lengths[over_mask] = 0

    elapsed = time.time() - start_time
    if completed_returns:
        returns = np.asarray(completed_returns, dtype=np.float32)
        lengths = np.asarray(completed_lengths, dtype=np.float32)
        print("")
        print(f"[PLAY] Episodes: {len(returns)}")
        print(
            f"[PLAY] Return mean/std/min/max: "
            f"{returns.mean():.3f} / {returns.std():.3f} / {returns.min():.3f} / {returns.max():.3f}"
        )
        print(
            f"[PLAY] Length mean/std/min/max: "
            f"{lengths.mean():.1f} / {lengths.std():.1f} / {lengths.min():.0f} / {lengths.max():.0f}"
        )
        print(f"[PLAY] FPS: {total_steps / max(elapsed, 1e-6):.1f}")
    else:
        print("[PLAY] No episodes completed.")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    play()
