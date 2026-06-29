"""Collect offline medium-state supervision data from the Amphibious IsaacLab task."""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback

import numpy as np

from isaaclab.app import AppLauncher


LABEL_NAMES = [
    "eta_wheel",
    "eta_thruster",
]


def _sample_action(
    action_dim: int,
    num_envs: int,
    step: int,
    rng: np.random.Generator,
    labels: np.ndarray | None = None,
) -> np.ndarray:
    """Semi-structured excitation for waterland medium-state data collection."""
    actions = np.zeros((num_envs, action_dim), dtype=np.float32)
    actions[:, 0] = rng.uniform(0.6, 1.0, size=num_envs)
    actions[:, 1] = rng.uniform(-0.05, 0.05, size=num_envs)
    if action_dim >= 3:
        actions[:, 2] = rng.uniform(0.3, 1.0, size=num_envs)
    return np.clip(actions, -1.0, 1.0)


def main():
    parser = argparse.ArgumentParser(description="Collect Amphibious medium-state dataset.")
    parser.add_argument("--task", type=str, default="Isaac-Navigation-Car-Dreamer-AmphibiousTerrain-v0")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="runs/medium_dataset/amphibious_terrain_medium_dataset.npz")
    parser.add_argument("--viewer", action="store_true", help="Enable Isaac Sim viewer window.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    if not args.viewer:
        args.headless = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    backend = None

    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.user.dreamerv3.env  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        here = pathlib.Path(__file__).resolve().parent
        sys.path.insert(0, str(here))
        from isaaclab_adapter import make_isaaclab_env_proxies

        rng = np.random.default_rng(args.seed)
        env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs, use_fabric=True)
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = int(args.seed)

        base_env = gym.make(args.task, cfg=env_cfg)
        backend, envs = make_isaaclab_env_proxies(base_env)
        num_envs = len(envs)
        action_dim = int(envs[0].action_space.shape[0])

        reset_thunks = [env.reset() for env in envs]
        obs_batch = [thunk() for thunk in reset_thunks]
        if "medium_state_label" not in obs_batch[0]:
            raise KeyError("Expected observation key 'medium_state_label'. Check Amphibious observation config.")

        obs_steps = []
        action_steps = []
        label_steps = []
        reward_steps = []
        done_steps = []

        for step in range(int(args.steps)):
            current_labels = np.stack([obs["medium_state_label"] for obs in obs_batch], axis=0).astype(np.float32)
            actions = _sample_action(action_dim, num_envs, step, rng, current_labels)
            step_thunks = [env.step(actions[i]) for i, env in enumerate(envs)]
            results = [thunk() for thunk in step_thunks]
            obs_batch = [result[0] for result in results]

            obs_steps.append(np.stack([obs["obs"] for obs in obs_batch], axis=0).astype(np.float32))
            label_steps.append(
                np.stack([obs["medium_state_label"] for obs in obs_batch], axis=0).astype(np.float32)
            )
            action_steps.append(actions.astype(np.float32))
            reward_steps.append(np.asarray([result[1] for result in results], dtype=np.float32))
            done_steps.append(np.asarray([result[2] for result in results], dtype=np.bool_))

            if (step + 1) % 500 == 0:
                print(f"[COLLECT] step={step + 1}/{args.steps}", flush=True)

        output = pathlib.Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        obs = np.stack(obs_steps, axis=0)
        actions = np.stack(action_steps, axis=0)
        labels = np.stack(label_steps, axis=0)
        rewards = np.stack(reward_steps, axis=0)
        dones = np.stack(done_steps, axis=0)

        # 数据采集，保存成.npz数据集
        np.savez_compressed(
            output,
            obs=obs,
            action=actions,
            medium_state_label=labels,
            reward=rewards,
            done=dones,
            label_names=np.asarray(LABEL_NAMES),
        )
        print(f"[COLLECT] saved={output}")
        print(f"[COLLECT] obs shape={obs.shape}")
        print(f"[COLLECT] action shape={actions.shape}")
        print(f"[COLLECT] medium_state_label shape={labels.shape}")
        for idx, name in enumerate(LABEL_NAMES[: labels.shape[-1]]):
            values = labels[..., idx]
            print(
                f"[COLLECT] {name} min/max/mean="
                f"({values.min():.3f}, {values.max():.3f}, {values.mean():.3f})"
            )
        eta_wheel = labels[..., 0]
        eta_thruster = labels[..., 1]
        wheel_transition = np.mean((eta_wheel > 0.05) & (eta_wheel < 0.95))
        thruster_transition = np.mean((eta_thruster > 0.05) & (eta_thruster < 0.95))
        print(f"[COLLECT] eta_wheel transition ratio={wheel_transition:.3f}")
        print(f"[COLLECT] eta_thruster transition ratio={thruster_transition:.3f}")
        print(f"[COLLECT] reward shape={rewards.shape} done shape={dones.shape}")
    except Exception:
        traceback.print_exc()
        raise
    finally:
        if backend is not None:
            backend.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
