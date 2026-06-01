"""Play/evaluate a trained dreamerv3-torch checkpoint on IsaacLab."""

import argparse
import pathlib
import sys
import traceback
from datetime import datetime

import numpy as np
from ruamel.yaml import YAML

from isaaclab.app import AppLauncher


def _load_config(config_path: pathlib.Path, names: list[str], overrides: list[str]):
    import tools

    y = YAML(typ="safe", pure=True)
    configs = y.load(config_path.read_text())

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                recursive_update(base[key], value)
            else:
                base[key] = value

    merged = {}
    for name in names:
        if name not in configs:
            raise KeyError(f"Config '{name}' not found in {config_path}.")
        recursive_update(merged, configs[name])

    parser = argparse.ArgumentParser(add_help=False)
    for key, value in sorted(merged.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    return parser.parse_args(overrides)


def _dummy_dataset():
    while True:
        # Play mode uses training=False, so dataset should not be consumed.
        # Keep a valid endless generator as a safe placeholder.
        yield {}


def main():
    parser = argparse.ArgumentParser(description="dreamerv3-torch play + IsaacLab")
    parser.add_argument("--task", type=str, default="Isaac-Dreamer-Smoke-Cartpole-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--logdir", type=str, required=True, help="Training logdir containing latest.pt")
    parser.add_argument("--checkpoint", type=str, default="latest.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--viewer", action="store_true", help="Enable Isaac Sim GUI window")
    AppLauncher.add_app_launcher_args(parser)
    args, remaining = parser.parse_known_args()
    print(f"[PLAY] args episodes={args.episodes} viewer={args.viewer} headless={getattr(args, 'headless', None)}", flush=True)

    if not args.viewer:
        args.headless = True
    else:
        # In viewer mode, force GUI on to keep the app event loop alive.
        args.headless = False
    print(f"[PLAY] resolved headless={args.headless}", flush=True)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    logger = None
    try:
        import torch
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.user.dreamerv3.env  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        here = pathlib.Path(__file__).resolve().parent
        sys.path.insert(0, str(here))

        import tools
        from dreamer import Dreamer
        from isaaclab_adapter import make_isaaclab_env_proxies

        config = _load_config(here / "configs.yaml", args.configs, remaining)
        config.task = args.task
        config.seed = int(args.seed)
        config.envs = int(args.num_envs)
        config.parallel = False
        config.eval_episode_num = int(args.episodes)
        print(f"[PLAY] config.eval_episode_num={config.eval_episode_num}", flush=True)
        config.video_pred_log = False
        config.encoder["mlp_keys"] = ".*"
        config.encoder["cnn_keys"] = "$^"
        config.decoder["mlp_keys"] = ".*"
        config.decoder["cnn_keys"] = "$^"

        # Keep same runtime behavior as training script.
        if isinstance(config.device, str) and config.device.startswith("cuda") and not torch.cuda.is_available():
            config.device = "cpu"

        env_cfg = parse_env_cfg(
            args.task, device=config.device, num_envs=config.envs, use_fabric=True
        )
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = config.seed

        logdir = pathlib.Path(args.logdir).expanduser()
        ckpt_path = (logdir / args.checkpoint).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        playdir = logdir / "play_eval" / datetime.now().strftime("%Y%m%d_%H%M%S")
        playdir.mkdir(parents=True, exist_ok=True)
        logger = tools.Logger(playdir, step=0, wandb_cfg=None)

        def make_env():
            base = gym.make(args.task, cfg=env_cfg)
            backend, envs = make_isaaclab_env_proxies(
                base, time_limit_steps=config.time_limit // config.action_repeat
            )
            return backend, envs

        backend, eval_envs = make_env()
        acts = eval_envs[0].action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

        # Dreamer expects a dataset handle in ctor, but play mode won't consume it.
        agent = Dreamer(
            eval_envs[0].observation_space,
            eval_envs[0].action_space,
            config,
            logger,
            _dummy_dataset(),
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)

        try:
            checkpoint = torch.load(
                ckpt_path, map_location=config.device, weights_only=True
            )
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=config.device)
        agent.load_state_dict(checkpoint["agent_state_dict"])
        print("[PLAY] checkpoint loaded", flush=True)

        eval_eps = tools.load_episodes(playdir, limit=1)
        eval_policy = lambda o, d, s: agent(o, d, s, training=False)
        print("[PLAY] entering rollout", flush=True)
        if config.eval_episode_num > 0:
            print(f"[PLAY] Run fixed episodes: {config.eval_episode_num}", flush=True)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                playdir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
        else:
            # --episodes 0 means run continuously until viewer is closed / process interrupted.
            print("[PLAY] Run continuously (episodes=0). Close viewer to stop.", flush=True)
            state = None
            ticks = 0
            while simulation_app.is_running():
                state = tools.simulate(
                    eval_policy,
                    eval_envs,
                    eval_eps,
                    playdir,
                    logger,
                    is_eval=True,
                    steps=int(config.envs),
                    state=state,
                )
                ticks += 1
                if ticks % 200 == 0:
                    print(f"[PLAY] ticks={ticks}", flush=True)

        print(f"[PLAY] Done. Results saved under: {playdir}")
        backend.close()
    except BaseException:
        print("[PLAY] fatal exception:", flush=True)
        traceback.print_exc()
        raise
    finally:
        try:
            if logger is not None:
                logger.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
