"""Run NM512 dreamerv3-torch with an IsaacLab environment adapter."""

import argparse
import functools
import pathlib
import sys
import traceback
from datetime import datetime

from ruamel.yaml import YAML
import torch
from torch import distributions as torchd

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


def main():
    parser = argparse.ArgumentParser(description="dreamerv3-torch + IsaacLab")
    parser.add_argument("--task", type=str, default="Isaac-Dreamer-Smoke-Cartpole-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--configs", nargs="+", default=["defaults"])
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    parser.add_argument("--wandb_project", type=str, default="dreamerv3-isaaclab")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default="")
    parser.add_argument("--wandb_notes", type=str, default="")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable Isaac Sim viewer window. Default is headless for stability.",
    )
    parser.add_argument("--eval_episode_num", type=int, default=0)
    parser.add_argument("--video_pred_log", action="store_true", default=False)
    AppLauncher.add_app_launcher_args(parser)
    args, remaining = parser.parse_known_args()

    if not args.viewer:
        args.headless = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    try:
        import gymnasium as gym
        import isaaclab_tasks  # noqa: F401
        import isaaclab_tasks.user.dreamerv3.env  # noqa: F401
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        here = pathlib.Path(__file__).resolve().parent
        sys.path.insert(0, str(here))

        import tools
        from dreamer import Dreamer, count_steps, make_dataset
        from isaaclab_adapter import make_isaaclab_env_proxies

        config = _load_config(here / "configs.yaml", args.configs, remaining)
        config.task = args.task
        config.seed = int(args.seed)
        config.envs = int(args.num_envs)
        config.parallel = False
        config.eval_episode_num = int(args.eval_episode_num)
        config.video_pred_log = bool(args.video_pred_log)
        config.encoder["mlp_keys"] = ".*"
        config.encoder["cnn_keys"] = "$^"
        config.decoder["mlp_keys"] = ".*"
        config.decoder["cnn_keys"] = "$^"
        if isinstance(config.device, str) and config.device.startswith("cuda") and not torch.cuda.is_available():
            config.device = "cpu"

        env_cfg = parse_env_cfg(
            args.task, device=config.device, num_envs=config.envs, use_fabric=True
        )
        if hasattr(env_cfg, "seed"):
            env_cfg.seed = config.seed

        if args.logdir:
            logdir = pathlib.Path(args.logdir).expanduser()
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            logdir = pathlib.Path("runs/dreamerv3_torch_isaaclab") / f"{args.task}_{ts}"

        config.logdir = logdir
        config.traindir = config.traindir or logdir / "train_eps"
        config.evaldir = config.evaldir or logdir / "eval_eps"
        config.steps //= config.action_repeat
        config.eval_every //= config.action_repeat
        config.log_every //= config.action_repeat
        config.time_limit //= config.action_repeat

        logdir.mkdir(parents=True, exist_ok=True)
        config.traindir.mkdir(parents=True, exist_ok=True)
        config.evaldir.mkdir(parents=True, exist_ok=True)
        step = count_steps(config.traindir)
        tags = [x.strip() for x in args.wandb_tags.split(",") if x.strip()]
        wb_name = args.wandb_name or f"{args.task}-{datetime.now().strftime('%m%d-%H%M%S')}"
        wandb_cfg = {
            "enabled": bool(args.wandb),
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "name": wb_name,
            "tags": tags,
            "notes": args.wandb_notes,
            "config": {
                "task": args.task,
                "seed": int(args.seed),
                "num_envs": int(config.envs),
                "config_names": list(args.configs),
                "device": str(config.device),
                "steps": float(config.steps),
                "train_ratio": float(config.train_ratio),
                "batch_size": int(config.batch_size),
                "batch_length": int(config.batch_length),
            },
        }
        logger = tools.Logger(logdir, config.action_repeat * step, wandb_cfg=wandb_cfg)

        def make_env():
            base = gym.make(args.task, cfg=env_cfg)
            backend, envs = make_isaaclab_env_proxies(
                base, time_limit_steps=config.time_limit
            )
            return backend, envs

        print("Create IsaacLab env.")
        backend, train_envs = make_env()
        acts = train_envs[0].action_space
        config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

        train_eps = tools.load_episodes(config.traindir, limit=config.dataset_size)
        eval_eps = tools.load_episodes(config.evaldir, limit=1)
        state = None

        prefill = max(0, config.prefill - count_steps(config.traindir))
        if prefill > 0:
            print(f"Prefill dataset ({prefill} steps).")
            if hasattr(acts, "discrete"):
                random_actor = tools.OneHotDist(
                    torch.zeros(config.num_actions).repeat(config.envs, 1)
                )
            else:
                low = torch.tensor(acts.low, dtype=torch.float32)
                high = torch.tensor(acts.high, dtype=torch.float32)
                low = torch.where(torch.isfinite(low), low, torch.full_like(low, -1.0))
                high = torch.where(torch.isfinite(high), high, torch.full_like(high, 1.0))
                random_actor = torchd.Independent(
                    torchd.Uniform(
                        low.repeat(config.envs, 1),
                        high.repeat(config.envs, 1),
                    ),
                    1,
                )

            def random_agent(o, d, s):
                action = random_actor.sample()
                logprob = random_actor.log_prob(action)
                return {"action": action, "logprob": logprob}, None

            state = tools.simulate(
                random_agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=prefill,
            )
            logger.step += prefill * config.action_repeat
            print(f"Logger: ({logger.step} steps).")

        train_dataset = make_dataset(train_eps, config)
        eval_dataset = make_dataset(eval_eps, config)
        agent = Dreamer(
            train_envs[0].observation_space,
            train_envs[0].action_space,
            config,
            logger,
            train_dataset,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)

        if (logdir / "latest.pt").exists():
            checkpoint = torch.load(logdir / "latest.pt")
            agent.load_state_dict(checkpoint["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
            agent._should_pretrain._once = False

        while agent._step < config.steps + config.eval_every:
            logger.write()
            if config.eval_episode_num > 0:
                print("Start evaluation.")
                eval_policy = functools.partial(agent, training=False)
                tools.simulate(
                    eval_policy,
                    train_envs,
                    eval_eps,
                    config.evaldir,
                    logger,
                    is_eval=True,
                    episodes=config.eval_episode_num,
                )
                if config.video_pred_log:
                    video_pred = agent._wm.video_pred(next(eval_dataset))
                    logger.video("eval_openl", video_pred.detach().cpu().numpy())

            print("Start training.")
            state = tools.simulate(
                agent,
                train_envs,
                train_eps,
                config.traindir,
                logger,
                limit=config.dataset_size,
                steps=config.eval_every,
                state=state,
            )
            ckpt = {
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            }
            # Keep latest for auto-resume, and also save immutable step snapshots.
            torch.save(ckpt, logdir / "latest.pt")
            ckpt_dir = logdir / "checkpoints"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            step_ckpt = ckpt_dir / f"step_{int(agent._step):012d}.pt"
            if not step_ckpt.exists():
                torch.save(ckpt, step_ckpt)
        backend.close()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        try:
            if "logger" in locals() and logger is not None:
                logger.close()
        except Exception:
            pass
        simulation_app.close()


if __name__ == "__main__":
    main()
