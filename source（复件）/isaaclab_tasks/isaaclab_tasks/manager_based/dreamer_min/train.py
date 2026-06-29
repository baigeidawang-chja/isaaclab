from __future__ import annotations

"""
Minimal Dreamer-style trainer for IsaacLab.

IMPORTANT:
Isaac Sim must be launched before importing isaaclab envs/assets (which import isaacsim.*).
We follow the same pattern as scripts/reinforcement_learning/skrl/train.py.
"""

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Any, Optional
import traceback
import importlib
import numpy as np
try:
    import wandb
except Exception:
    wandb = None

# --- Launch Isaac Sim first (before isaaclab / isaacsim imports) ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="Gym task name (IsaacLab registered env id)")
parser.add_argument("--torch_device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=0)

# AppLauncher CLI args (headless, livestream, etc.)
AppLauncher.add_app_launcher_args(parser)

# parse known args; keep unknown for Hydra-like usage if needed
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- Now safe to import the rest ---
import torch
import torch.nn.functional as F
import gymnasium as gym

import isaaclab_tasks  # noqa: F401  # register tasks

from isaaclab_tasks.manager_based.dreamer_min.agent import DreamerMin
from isaaclab_tasks.manager_based.dreamer_min.replay import EpisodeReplay
from isaaclab_tasks.manager_based.dreamer_min.utils import (
    flatten_any,
    obs_extract_grid,
    obs_extract_policy,
    soft_update_,
    to_torch,
)
from isaaclab_tasks.manager_based.dreamer_min.world_model import WorldModel


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda:0"

    # data
    # 回放缓冲区步数
    capacity_steps: int = 200_000
    # 采样序列长度
    seq_len: int = 50
    batch_size: int = 16
    collect_steps_per_iter: int = 1000   # across vector envs
    prefill_steps: int = 5000

    # train
    iters: int = 10000
    model_updates_per_iter: int = 50
    ac_updates_per_iter: int = 50
    imagine_horizon: int = 15

    # losses
    lr_model: float = 3e-4
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    kl_scale: float = 1e-3
    reward_scale: float = 1.0
    cont_scale: float = 1.0
    grid_scale: float = 0.1

    # discounts
    gamma: float = 0.99
    lam: float = 0.95

    # target update
    tau: float = 0.01


def make_env(task: str):
    # IsaacLab ManagerBased envs require cfg=...
    spec = gym.spec(task)
    kw = dict(spec.kwargs) if spec.kwargs is not None else {}

    # Preferred key used by IsaacLab task registrations
    env_cfg_ep = kw.get("env_cfg_entry_point", None)
    if env_cfg_ep is None:
        raise RuntimeError(
            f"Env spec for '{task}' does not provide 'env_cfg_entry_point'. "
            f"Available kwargs keys: {list(kw.keys())}"
        )

    env_cfg = _load_cfg_from_entry_point(env_cfg_ep)
    env = gym.make(task, cfg=env_cfg)
    return env

def infer_action_dim(env) -> int:
    return int(env.action_space.shape[0])


def policy_and_grid_from_obs(obs: Any, device: torch.device):
    pol = obs_extract_policy(obs)
    pol = to_torch(pol, device=device, dtype=torch.float32)
    pol = flatten_any(pol)

    grid = obs_extract_grid(obs)
    if grid is not None:
        grid = to_torch(grid, device=device, dtype=torch.float32)
        grid = flatten_any(grid)
        grid = (grid > 0.5).to(torch.float32)
    return pol, grid

def _load_cfg_from_entry_point(ep: str):
    """Load a configclass object from 'module.path:ClassName'."""
    if ":" not in ep:
        raise ValueError(f"Invalid entry point: {ep}")
    mod_name, attr = ep.split(":", 1)
    mod = importlib.import_module(mod_name)
    obj = getattr(mod, attr)
    # If it's a class, instantiate it; if already an instance, return as-is.
    return obj() if isinstance(obj, type) else obj

def _infer_action_spec(env):
    """Return (kind, act_dim) where kind in {'box','dict'}."""
    space = env.action_space
    print("[dreamer_min] action_space:", env.action_space)

    # Box space: treat last dimension as action dimension (supports shapes like (act_dim,) or (1, act_dim))
    if hasattr(space, "shape") and space.shape is not None:
        shape = tuple(space.shape)
        if len(shape) == 0:
            raise RuntimeError(f"Invalid Box action_space shape: {shape}")
        return "box", int(shape[-1])

    # Dict space: try single entry
    if hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
        if len(space.spaces) == 1:
            k = next(iter(space.spaces.keys()))
            sub = space.spaces[k]
            if hasattr(sub, "shape") and sub.shape is not None:
                sub_shape = tuple(sub.shape)
                return "dict", int(sub_shape[-1])
        raise RuntimeError(f"Unsupported Dict action_space keys={list(space.spaces.keys())}")

    raise RuntimeError(f"Unsupported action_space type: {type(space)}")


def _to_env_action(env, action_tensor: torch.Tensor):
    """Convert torch action -> env.step() compatible action (torch), with exact action_space.shape."""
    space = env.action_space
    a = action_tensor.to(dtype=torch.float32)

    # Dict action_space
    if hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
        if len(space.spaces) != 1:
            raise RuntimeError(f"Unsupported Dict action_space keys={list(space.spaces.keys())}")
        key = next(iter(space.spaces.keys()))
        sub = space.spaces[key]
        return {key: a.reshape(sub.shape)}

    # Box action_space
    return a.reshape(space.shape)


def _random_env_action(env, device: torch.device):
    """Sample exactly matching env.action_space.shape, return torch tensor with same shape."""
    space = env.action_space

    if hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
        if len(space.spaces) != 1:
            raise RuntimeError(f"Unsupported Dict action_space keys={list(space.spaces.keys())}")
        key = next(iter(space.spaces.keys()))
        sub = space.spaces[key]
        a = np.asarray(sub.sample(), dtype=np.float32).reshape(sub.shape)
        return torch.as_tensor(a, device=device, dtype=torch.float32)

    a = np.asarray(space.sample(), dtype=np.float32).reshape(space.shape)
    return torch.as_tensor(a, device=device, dtype=torch.float32)

def main():
    cfg = Config(seed=args_cli.seed, device=args_cli.torch_device) 

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)

    run_wandb = False
    if wandb is not None:
        try:
            wandb.init(project="dreamer_min", name=f"{args_cli.task}-s{cfg.seed}", config=asdict(cfg), reinit=True)
            run_wandb = True
        except Exception as e:
            print("[dreamer_min] wandb init failed:", e)
            run_wandb = False

    env = None  # CHANGED: ensure defined for finally

    try:
        env = make_env(args_cli.task)
        obs, _ = env.reset()
    except Exception as e:
        print("\n[dreamer_min] Failed to create/reset environment. Original exception:\n")
        traceback.print_exc()
        print("\n[dreamer_min] Note: the '__del__/_is_closed' error afterwards is a side-effect of partial init.\n")
        # Important: avoid calling env.close() on a partially constructed env
        env = None
        return

    # infer dims dynamically
    pol0, grid0 = policy_and_grid_from_obs(obs, device=device)
    obs_dim = int(pol0.shape[-1])

    # CHANGED: robust action dim inference
    _, act_dim = _infer_action_spec(env)
    grid_dim = int(grid0.shape[-1]) if grid0 is not None else None

    print(f"[dreamer_min] obs_dim={obs_dim} act_dim={act_dim} grid_dim={grid_dim}")

    cur_returns = defaultdict(float)
    cur_lengths = defaultdict(int)
    global_step = 0

    wm = WorldModel(action_dim=act_dim, obs_dim=obs_dim).to(device)
    if grid_dim is not None:
        wm.ensure_grid_head(grid_dim)

    agent = DreamerMin(feat_dim=wm.feat_dim, act_dim=act_dim).to(device)

    opt_model = torch.optim.Adam(wm.parameters(), lr=cfg.lr_model)
    opt_actor = torch.optim.Adam(agent.actor.parameters(), lr=cfg.lr_actor)
    opt_critic = torch.optim.Adam(agent.critic.parameters(), lr=cfg.lr_critic)

    replay = EpisodeReplay(capacity_steps=cfg.capacity_steps, device=device)

    # -------- prefill --------
    steps = 0
    while steps < cfg.prefill_steps:
        # use current number of envs inferred from latest obs
        a = _random_env_action(env, device=device)
        try:
            obs2, rew, terminated, truncated, info = env.step(_to_env_action(env, a))
        except Exception:
            print("\n[dreamer_min] env.step() failed during prefill. action shape:", tuple(a.shape))
            traceback.print_exc()
            return
        done = (to_torch(terminated, device=device) | to_torch(truncated, device=device)).to(torch.float32)
        rew_t = to_torch(rew, device=device, dtype=torch.float32).view(-1, 1)

        pol, grid = policy_and_grid_from_obs(obs, device=device)

        # CHANGED: flatten action buffer to (N, act_dim) to store
        act_t = a.to(device=device, dtype=torch.float32).reshape(-1, act_dim)
        done_t = done.view(-1, 1)

        N = pol.shape[0]
        rew_np = rew_t.detach().cpu().numpy().reshape(-1)
        done_np = done_t.detach().cpu().numpy().reshape(-1)
        for i in range(N):
            cur_returns[i] += float(rew_np[i])
            cur_lengths[i] += 1
            if bool(done_np[i]):
                if run_wandb:
                    wandb.log({"episode/return": cur_returns[i], "episode/length": cur_lengths[i], "time/step": global_step})
                cur_returns[i] = 0.0
                cur_lengths[i] = 0

        for i in range(N):
            replay.add_step(i, pol[i], act_t[i], rew_t[i], done_t[i], grid[i] if grid is not None else None)

        obs = obs2
        steps += N
        global_step += N

        if bool(done.any().item()):
            obs, _ = env.reset()

    print(f"[dreamer_min] prefill done. replay steps={replay.num_steps} episodes={replay.num_episodes}")

    # -------- train loop --------
    obs, _ = env.reset()
    pol, _ = policy_and_grid_from_obs(obs, device=device)
    n_envs = pol.shape[0]
    rssm_state = wm.rssm.init_state(n_envs, device=device)
    prev_action = torch.zeros((n_envs, act_dim), device=device)

    for it in range(cfg.iters):
        collected = 0
        while collected < cfg.collect_steps_per_iter:
            with torch.no_grad():
                feat = wm.rssm.feat(rssm_state)
                act_flat = agent.act(feat, deterministic=False)  # (N,act_dim)

                # CHANGED: reshape to env action_space.shape before stepping
                act_env = act_flat.reshape(env.action_space.shape)

            obs2, rew, terminated, truncated, info = env.step(_to_env_action(env, act_env))
            done = (to_torch(terminated, device=device) | to_torch(truncated, device=device)).to(torch.float32).view(-1, 1)
            rew_t = to_torch(rew, device=device, dtype=torch.float32).view(-1, 1)

            pol, grid = policy_and_grid_from_obs(obs, device=device)
            act_t = act_flat.to(device=device, dtype=torch.float32).reshape(-1, act_dim)

            for i in range(pol.shape[0]):
                replay.add_step(i, pol[i], act_t[i], rew_t[i], done[i], grid[i] if grid is not None else None)

            with torch.no_grad():
                embed = wm.encoder(pol)
                post, _ = wm.rssm.obs_step(rssm_state, prev_action, embed)
                rssm_state = post
                prev_action = act_t

            obs = obs2
            collected += pol.shape[0]

            if bool(done.any().item()):
                obs, _ = env.reset()
                pol, _ = policy_and_grid_from_obs(obs, device=device)
                rssm_state = wm.rssm.init_state(pol.shape[0], device=device)
                prev_action = torch.zeros((pol.shape[0], act_dim), device=device)

        if not replay.can_sample(cfg.batch_size, cfg.seq_len):
            continue

        for _ in range(cfg.model_updates_per_iter):
            batch = replay.sample_sequences(cfg.batch_size, cfg.seq_len)
            obs_b = batch.obs.to(device).to(torch.float32)
            act_b = batch.action.to(device).to(torch.float32)
            rew_b = batch.reward.to(device).to(torch.float32)
            done_b = batch.done.to(device).to(torch.float32)

            grid_b = None
            if batch.grid is not None:
                grid_b = batch.grid.to(device).to(torch.float32)
                wm.ensure_grid_head(grid_b.shape[-1])

            out = wm.forward_sequence(obs_b, act_b, predict_grid=(grid_b is not None))

            loss_r = F.mse_loss(out.reward_pred, rew_b) * cfg.reward_scale
            not_done = 1.0 - done_b
            loss_c = F.binary_cross_entropy_with_logits(out.cont_logits, not_done) * cfg.cont_scale

            loss_g = torch.tensor(0.0, device=device)
            if grid_b is not None and out.grid_logits is not None:
                loss_g = F.binary_cross_entropy_with_logits(out.grid_logits, grid_b) * cfg.grid_scale

            loss_kl = out.kl.mean() * cfg.kl_scale
            loss_model = loss_r + loss_c + loss_g + loss_kl

            opt_model.zero_grad(set_to_none=True)
            loss_model.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), 100.0)
            opt_model.step()

            if run_wandb:
                wandb.log({
                    "loss/model": loss_model.item(),
                    "loss/reward": loss_r.item(),
                    "loss/cont": loss_c.item(),
                    "loss/grid": float(loss_g.item()) if isinstance(loss_g, torch.Tensor) else float(loss_g),
                    "loss/kl": loss_kl.item(),
                    "time/step": global_step,
                    "replay/steps": replay.num_steps
                }, commit=False)

        for _ in range(cfg.ac_updates_per_iter):
            batch = replay.sample_sequences(cfg.batch_size, cfg.seq_len)
            obs_b = batch.obs.to(device).to(torch.float32)
            act_b = batch.action.to(device).to(torch.float32)

            with torch.no_grad():
                init = wm.rssm.init_state(cfg.batch_size, device=device)
                warm = min(5, cfg.seq_len)
                out = wm.forward_sequence(obs_b[:, :warm], act_b[:, :warm], init_state=init, predict_grid=False)
                start_state = out.post_states[-1]

                feats, rews_hat, cont_hat, _ = wm.imagine(start_state, agent.actor, horizon=cfg.imagine_horizon)

            actor_loss, critic_loss = agent.actor_critic_loss(
                feats=feats,
                rewards=rews_hat.detach(),
                cont=cont_hat.detach(),
                gamma=cfg.gamma,
                lam=cfg.lam,
            )

            opt_critic.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 100.0)
            opt_critic.step()

            opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 100.0)
            opt_actor.step()

            soft_update_(agent.critic_target, agent.critic, tau=cfg.tau)

            if run_wandb:
                wandb.log({
                    "loss/actor": actor_loss.item(),
                    "loss/critic": critic_loss.item(),
                    "time/step": global_step,
                    "replay/steps": replay.num_steps
                })

        if it % 10 == 0:
            print(f"[it {it:05d}] replay_steps={replay.num_steps} episodes={replay.num_episodes}")

    env.close()

    if run_wandb:
        try:
            torch.save(wm.state_dict(), "wm_final.pt")
            torch.save(agent.state_dict(), "agent_final.pt")
            wandb.save("wm_final.pt")
            wandb.save("agent_final.pt")
            wandb.finish()
        except Exception as e:
            print("[dreamer_min] wandb save/finish failed:", e)


if __name__ == "__main__":
    try:
        main()
    finally:
        # ensure the sim app is closed on exit
        simulation_app.close()