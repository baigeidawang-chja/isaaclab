"""
DreamerV3 训练入口 —— 适配 IsaacLab 框架。

用法:
    # 从 IsaacLab 根目录
    python -m isaaclab_tasks.user.dreamerv3.main --task Isaac-Cartpole-v0 --num_envs 16

    # 或使用 isaaclab 启动器
    ./isaaclab.sh -p -m isaaclab_tasks.user.dreamerv3.main --task Isaac-Cartpole-v0
"""

import argparse
import copy
import math
import pathlib
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import os

# ─────────────────────────────────────────────
# IsaacLab 导入（必须最早启动 App）
# ─────────────────────────────────────────────
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DreamerV3 + IsaacLab")
parser.add_argument("--task", type=str, default="Isaac-Cartpole-v0", help="IsaacLab 任务名称")
parser.add_argument("--num_envs", type=int, default=0, help="并行环境数量；设为0时使用 configs.yaml 里的 run.envs")
parser.add_argument("--seed", type=int, default=42, help="随机种子")

# WandB 参数
parser.add_argument("--wandb_project", type=str, default="dreamerv3-isaaclab", 
                    help="WandB project name")
parser.add_argument("--wandb_entity", type=str, default=None, 
                    help="WandB entity/username")
parser.add_argument("--wandb_name", type=str, default=None, 
                    help="WandB run name (default: task_timestamp)")
parser.add_argument("--wandb_tags", type=str, default="", 
                    help="Comma-separated tags for WandB")
parser.add_argument("--wandb_notes", type=str, default="", 
                    help="Notes for WandB run")
parser.add_argument("--no_wandb", action="store_true", 
                    help="Disable WandB logging")

# 新增：使用 IsaacLab/Hydra 的 task 配置解析
parser.add_argument(
    "--task_config",
    type=str,
    default=None,
    help=(
        "Hydra task config 的 entry point 或路径。"
        "例如: isaaclab_tasks.manager_based.classic.cartpole:CartpoleEnvCfg "
        "或某个 *.yaml 配置文件路径。"
    ),
)

parser.add_argument("--config", type=str, default="defaults", help="configs.yaml 中的配置名称，逗号分隔")
parser.add_argument("--logdir", type=str, default=None, help="日志目录，默认 runs/dreamerv3/{timestamp}")

parser.add_argument("--max_steps", type=int, default=int(1e8), help="总环境交互步数")
parser.add_argument("--eval_every", type=int, default=10000, help="每 N 步评估一次")
parser.add_argument("--save_every", type=int, default=5000, help="每 N 步保存 checkpoint")
parser.add_argument("--log_every", type=int, default=1000, help="每 N 步打印日志")
parser.add_argument(
    "--max_train_burst",
    type=int,
    default=0,
    help="单次环境步之后最多连续执行多少个训练 step；设为0时按 num_envs、train_ratio、batch_length 自动计算",
)
parser.add_argument(
    "--ui_yield_every",
    type=int,
    default=1,
    help="每多少个环境循环主动 pump 一次 Isaac Sim UI；仅在非 headless 下生效",
)
parser.add_argument(
    "--ui_sleep",
    type=float,
    default=0.0,
    help="每次 UI pump 后额外 sleep 的秒数；仅在非 headless 下生效",
)
parser.add_argument(
    "--prefill_steps",
    type=int,
    default=0,
    help="开始训练前至少先采样多少个 environment steps；0 表示不额外预填充",
)
parser.add_argument(
    "--collect_env_loops_per_iter",
    type=int,
    default=10,
    help="每次训练前连续执行多少轮向量化 env.step()；更接近官方 driver(..., steps=10) 的 collect-then-train 节奏",
)
parser.add_argument(
    "--debug_data_once",
    action="store_true",
    help="在第一次 train_step 前后打印一次 batch 与模型预测统计，用于排查数据链问题",
)
parser.add_argument(
    "--debug_data_steps",
    type=str,
    default="",
    help="额外在哪些 train_steps 上打印数据链快照，例如 '1000,5000'；会和 --debug_data_once 一起生效",
)

# 设备参数
AppLauncher.add_app_launcher_args(parser)
args, unknown = parser.parse_known_args()

import sys
sys.argv = [sys.argv[0]] + unknown

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Isaac Sim 启动后才能导入以下模块
import gymnasium as gym
import isaaclab_tasks
import isaaclab_tasks.user.dreamerv3.env

from isaaclab_tasks.user.dreamerv3.envwrapper import IsaacLabDreamerWrapper
from isaaclab_tasks.user.dreamerv3.agent import Agent, SimpleSpace
from isaaclab_tasks.user.dreamerv3.runtime_official import OfficialRunner

# Isaac Lab task config loader
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg, load_cfg_from_registry
from isaaclab.utils.io import dump_yaml

# 尝试导入WandB
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("[WARNING] wandb not installed. Install with: pip install wandb")
    WANDB_AVAILABLE = False


def summarize_tensor(name: str, value: torch.Tensor) -> str:
    x = value.detach().float()
    return (
        f"{name}: shape={tuple(value.shape)} "
        f"mean={x.mean().item():.5f} std={x.std().item():.5f} "
        f"min={x.min().item():.5f} max={x.max().item():.5f}"
    )


def encode_stepid(step_ids: torch.Tensor, width: int = 20) -> torch.Tensor:
    """Encode integer step ids to fixed-width uint8 ASCII tensor, shape (..., width)."""
    flat = step_ids.reshape(-1).detach().cpu().long().tolist()
    arr = np.zeros((len(flat), width), dtype=np.uint8)
    for i, sid in enumerate(flat):
        s = str(max(0, int(sid))).rjust(width, "0")[-width:]
        arr[i] = np.frombuffer(s.encode("ascii"), dtype=np.uint8)
    out = torch.from_numpy(arr).to(step_ids.device)
    return out.reshape(*step_ids.shape, width)


# ─────────────────────────────────────────────
# 增强的日志工具（集成WandB）
# ─────────────────────────────────────────────

class EnhancedLogger:
    def __init__(self, logdir: str, args, config):
        self.logdir = pathlib.Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._jsonl_path = self.logdir / "metrics.jsonl"
        self._scores_path = self.logdir / "scores.jsonl"
        self._step = 0
        
        # TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb = SummaryWriter(str(self.logdir / "tb"))
        except ImportError:
            self._tb = None
        
        # WandB 初始化
        self.wandb_enabled = WANDB_AVAILABLE and not args.no_wandb
        self.wandb_run = None
        
        if self.wandb_enabled:
            self._initialize_wandb(args, config)
        
        # 性能监控
        self.start_time = time.time()
        self.episode_scores = []
        self.episode_lengths = []
        
        print(f"[LOGGER] Logging to: {logdir}")
        if self.wandb_enabled and self.wandb_run:
            print(f"[LOGGER] WandB enabled: {self.wandb_run.name}")
            print(f"[LOGGER] WandB URL: {self.wandb_run.url}")
        else:
            print("[LOGGER] WandB disabled or not available")

    def _initialize_wandb(self, args, config):
        """初始化WandB运行"""
        try:
            # 准备配置
            wandb_config = {
                "task": args.task,
                "num_envs": args.num_envs,
                "seed": args.seed,
                "max_steps": args.max_steps,
                "log_every": args.log_every,
                "save_every": args.save_every,
                "eval_every": args.eval_every,
                "config_name": args.config,
            }
            
            # 合并Dreamer配置
            wandb_config.update(config)
            
            # 准备标签
            tags = ["dreamerv3", "isaaclab", args.task]
            if args.wandb_tags:
                tags.extend([tag.strip() for tag in args.wandb_tags.split(",")])
            
            # 运行名称
            run_name = args.wandb_name
            if run_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_name = f"{args.task}_{timestamp}"
            
            # 初始化WandB
            self.wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=run_name,
                config=wandb_config,
                tags=tags,
                notes=args.wandb_notes,
                dir=str(self.logdir),
                save_code=True,
                settings=wandb.Settings(start_method="thread")
            )
            
            print(f"[WandB] Initialized run: {self.wandb_run.name} ({self.wandb_run.id})")
            
        except Exception as e:
            print(f"[WandB] Failed to initialize: {e}")
            self.wandb_enabled = False

    def step(self, n=1):
        self._step += n
        return self._step

    def log_metrics(self, metrics: dict, prefix: str = "", commit=True):
        import json
        
        # 准备记录数据
        row = {"step": self._step}
        wandb_metrics = {}
        
        for k, v in metrics.items():
            key = f"{prefix}/{k}" if prefix else k
            
            # 标量与媒体分流
            if isinstance(v, torch.Tensor):
                if v.numel() == 1:
                    v = v.item()
                else:
                    v = v.detach().cpu().numpy()

            is_scalar = isinstance(v, (int, float, np.integer, np.floating))
            if is_scalar:
                v = float(v)
                row[key] = v
                wandb_metrics[key] = v
                if self._tb is not None:
                    self._tb.add_scalar(key, v, self._step)
                continue

            # 4D uint8 video grid: (T, H, W, C), official openloop layout.
            is_video_grid = isinstance(v, np.ndarray) and v.ndim == 4 and v.shape[-1] in (1, 3, 4)
            if is_video_grid:
                arr = v.astype(np.uint8, copy=False)
                wandb_metrics[key] = wandb.Video(arr, fps=10, format="mp4") if self.wandb_enabled and self.wandb_run is not None else arr
                if self._tb is not None:
                    tb_vid = torch.from_numpy(arr).permute(0, 3, 1, 2).unsqueeze(0)  # (1,T,C,H,W)
                    self._tb.add_video(key, tb_vid, self._step, fps=10)
        
        # 本地JSONL文件
        with open(self._jsonl_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        
        # WandB
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                self.wandb_run.log(wandb_metrics, step=self._step, commit=commit)
            except Exception as e:
                print(f"[WandB] Logging failed: {e}")

    def log_episode(self, score: float, length: int):
        import json
        
        # 本地文件
        row = {"step": self._step, "score": score, "length": length}
        with open(self._scores_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        
        # TensorBoard
        if self._tb is not None:
            self._tb.add_scalar("episode/score", score, self._step)
            self._tb.add_scalar("episode/length", length, self._step)
        
        # WandB
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                self.wandb_run.log({
                    "episode/score": score,
                    "episode/length": length,
                    "episode/total": self.wandb_run.summary.get("episode/total", 0) + 1
                }, step=self._step)
            except Exception as e:
                print(f"[WandB] Episode logging failed: {e}")
        
        # 存储用于统计
        self.episode_scores.append(score)
        self.episode_lengths.append(length)

    def print(self, msg: str):
        print(f"[Step {self._step:>10d}] {msg}")

    def close(self):
        # 关闭TensorBoard
        if self._tb is not None:
            self._tb.close()
        
        # 关闭WandB
        if self.wandb_enabled and self.wandb_run is not None:
            try:
                # 保存模型检查点
                self._save_wandb_artifacts()
                self.wandb_run.finish()
                print("[WandB] Run finished")
            except Exception as e:
                print(f"[WandB] Finish failed: {e}")
    
    def _save_wandb_artifacts(self):
        """保存模型检查点到WandB"""
        if not self.wandb_enabled or self.wandb_run is None:
            return
        
        try:
            # 保存配置文件
            config_files = [
                self.logdir / "env_cfg.yaml",
                self.logdir / "dreamer_cfg.yaml",
            ]
            
            for config_file in config_files:
                if config_file.exists():
                    self.wandb_run.save(str(config_file))
            
            # 保存检查点
            ckpt_dir = self.logdir / "checkpoints"
            if ckpt_dir.exists():
                for ckpt_file in ckpt_dir.glob("*.pt"):
                    artifact = wandb.Artifact(
                        name=f"checkpoint_{ckpt_file.stem}",
                        type="model",
                        description=f"Checkpoint at step {ckpt_file.stem.split('_')[-1]}"
                    )
                    artifact.add_file(str(ckpt_file))
                    self.wandb_run.log_artifact(artifact)
            
            print("[WandB] Artifacts saved")
            
        except Exception as e:
            print(f"[WandB] Artifact save failed: {e}")


# ─────────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────────

class ReplayBuffer:
    """Ring buffer storing per-env trajectories.
    Storage layout: (time, env, ...). Sampling returns (batch, time, ...).
    """

    def __init__(self, capacity: int, sequence_length: int, replay_cfg: dict | None = None):
        self.capacity = int(capacity)
        self.seq_len = int(sequence_length)
        replay_cfg = replay_cfg or {}
        fracs = replay_cfg.get("fracs", {})
        self._frac_uniform = float(fracs.get("uniform", 1.0))
        self._frac_priority = float(fracs.get("priority", 0.0))
        self._frac_recency = float(fracs.get("recency", 0.0))
        frac_sum = self._frac_uniform + self._frac_priority + self._frac_recency
        if frac_sum <= 0:
            self._frac_uniform, self._frac_priority, self._frac_recency = 1.0, 0.0, 0.0
        else:
            self._frac_uniform /= frac_sum
            self._frac_priority /= frac_sum
            self._frac_recency /= frac_sum
        prio_cfg = replay_cfg.get("prio", {})
        self._prio_exponent = float(prio_cfg.get("exponent", 0.8))
        self._prio_maxfrac = float(prio_cfg.get("maxfrac", 0.5))
        self._prio_zero_on_sample = bool(prio_cfg.get("zero_on_sample", True))
        initial = prio_cfg.get("initial", 1.0)
        if isinstance(initial, str) and initial.lower() in ("inf", "+inf"):
            self._prio_initial = float("inf")
        else:
            self._prio_initial = float(initial)
        self._recexp = float(replay_cfg.get("recexp", 1.0))
        self._online = bool(replay_cfg.get("online", True))
        self._chunksize = int(replay_cfg.get("chunksize", 1024))
        self._storage: dict[str, torch.Tensor] = {}
        self._pos = 0
        self._size = 0
        self._num_envs = None
        self._priority: torch.Tensor | None = None

    def _alloc_if_needed(self, data: dict):
        if self._storage:
            return
        # infer num_envs from first tensor
        first = next(iter(data.values()))
        if not isinstance(first, torch.Tensor) or first.dim() < 1:
            raise ValueError("ReplayBuffer expects tensor values with leading dim = num_envs.")
        self._num_envs = int(first.shape[0])

        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            if v.shape[0] != self._num_envs:
                raise ValueError(f"Key '{k}' has inconsistent num_envs: {v.shape[0]} vs {self._num_envs}")
            # 在时间维度上预留 capacity，环境维度上预留 num_envs，特征维度保持不变
            # store as (capacity, num_envs, *feat)
            feat_shape = tuple(v.shape[1:])  # may be empty for scalars -> ()
            self._storage[k] = torch.zeros(
                (self.capacity, self._num_envs, *feat_shape), dtype=v.dtype
            )
        self._priority = torch.ones((self.capacity, self._num_envs), dtype=torch.float32)

    def add(self, data: dict):
        """Add one environment step for all envs.
        data[key] shape: (num_envs, ...)
        """
        self._alloc_if_needed(data)
        assert self._priority is not None

        for k, v in data.items():
            if not isinstance(v, torch.Tensor):
                v = torch.as_tensor(v)
            v = v.detach().cpu()
            # ensure leading dim is num_envs
            if v.shape[0] != self._num_envs:
                raise ValueError(f"Key '{k}' has wrong leading dim: {v.shape[0]} vs {self._num_envs}")
            self._storage[k][self._pos].copy_(v)

        if self._size == 0:
            insert_prio = 1.0
        elif np.isinf(self._prio_initial):
            insert_prio = float(self._priority[: self._size].max().item())
        else:
            insert_prio = self._prio_initial
        self._priority[self._pos].fill_(max(insert_prio, 1e-8))

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    @property
    def total(self):
        # number of time steps stored (not transitions)
        return self._size

    def can_sample(self, batch_size: int, seq_len: int) -> bool:
        # need at least seq_len contiguous steps in time dimension
        return self._size >= seq_len

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        device="cpu",
        stream_state: dict | None = None,
        stride: int | None = None,
    ) -> tuple[dict, dict]:
        if self._size < seq_len:
            raise ValueError(f"Buffer has {self._size} steps, need seq_len={seq_len}")
        assert self._priority is not None
        if stride is None:
            stride = seq_len
        stride = int(stride)

        # sample start indices in the valid range of the ring buffer
        # we map logical time [0, _size) into physical indices ending at _pos-1
        max_start = self._size - seq_len
        oldest = (self._pos - self._size) % self.capacity
        starts = []
        env_ids = []
        consec = []
        mix = torch.tensor([self._frac_uniform, self._frac_priority, self._frac_recency], dtype=torch.float32)
        modes = torch.multinomial(mix, batch_size, replacement=True)  # 0=uniform,1=priority,2=recency

        start_phys = (oldest + torch.arange(max_start + 1)) % self.capacity
        start_prios = self._priority[start_phys].clamp(min=1e-8)  # (S, E)
        prio_flat = (start_prios ** self._prio_exponent).reshape(-1)
        if prio_flat.sum() <= 0:
            prio_flat = torch.ones_like(prio_flat)
        prio_prob = prio_flat / prio_flat.sum()
        uniform_prob = torch.ones_like(prio_prob) / prio_prob.numel()
        prio_mix = (1.0 - self._prio_maxfrac) * uniform_prob + self._prio_maxfrac * prio_prob
        prio_mix = prio_mix / prio_mix.sum()

        recency_rank = torch.arange(max_start + 1, dtype=torch.float32) + 1.0
        recency_w = (recency_rank / recency_rank.max()) ** max(self._recexp, 1e-6)
        if recency_w.sum() <= 0:
            recency_w = torch.ones_like(recency_w)
        if self._online:
            recent_span = int(max(1, min(max_start + 1, self._chunksize)))
            recent_from = max_start + 1 - recent_span
            recent_choices = torch.arange(recent_from, max_start + 1, dtype=torch.long)
        else:
            recent_choices = None

        prev_starts = stream_state.get("start") if stream_state is not None else None
        prev_envs = stream_state.get("env") if stream_state is not None else None
        prev_consec = stream_state.get("consec") if stream_state is not None else None
        for b in range(batch_size):
            continued = False
            if prev_starts is not None and prev_envs is not None and prev_consec is not None:
                next_s = int(prev_starts[b].item()) + stride
                e = int(prev_envs[b].item())
                if next_s <= max_start:
                    s = next_s
                    c = int(prev_consec[b].item()) + 1
                    continued = True
            if not continued:
                mode = int(modes[b].item())
                if mode == 1:
                    flat_idx = torch.multinomial(prio_mix, 1, replacement=True).item()
                    s = flat_idx // self._num_envs
                    e = flat_idx % self._num_envs
                    if self._prio_zero_on_sample:
                        phys_s = int((oldest + s) % self.capacity)
                        self._priority[phys_s, e] = 1e-8
                elif mode == 2:
                    s = int(torch.multinomial(recency_w, 1, replacement=True).item())
                    e = int(torch.randint(0, self._num_envs, (1,)).item())
                else:
                    if recent_choices is not None and recent_choices.numel() > 0:
                        s = int(recent_choices[torch.randint(0, recent_choices.numel(), (1,)).item()].item())
                    else:
                        s = int(torch.randint(0, max_start + 1, (1,)).item())
                    e = int(torch.randint(0, self._num_envs, (1,)).item())
                c = 0
            starts.append(s)
            env_ids.append(e)
            consec.append(c)

        start = torch.tensor(starts, dtype=torch.long)
        env_ids = torch.tensor(env_ids, dtype=torch.long)
        consec = torch.tensor(consec, dtype=torch.int32)

        batch = {}
        for k, stor in self._storage.items():
            # stor: (capacity, num_envs, *feat)
            seq_list = []
            for b in range(batch_size):
                t0 = int(start[b].item())
                phys = (oldest + torch.arange(t0, t0 + seq_len)) % self.capacity  # (T,)
                e = int(env_ids[b].item())
                seq = stor[phys, e]  # (T, *feat)
                seq_list.append(seq)
            batch[k] = torch.stack(seq_list, dim=0).to(device)  # (B, T, *feat)
        # Official-style stream metadata for replay context routing.
        batch["consec"] = consec.unsqueeze(1).expand(-1, seq_len).to(device)
        batch["_meta_start"] = start.to(device)
        batch["_meta_env"] = env_ids.to(device)
        new_state = {"start": start, "env": env_ids, "consec": consec}
        return batch, new_state

    def update_priority(self, start: torch.Tensor, env_ids: torch.Tensor, priority: torch.Tensor):
        if self._priority is None:
            return
        oldest = (self._pos - self._size) % self.capacity
        start_cpu = start.detach().cpu().long()
        env_cpu = env_ids.detach().cpu().long()
        prio_cpu = priority.detach().cpu().float().clamp(min=1e-8)
        for i in range(start_cpu.shape[0]):
            logical = int(start_cpu[i].item())
            env_id = int(env_cpu[i].item())
            phys = int((oldest + logical) % self.capacity)
            self._priority[phys, env_id] = prio_cpu[i]

    def update_context(
        self,
        start: torch.Tensor,
        env_ids: torch.Tensor,
        updates: dict[str, torch.Tensor],
        stepid: torch.Tensor | None = None,
    ):
        if not updates:
            return
        oldest = (self._pos - self._size) % self.capacity
        start_cpu = start.detach().cpu().long()
        env_cpu = env_ids.detach().cpu().long()
        stepid_cpu = stepid.detach().cpu() if isinstance(stepid, torch.Tensor) else None
        upd_cpu = {k: v.detach().cpu() for k, v in updates.items() if isinstance(v, torch.Tensor)}
        if not upd_cpu:
            return
        T = next(iter(upd_cpu.values())).shape[1]
        for k, v in upd_cpu.items():
            if v.shape[1] != T:
                raise ValueError(f"Replay update key '{k}' has inconsistent time dim {v.shape[1]} vs {T}")
            if k not in self._storage:
                feat_shape = tuple(v.shape[2:])
                self._storage[k] = torch.zeros((self.capacity, self._num_envs, *feat_shape), dtype=v.dtype)

        for b in range(start_cpu.shape[0]):
            s0 = int(start_cpu[b].item())
            e = int(env_cpu[b].item())
            phys = (oldest + torch.arange(s0, s0 + T)) % self.capacity
            if stepid_cpu is not None and "stepid" in self._storage:
                stored = self._storage["stepid"][phys, e]
                target = stepid_cpu[b]
                if stored.shape != target.shape or not torch.equal(stored, target):
                    continue
            for k, v in upd_cpu.items():
                self._storage[k][phys, e] = v[b]


# ─────────────────────────────────────────────
# 加载 Dreamer 配置（仍来自本地 configs.yaml）
# ─────────────────────────────────────────────

def load_local_dreamer_config(config_names: str) -> dict:
    """继续使用你当前 dreamerv3/configs.yaml 作为 agent 超参来源。"""
    import ruamel.yaml as yaml

    config_path = pathlib.Path(__file__).parent / "configs.yaml"
    with open(config_path, "r") as f:
        all_configs = yaml.YAML(typ="safe").load(f)

    config = dict(all_configs["defaults"])
    for name in config_names.split(","):
        name = name.strip()
        if name and name != "defaults" and name in all_configs:
            config = deep_update(config, all_configs[name])
    return config


def deep_update(base: dict, update: dict) -> dict:
    result = base.copy()
    for k, v in update.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_update(result[k], v)
        else:
            result[k] = v
    return result


# ─────────────────────────────────────────────
# 训练主循环
# ─────────────────────────────────────────────

def train(args, dreamer_cfg: dict, env_cfg):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 确认配置值
    batch_size = dreamer_cfg.get('batch_size', 2)
    batch_length = dreamer_cfg.get('batch_length', 16)
    num_envs = dreamer_cfg.get('run', {}).get('envs', 4)
    dyn_cfg = dreamer_cfg.get('agent', {}).get('dyn', {})
    print(f"[CONFIG] batch_size={batch_size}, batch_length={batch_length}, num_envs={num_envs}")
    print(f"[CONFIG] dyn: deter={dyn_cfg.get('deter')}, stoch={dyn_cfg.get('stoch')}, classes={dyn_cfg.get('classes')}")
    print(f"[CONFIG] imag_last={dreamer_cfg.get('agent', {}).get('imag_last')}, imag_length={dreamer_cfg.get('agent', {}).get('imag_length')}")
    
    # 日志目录
    if args.logdir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logdir = f"runs/dreamerv3/{args.task}/{timestamp}"
    else:
        logdir = args.logdir

    # 创建增强的Logger（包含WandB）
    logger = EnhancedLogger(logdir, args, dreamer_cfg)

    agent_cls = Agent

    for line in agent_cls.banner:
        print(line)
    print(f"Task: {args.task}")
    print(f"Num envs: {args.num_envs}")
    print(f"Device: {device}")
    print("Agent kind: dreamer")
    print(f"Logdir: {logdir}")
    print(f"WandB enabled: {logger.wandb_enabled}")

    # 保存环境 cfg
    pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
    env_cfg_path = os.path.join(logdir, "env_cfg.yaml")
    dump_yaml(env_cfg_path, env_cfg)

    # 保存 dreamer cfg
    import ruamel.yaml as ryaml
    with open(pathlib.Path(logdir) / "dreamer_cfg.yaml", "w") as f:
        ryaml.YAML().dump(dreamer_cfg, f)

    replay_cfg = dreamer_cfg.get("replay", {})
    buffer_capacity = int(float(replay_cfg.get("size", 1e6)))
    batch_size = int(dreamer_cfg.get("batch_size", 16))
    batch_length = int(dreamer_cfg.get("batch_length", 64))
    report_length = int(dreamer_cfg.get("report_length", batch_length))
    replay_context = int(dreamer_cfg.get("replay_context", 0))
    consec_train = int(dreamer_cfg.get("consec_train", 1))
    consec_report = int(dreamer_cfg.get("consec_report", 1))
    train_ratio = float(dreamer_cfg.get("run", {}).get("train_ratio", 32.0))
    eval_envs = int(dreamer_cfg.get("run", {}).get("eval_envs", 1))
    eval_eps = int(dreamer_cfg.get("run", {}).get("eval_eps", 1))
    report_every = int(dreamer_cfg.get("run", {}).get("report_every", 0))
    report_batches = int(dreamer_cfg.get("run", {}).get("report_batches", 1))
    eval_every = int(args.eval_every)

    env = gym.make(args.task, cfg=env_cfg)
    env = IsaacLabDreamerWrapper(env, device=device)
    eval_env = None
    if eval_every > 0 and eval_eps > 0:
        try:
            eval_env_cfg = copy.deepcopy(env_cfg)
            if hasattr(eval_env_cfg, "scene") and hasattr(eval_env_cfg.scene, "num_envs"):
                eval_env_cfg.scene.num_envs = eval_envs
            eval_env = gym.make(args.task, cfg=eval_env_cfg)
            eval_env = IsaacLabDreamerWrapper(eval_env, device=device)
        except RuntimeError as exc:
            msg = str(exc)
            if "Simulation context already exists" in msg:
                eval_env = None
                eval_every = -1
                print(
                    "[WARN] IsaacLab does not allow creating a second simulation context in-process; "
                    "periodic eval env is disabled for this run. Use play.py for separate evaluation."
                )
            else:
                raise

    obs_space = env.obs_space
    act_space = env.act_space

    print(f"Observation space: {obs_space}")
    print(f"Action space: {act_space}")

    agent = agent_cls(obs_space, act_space, dreamer_cfg.get("agent", {})).to(device)

    interactive_cfg = dreamer_cfg.get("ui", {})
    max_train_burst = int(interactive_cfg.get("max_train_burst", args.max_train_burst))
    ui_yield_every = int(interactive_cfg.get("yield_every", args.ui_yield_every))
    ui_sleep = float(interactive_cfg.get("sleep", args.ui_sleep))
    collect_env_loops_per_iter = int(
        interactive_cfg.get("collect_env_loops_per_iter", args.collect_env_loops_per_iter)
    )
    # Match official driver semantics: collect continuously step-by-step and train by ratio budget.
    collect_env_loops_per_iter = 1
    is_headless = bool(getattr(args, "headless", False))
    updates_per_env_step = train_ratio / max(batch_size * batch_length, 1)
    if max_train_burst <= 0:
        # Official semantics: updates are ratio-driven from budget, not hard-capped by burst.
        max_train_burst = -1
    print(
        f"[CONFIG] train_ratio={train_ratio}, updates_per_env_step={updates_per_env_step:.6f}, "
        f"effective_max_train_burst={max_train_burst}, collect_env_loops_per_iter={collect_env_loops_per_iter}"
    )

    buffer_capacity_transitions = int(float(replay_cfg.get("size", 1e6)))
    buffer_capacity_steps = max(1, buffer_capacity_transitions // args.num_envs)
    buffer = ReplayBuffer(capacity=buffer_capacity_steps, sequence_length=batch_length, replay_cfg=replay_cfg)

    carry = agent.init_carry(args.num_envs, device)
    obs, info = env.reset()
    prev_progress_gain = None
    if isinstance(obs, dict) and "local_tracking_state" in obs:
        prev_progress_gain = obs["local_tracking_state"][:, 2].detach().clone()

    episode_returns = torch.zeros(args.num_envs, device=device)
    episode_lengths = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
    env_step_ids = torch.zeros(args.num_envs, dtype=torch.int64, device=device)
    total_episodes = 0

    global_step = 0
    train_steps = 0
    train_carry = agent.init_carry(batch_size, device)
    report_carry = agent.init_carry(batch_size, device)
    next_eval_step = max(eval_every, 1) if eval_every > 0 else -1
    next_report_step = max(report_every, 1) if report_every > 0 else -1
    best_eval_score = -float("inf")
    train_budget = 0.0
    start_time = time.time()
    metrics_accum = defaultdict(list)
    env_loops = 0
    train_stream_state = None
    report_stream_state = None
    debug_steps = set()
    if args.debug_data_once:
        debug_steps.add(0)
    if args.debug_data_steps.strip():
        for item in args.debug_data_steps.split(","):
            item = item.strip()
            if item:
                debug_steps.add(int(item))

    print("\n=== Training started ===\n")

    def pump_ui():
        if is_headless:
            return
        if simulation_app is None:
            return
        try:
            simulation_app.update()
        except Exception:
            return
        if ui_sleep > 0:
            time.sleep(ui_sleep)

    def run_eval_once() -> dict:
        if eval_env is None:
            return {}
        was_training = agent.training
        agent.eval()
        ecarry = agent.init_carry(eval_env.num_envs, device)
        eobs, _ = eval_env.reset()
        eret = torch.zeros(eval_env.num_envs, device=device)
        elen = torch.zeros(eval_env.num_envs, dtype=torch.int64, device=device)
        completed_scores = []
        completed_lengths = []

        while len(completed_scores) < eval_eps and simulation_app.is_running():
            with torch.no_grad():
                ecarry, eaction, _ = agent.policy(ecarry, eobs, mode="eval")
            enext_obs, erew, eterm, etrunc, _ = eval_env.step(eaction)
            edone = eterm | etrunc
            eret += erew
            elen += 1
            if edone.any():
                mask = edone.bool()
                for i in range(eval_env.num_envs):
                    if mask[i] and len(completed_scores) < eval_eps:
                        completed_scores.append(float(eret[i].item()))
                        completed_lengths.append(int(elen[i].item()))
                eret[mask] = 0
                elen[mask] = 0
            eobs = enext_obs

        if was_training:
            agent.train()
        if not completed_scores:
            return {}
        return {
            "eval/score_mean": float(np.mean(completed_scores)),
            "eval/score_max": float(np.max(completed_scores)),
            "eval/length_mean": float(np.mean(completed_lengths)),
            "eval/episodes": float(len(completed_scores)),
        }

    # Official-congruent runtime entry: Driver + Replay + ConsecStream callbacks.
    runner = OfficialRunner(
        env=env,
        eval_env=eval_env,
        agent=agent,
        buffer=buffer,
        logger=logger,
        args=args,
        dreamer_cfg=dreamer_cfg,
        simulation_app=simulation_app,
        device=device,
        init_obs=obs,
        init_carry=carry,
        encode_stepid=encode_stepid,
    )
    runner.run(max_steps=args.max_steps, logdir=logdir)
    env.close()
    if eval_env is not None:
        eval_env.close()
    logger.close()
    simulation_app.close()
    return

# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main():
    """主函数：解析配置并启动训练"""
    print("=== DreamerV3 IsaacLab Training ===")
    
    # 加载 DreamerV3 配置
    dreamer_cfg = load_local_dreamer_config(args.config)
    print(f"[CONFIG] Loaded DreamerV3 config: {args.config}")
    if args.num_envs <= 0:
        args.num_envs = int(dreamer_cfg.get("run", {}).get("envs", 1))
        print(f"[CONFIG] num_envs not set on CLI, using config value: {args.num_envs}")
    
    # 加载环境配置
    if args.task_config:
        env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
        if isinstance(args.task_config, str) and args.task_config.endswith(".yaml"):
            import yaml
            with open(args.task_config, "r", encoding="utf-8") as f:
                overrides = yaml.safe_load(f)
            env_cfg.from_dict(overrides) # type: ignore
        else:
            print(
                f"[CONFIG] task_config='{args.task_config}' is not supported as a direct override entry point here; "
                "falling back to the registered task config."
            )
        env_cfg.sim.device = args.device # type: ignore
        if hasattr(env_cfg, "scene") and hasattr(env_cfg.scene, "num_envs"): # type: ignore
            env_cfg.scene.num_envs = args.num_envs #type: ignore
        print(f"[CONFIG] Loaded environment config for task: {args.task}")
    else:
        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        print(f"[CONFIG] Using registered environment config for {args.task}")
    
    # 启动训练
    try:
        train(args, dreamer_cfg, env_cfg)
    except KeyboardInterrupt:
        print("\n=== Training interrupted by user ===")
    except Exception as e:
        print(f"\n=== Training failed with error: {e} ===")
        import traceback
        traceback.print_exc()
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    main()
