from __future__ import annotations

"""
阶段 2：冻结阶段 1 Actor，训练 GridEstimator。

GridEstimator 从 (policy_obs, action) 序列推断 grid 的隐式表达，
输出的 latent 替代阶段 1 Actor 中 CNN 的 grid_feat。
"""

import argparse
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass, asdict

import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

# --- Launch Isaac Sim first ---
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True, help="Gym task name")
parser.add_argument("--torch_device", type=str, default="cuda:0")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--checkpoint", type=str, required=True, help="Path to stage-1 skrl checkpoint (.pt)")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# --- Now safe to import the rest ---
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.manager_based.gridEstimator.grid_estimator import GridEstimator
from isaaclab_tasks.manager_based.dreamer_min.replay import EpisodeReplay


@dataclass
class Config:
    seed: int = 0
    device: str = "cuda:0"

    # 维度（必须和阶段1 Actor 匹配）
    other_obs_size: int = 20
    grid_side: int = 20
    grid_embed_dim: int = 32
    other_embed_dim: int = 64
    num_actions: int = 2

    # GridEstimator
    gru_hidden_dim: int = 256

    # replay
    capacity_steps: int = 200_000
    seq_len: int = 50
    batch_size: int = 16
    prefill_steps: int = 5000
    collect_steps_per_iter: int = 1000

    # train
    iters: int = 5000
    updates_per_iter: int = 50
    lr: float = 3e-4

    # loss weights
    recon_scale: float = 1.0

    # 评估频率
    eval_interval: int = 100
    eval_episodes: int = 5
    eval_max_steps: int = 1000


# ============================================================
# TeacherActor
# ============================================================
class TeacherActor(nn.Module):
    """阶段1完整 Actor：other_encoder + grid_encoder(CNN) + head"""
    
    def __init__(
        self,
        other_obs_size: int = 20,
        grid_side: int = 20,
        grid_embed_dim: int = 32,
        other_embed_dim: int = 64,
        num_actions: int = 2,
        head_hidden=(64, 32),
        device=None,
    ):
        super().__init__()
        self.other_obs_size = other_obs_size
        self.grid_side = grid_side
        self.grid_obs_size = grid_side * grid_side
        self.grid_embed_dim = grid_embed_dim
        self.other_embed_dim = other_embed_dim
        self.num_actions = num_actions

        self.other_encoder = nn.Sequential(
            nn.Linear(other_obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, other_embed_dim),
            nn.ReLU(),
        )

        self.grid_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算 CNN 输出维度
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_side, grid_side)
            cnn_out_dim = self.grid_encoder(dummy).shape[-1]
        
        # 网格投影到隐空间 
        self.grid_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, grid_embed_dim),
            nn.ReLU(),
        )

        # 
        fused_dim = other_embed_dim + grid_embed_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, head_hidden[0]),
            nn.ReLU(),
            nn.Linear(head_hidden[0], head_hidden[1]),
            nn.ReLU(),
            nn.Linear(head_hidden[1], num_actions),
            nn.Tanh(),
        )

        self.log_std_parameter = nn.Parameter(torch.ones(num_actions) * -2.0)

        if device is not None:
            self.to(device)

    def encode_grid(self, grid_flat: torch.Tensor) -> torch.Tensor:
        """
        只运行 Grid encoder，返回 grid_feat
        
        这是 GridEstimator 需要对齐的目标！
        
        Args:
            grid_flat: (B, 400) 或 (B, T, 400)
        
        Returns:
            grid_feat: (B, 32) 或 (B, T, 32)
        """
        input_shape = grid_flat.shape
        
        # 处理 3D 输入 (B, T, 400)
        if grid_flat.dim() == 3:
            B, T, G = grid_flat.shape
            grid_flat = grid_flat.view(B * T, G)
            reshape_back = True
        else:
            reshape_back = False
        
        # CNN encoder
        x = grid_flat.view(-1, 1, self.grid_side, self.grid_side)
        x = self.grid_encoder(x)
        x = self.grid_proj(x)  # (B*T, 32)
        
        if reshape_back:
            x = x.view(B, T, -1)
        
        return x
    
    def encode_other(self, other_obs: torch.Tensor) -> torch.Tensor:
        """编码本体感知"""
        return self.other_encoder(other_obs)
    
    def forward(self, other_obs: torch.Tensor, grid_flat: torch.Tensor):
        grid_feat = self.encode_grid(grid_flat)
        other_feat = self.encode_other(other_obs)
        combined = torch.cat([other_feat, grid_feat], dim=-1)
        action = self.head(combined)
        return action

    # def forward(self, other_obs: torch.Tensor, grid_flat: torch.Tensor):
    #     other_feat = self.other_encoder(other_obs)
    #     grid_nchw = grid_flat.view(-1, 1, self.grid_side, self.grid_side)
    #     grid_feat = self.grid_proj(self.grid_encoder(grid_nchw))
    #     fused = torch.cat([other_feat, grid_feat], dim=-1)
    #     action_mean = self.head(fused)
    #     return action_mean, self.log_std_parameter

    # def act(self, other_obs: torch.Tensor, grid_flat: torch.Tensor, deterministic: bool = False):
    #     action_mean, log_std = self.forward(other_obs, grid_flat)
    #     if deterministic:
    #         return action_mean
    #     std = torch.exp(log_std)
    #     action = action_mean + std * torch.randn_like(action_mean)
    #     return action.clamp(-1.0, 1.0)

    # def act_with_grid_latent(self, other_obs: torch.Tensor, grid_latent: torch.Tensor, deterministic: bool = False):
    #     other_feat = self.other_encoder(other_obs)
    #     fused = torch.cat([other_feat, grid_latent], dim=-1)
    #     action_mean = self.head(fused)
    #     if deterministic:
    #         return action_mean
    #     std = torch.exp(self.log_std_parameter)
    #     action = action_mean + std * torch.randn_like(action_mean)
    #     return action.clamp(-1.0, 1.0)


def load_teacher_from_skrl_checkpoint(checkpoint_path: str, cfg: Config, device: torch.device):
    """从 skrl checkpoint 加载完整的阶段1 Actor"""
    teacher = TeacherActor(
        other_obs_size=cfg.other_obs_size,
        grid_side=cfg.grid_side,
        grid_embed_dim=cfg.grid_embed_dim,
        other_embed_dim=cfg.other_embed_dim,
        num_actions=cfg.num_actions,
        device=device,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "policy" in ckpt:
        state_dict = ckpt["policy"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    print(f"[load_teacher] checkpoint keys: {list(state_dict.keys())[:20]}...")

    try:
        teacher.load_state_dict(state_dict, strict=False)
        print("[load_teacher] Loaded with strict=False")
    except Exception as e:
        print(f"[load_teacher] Load failed: {e}")

    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    print(f"[load_teacher] other_encoder.0.weight mean={teacher.other_encoder[0].weight.mean().item():.4f}")
    print(f"[load_teacher] head.0.weight mean={teacher.head[0].weight.mean().item():.4f}")
    print(f"[load_teacher] log_std_parameter={teacher.log_std_parameter.data}")

    return teacher


# ============================================================
# 工具函数
# ============================================================
def make_env(task_name: str):
    """创建 IsaacLab 环境"""
    print(f"[make_env] Creating environment: {task_name}")
    
    # 获取 env spec 和 cfg entry point
    spec = gym.spec(task_name)
    kw = dict(spec.kwargs) if spec.kwargs is not None else {}
    
    env_cfg_ep = kw.get("env_cfg_entry_point", None)
    if env_cfg_ep is None:
        raise RuntimeError(
            f"Env spec for '{task_name}' does not provide 'env_cfg_entry_point'. "
            f"Available kwargs keys: {list(kw.keys())}"
        )
    
    # 加载 env cfg
    if isinstance(env_cfg_ep, str):
        # "module.path:ClassName" 格式
        mod_path, cls_name = env_cfg_ep.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(mod_path)
        env_cfg_cls = getattr(mod, cls_name)
        env_cfg = env_cfg_cls()
    else:
        # 直接是 class
        env_cfg = env_cfg_ep()
    
    # 创建环境
    env = gym.make(task_name, cfg=env_cfg)
    print(f"[make_env] Environment created successfully")
    return env



def close_env_safe(env):
    """安全关闭环境"""
    try:
        env.close()
    except Exception as e:
        print(f"[WARNING] Error closing env: {e}")


def extract_obs(obs, device, other_obs_size, grid_obs_size):
    """从 env obs 中提取 other_obs 和 grid"""
    if isinstance(obs, dict):
        if "policy" in obs:
            pol = obs["policy"]
        else:
            pol = list(obs.values())[0]
    else:
        pol = obs

    pol = torch.as_tensor(pol, device=device, dtype=torch.float32)
    if pol.dim() == 1:
        pol = pol.unsqueeze(0)

    total_dim = pol.shape[-1]
    expected_dim = other_obs_size + grid_obs_size

    if total_dim == expected_dim:
        other_obs = pol[:, :other_obs_size]
        grid = pol[:, other_obs_size:]
    else:
        print(f"[WARNING] obs dim {total_dim} != expected {expected_dim}")
        other_obs = pol[:, :other_obs_size]
        grid = pol[:, other_obs_size:other_obs_size + grid_obs_size] if total_dim > other_obs_size else torch.zeros(pol.shape[0], grid_obs_size, device=device)

    if grid is not None:
        grid = (grid > 0.5).float()

    return other_obs, grid


def format_action(env, action: torch.Tensor):
    """将 action tensor 转为 env.step 需要的格式（保持为 torch.Tensor）"""
    space = env.action_space
    a = action.detach()
    
    # Dict action_space
    if hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
        if len(space.spaces) != 1:
            raise RuntimeError(f"Unsupported Dict action_space keys={list(space.spaces.keys())}")
        key = next(iter(space.spaces.keys()))
        sub = space.spaces[key]
        return {key: a.reshape(sub.shape)}
    
    # Box action_space - reshape to match expected shape
    return a.reshape(space.shape)

def weighted_bce_loss(pred_logits, target, pos_weight=10.0):
    """
    加权 BCE loss，给障碍物（正样本）更高权重
    """
    weights = torch.ones_like(target)
    weights[target == 1] = pos_weight
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')
    return (bce * weights).mean()


# ============================================================
# 评估函数
# ============================================================
@torch.no_grad()
def evaluate_with_grid_estimator(
    env, teacher, grid_est, cfg, device,
    num_episodes=5, max_steps=1000, deterministic=True,
):
    """用 GridEstimator 的 latent 替代 CNN 评估"""
    grid_obs_size = cfg.grid_side * cfg.grid_side
    
    teacher.eval()
    grid_est.eval()
    
    episode_returns = []
    episode_lengths = []
    grid_recalls = []      # 改用 recall
    grid_precisions = []   # 改用 precision
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        other_obs, grid_gt = extract_obs(obs, device, cfg.other_obs_size, grid_obs_size)
        
        hidden = grid_est.init_hidden(batch_size=1, device=device)
        prev_action = torch.zeros(1, cfg.num_actions, device=device)
        prev_obs = other_obs.clone()  # 新增：记录上一步 obs
        
        ep_return = 0.0
        ep_length = 0
        ep_tp, ep_fn, ep_fp = 0, 0, 0  # 统计
        
        for step in range(max_steps):
            # 使用新的 step 接口（带 prev_obs）
            grid_latent, grid_logits, hidden = grid_est.step(
                other_obs, prev_action, hidden, prev_obs
            )
            
            action = teacher.act_with_grid_latent(other_obs, grid_latent, deterministic=deterministic)
            
            # 统计 grid 预测
            if grid_gt is not None:
                grid_pred = (torch.sigmoid(grid_logits) > 0.5).float()
                gt = grid_gt.flatten()
                pred = grid_pred.flatten()
                
                ep_tp += ((gt == 1) & (pred == 1)).sum().item()
                ep_fn += ((gt == 1) & (pred == 0)).sum().item()
                ep_fp += ((gt == 0) & (pred == 1)).sum().item()
            
            obs2, rew, terminated, truncated, info = env.step(format_action(env, action))
            
            ep_return += float(rew.item() if hasattr(rew, 'item') else float(rew))
            ep_length += 1
            
            # 更新
            prev_action = action.detach()
            prev_obs = other_obs.clone()  # 保存当前 obs
            other_obs, grid_gt = extract_obs(obs2, device, cfg.other_obs_size, grid_obs_size)
            
            done = terminated or truncated
            if isinstance(done, (torch.Tensor, np.ndarray)):
                done = bool(done.any() if hasattr(done, 'any') else done)
            
            if done:
                break
        
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
        
        # 计算 recall 和 precision
        recall = ep_tp / (ep_tp + ep_fn + 1e-8)
        precision = ep_tp / (ep_tp + ep_fp + 1e-8)
        grid_recalls.append(recall)
        grid_precisions.append(precision)
    
    results = {
        "eval/return_mean": np.mean(episode_returns),
        "eval/return_std": np.std(episode_returns),
        "eval/length_mean": np.mean(episode_lengths),
        "eval/length_std": np.std(episode_lengths),
        "eval/grid_recall_mean": np.mean(grid_recalls),
        "eval/grid_precision_mean": np.mean(grid_precisions),
    }
    
    return results


@torch.no_grad()
def evaluate_with_teacher(
    env, teacher, cfg, device,
    num_episodes=5, max_steps=1000, deterministic=True,
):
    """评估 Teacher（完整 Actor，使用 CNN）的性能"""
    grid_obs_size = cfg.grid_side * cfg.grid_side
    
    teacher.eval()
    
    episode_returns = []
    episode_lengths = []
    
    for ep in range(num_episodes):
        obs, _ = env.reset()
        other_obs, grid_gt = extract_obs(obs, device, cfg.other_obs_size, grid_obs_size)
        
        ep_return = 0.0
        ep_length = 0
        
        for step in range(max_steps):
            action = teacher.act(other_obs, grid_gt, deterministic=deterministic)
            
            obs2, rew, terminated, truncated, info = env.step(format_action(env, action))
            
            ep_return += float(rew.item() if hasattr(rew, 'item') else float(rew))
            ep_length += 1
            
            other_obs, grid_gt = extract_obs(obs2, device, cfg.other_obs_size, grid_obs_size)
            
            done = terminated or truncated
            if isinstance(done, (torch.Tensor, np.ndarray)):
                done = bool(done.any() if hasattr(done, 'any') else done)
            
            if done:
                break
        
        episode_returns.append(ep_return)
        episode_lengths.append(ep_length)
    
    return {
        "teacher/return_mean": np.mean(episode_returns),
        "teacher/return_std": np.std(episode_returns),
        "teacher/length_mean": np.mean(episode_lengths),
        "teacher/length_std": np.std(episode_lengths),
    }


# ============================================================
# 主函数
# ============================================================
def main():
    print("=" * 60)
    print("GridEstimator Training - Stage 2")
    print("=" * 60)
    
    cfg = Config(seed=args_cli.seed, device=args_cli.torch_device)
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    
    print(f"[main] Device: {device}")
    print(f"[main] Task: {args_cli.task}")
    print(f"[main] Checkpoint: {args_cli.checkpoint}")

    grid_obs_size = cfg.grid_side * cfg.grid_side

    # wandb
    run_wandb = False
    if wandb is not None:
        try:
            wandb.init(
                project="grid_estimator",
                name=f"{args_cli.task}-stage2-s{cfg.seed}",
                config=asdict(cfg),
                reinit=True,
            )
            run_wandb = True
            print("[main] wandb initialized")
        except Exception as e:
            print(f"[main] wandb init failed: {e}")

    # -------- 创建环境 --------
    print("[main] Creating environment...")
    env = make_env(args_cli.task)
    print("[main] Resetting environment...")
    obs, _ = env.reset()
    print("[main] Environment ready")

    # DEBUG: 打印 obs 维度
    other_obs_dbg, grid_dbg = extract_obs(obs, device, cfg.other_obs_size, grid_obs_size)
    print(f"[main] other_obs shape={other_obs_dbg.shape}, grid shape={grid_dbg.shape if grid_dbg is not None else None}")
    print(f"[main] Expected: other={cfg.other_obs_size}, grid={grid_obs_size}")

    # -------- 加载 Teacher Actor --------
    print(f"[main] Loading teacher from: {args_cli.checkpoint}")
    teacher = load_teacher_from_skrl_checkpoint(args_cli.checkpoint, cfg, device)
    print("[main] Teacher loaded and frozen.")

    # -------- 创建 GridEstimator --------
    print("[main] Creating GridEstimator...")
    grid_est = GridEstimator(
        obs_dim=cfg.other_obs_size,
        action_dim=cfg.num_actions,
        grid_obs_size=grid_obs_size,
        latent_dim=cfg.grid_embed_dim,
        hidden_dim=cfg.gru_hidden_dim,
    ).to(device)
    print(f"[main] GridEstimator created: {sum(p.numel() for p in grid_est.parameters())} parameters")

    opt = torch.optim.Adam(grid_est.parameters(), lr=cfg.lr)

    # -------- Replay --------
    print("[main] Creating replay buffer...")
    replay = EpisodeReplay(capacity_steps=cfg.capacity_steps, device=device)

    # -------- 统计变量 --------
    cur_returns = defaultdict(float)
    cur_lengths = defaultdict(int)
    global_step = 0
    episodes_total = 0

    # -------- 初始评估：Teacher 基线 --------
    print("[main] Evaluating TEACHER baseline...")
    teacher_baseline = evaluate_with_teacher(
        env, teacher, cfg, device,
        num_episodes=cfg.eval_episodes,
        max_steps=cfg.eval_max_steps,
        deterministic=True
    )
    print(f"[TEACHER BASELINE] return={teacher_baseline['teacher/return_mean']:.2f}±{teacher_baseline['teacher/return_std']:.2f}, "
          f"length={teacher_baseline['teacher/length_mean']:.1f}")
    if run_wandb:
        wandb.log(teacher_baseline)

    # ============================================================
    # Prefill：用 Teacher 采集数据
    # ============================================================
    print(f"[main] Prefilling with TEACHER policy ({cfg.prefill_steps} steps)...")
    obs, _ = env.reset()
    steps = 0
    while steps < cfg.prefill_steps:
        other_obs, grid = extract_obs(obs, device, cfg.other_obs_size, grid_obs_size)

        with torch.no_grad():
            action = teacher.act(other_obs, grid, deterministic=False)

        obs2, rew, terminated, truncated, info = env.step(format_action(env, action))

        done = torch.as_tensor(terminated | truncated, device=device, dtype=torch.float32).view(-1, 1)
        rew_t = torch.as_tensor(rew, device=device, dtype=torch.float32).view(-1, 1)
        act_t = action.view(-1, cfg.num_actions)

        N = other_obs.shape[0]
        for i in range(N):
            replay.add_step(i, other_obs[i], act_t[i], rew_t[i], done[i], grid[i] if grid is not None else None)
            cur_returns[i] += float(rew_t[i].item())
            cur_lengths[i] += 1
            if bool(done[i].item()):
                episodes_total += 1
                if run_wandb:
                    wandb.log({
                        "episode/return": cur_returns[i],
                        "episode/length": cur_lengths[i],
                        "time/step": global_step,
                    })
                cur_returns[i] = 0.0
                cur_lengths[i] = 0

        obs = obs2
        steps += N
        global_step += N

        if steps % 1000 == 0:
            print(f"[prefill] steps={steps}/{cfg.prefill_steps}")

        if bool(done.any().item()):
            obs, _ = env.reset()

    print(f"[main] Prefill done. replay_steps={replay.num_steps}, episodes={episodes_total}")

    # ============================================================
    # 训练循环
    # ============================================================
    print(f"[main] Starting training loop ({cfg.iters} iterations)...")
    best_eval_return = -float('inf')
    
    for it in range(cfg.iters):
        # -------- Collect（用 Teacher）--------
        collected = 0
        while collected < cfg.collect_steps_per_iter:
            other_obs, grid = extract_obs(obs, device, cfg.other_obs_size, grid_obs_size)

            with torch.no_grad():
                action = teacher.act(other_obs, grid, deterministic=False)

            obs2, rew, terminated, truncated, info = env.step(format_action(env, action))

            done = torch.as_tensor(terminated | truncated, device=device, dtype=torch.float32).view(-1, 1)
            rew_t = torch.as_tensor(rew, device=device, dtype=torch.float32).view(-1, 1)
            act_t = action.view(-1, cfg.num_actions)

            N = other_obs.shape[0]
            for i in range(N):
                replay.add_step(i, other_obs[i], act_t[i], rew_t[i], done[i], grid[i] if grid is not None else None)
                cur_returns[i] += float(rew_t[i].item())
                cur_lengths[i] += 1
                if bool(done[i].item()):
                    episodes_total += 1
                    if run_wandb:
                        wandb.log({
                            "episode/return": cur_returns[i],
                            "episode/length": cur_lengths[i],
                            "time/step": global_step,
                        })
                    cur_returns[i] = 0.0
                    cur_lengths[i] = 0

            obs = obs2
            collected += N
            global_step += N

            if bool(done.any().item()):
                obs, _ = env.reset()

        # -------- Train GridEstimator --------
        if not replay.can_sample(cfg.batch_size, cfg.seq_len):
            print(f"[it {it:05d}] Cannot sample yet, skipping training")
            continue

        grid_est.train()
        teacher.eval()

        loss_recon_avg = 0.0
        loss_latent_avg = 0.0
        loss_total_avg = 0.0
        recall_avg = 0.0
        latent_cos_sim_avg = 0.0

        # loss_recon_avg = 0.0
        # obstacle_recall_avg = 0.0
        # obstacle_precision_avg = 0.0
        # obstacle_count_avg = 0.0
        

        # # 前版本
        # for _ in range(cfg.updates_per_iter):
        #     batch = replay.sample_sequences(cfg.batch_size, cfg.seq_len)
        #     obs_b = batch.obs.to(device).float()
        #     act_b = batch.action.to(device).float()
        #     grid_b = batch.grid.to(device).float() if batch.grid is not None else None

        #     if grid_b is None:
        #         print("[WARNING] No grid in batch, skipping")
        #         continue

        #     latent, grid_hat, _ = grid_est(obs_b, act_b)

        #     # 使用加权 BCE loss（障碍物权重更高）
        #     # 改的就是这里 BCE loss + 对齐损失
        #     loss_recon = weighted_bce_loss(grid_hat, grid_b, pos_weight=10.0)
        #     loss = loss_recon * cfg.recon_scale

        #     opt.zero_grad()
        #     loss.backward()
        #     torch.nn.utils.clip_grad_norm_(grid_est.parameters(), 100.0)
        #     opt.step()

        #     loss_recon_avg += loss_recon.item()
            
        #     # 详细统计
        #     with torch.no_grad():
        #         grid_pred = (torch.sigmoid(grid_hat) > 0.5).float()
                
        #         # 障碍物数量
        #         obstacle_count_avg += grid_b.sum().item() / grid_b.numel()
                
        #         # 召回率：真实障碍物中预测正确的比例
        #         obstacle_mask = (grid_b == 1)
        #         if obstacle_mask.sum() > 0:
        #             recall = (grid_pred[obstacle_mask] == 1).float().mean().item()
        #             obstacle_recall_avg += recall
        #         else:
        #             obstacle_recall_avg += 1.0
                
        #         # 精确率：预测为障碍物中真实是障碍物的比例
        #         pred_mask = (grid_pred == 1)
        #         if pred_mask.sum() > 0:
        #             precision = (grid_b[pred_mask] == 1).float().mean().item()
        #             obstacle_precision_avg += precision
        #         else:
        #             obstacle_precision_avg += 1.0

        # n = cfg.updates_per_iter
        # loss_recon_avg /= n
        # obstacle_recall_avg /= n
        # obstacle_precision_avg /= n
        # obstacle_count_avg /= n

        #现版本
        for _ in range(cfg.updates_per_iter):
            batch = replay.sample_sequences(cfg.batch_size, cfg.seq_len)
            obs_b = batch.obs.to(device).float()      # (B, T, 20)
            act_b = batch.action.to(device).float()   # (B, T, 2)
            grid_b = batch.grid.to(device).float()    # (B, T, 400)

            if grid_b is None:
                continue

            # 1. GridEstimator 前向传播
            latent_gru, grid_hat, _ = grid_est(obs_b, act_b)
            # latent_gru: (B, T, 32)
            # grid_hat:   (B, T, 400)

            # 2. 获取 Teacher 的 grid_feat 作为对齐目标
            with torch.no_grad():
                grid_feat_teacher = teacher.encode_grid(grid_b)  # (B, T, 32)

            # 3. 计算损失
            # 3.1 Grid 重建损失（辅助）
            loss_recon = weighted_bce_loss(grid_hat, grid_b, pos_weight=10.0)
            
            # 3.2 Latent 对齐损失（核心！）
            loss_latent = F.mse_loss(latent_gru, grid_feat_teacher)
            
            # 3.3 总损失
            loss = loss_recon * cfg.recon_scale + loss_latent * cfg.latent_scale

            # 4. 反向传播
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(grid_est.parameters(), 100.0)
            opt.step()

            # 5. 记录统计
            loss_recon_avg += loss_recon.item()
            loss_latent_avg += loss_latent.item()
            loss_total_avg += loss.item()
            
            with torch.no_grad():
                # Recall
                grid_pred = (torch.sigmoid(grid_hat) > 0.5).float()
                obstacle_mask = (grid_b == 1)
                if obstacle_mask.sum() > 0:
                    recall = (grid_pred[obstacle_mask] == 1).float().mean().item()
                    recall_avg += recall
                else:
                    recall_avg += 1.0
                
                # Latent 余弦相似度（衡量对齐程度）
                cos_sim = F.cosine_similarity(
                    latent_gru.view(-1, latent_gru.shape[-1]),
                    grid_feat_teacher.view(-1, grid_feat_teacher.shape[-1]),
                    dim=-1
                ).mean().item()
                latent_cos_sim_avg += cos_sim

        # 计算平均值
        n = cfg.updates_per_iter
        loss_recon_avg /= n
        loss_latent_avg /= n
        loss_total_avg /= n
        recall_avg /= n
        latent_cos_sim_avg /= n

        if run_wandb:
            wandb.log({
                "loss/recon": loss_recon_avg,
                # "train/obstacle_recall": obstacle_recall_avg,
                # "train/obstacle_precision": obstacle_precision_avg,
                # "train/obstacle_ratio": obstacle_count_avg,
                "time/step": global_step,
                "replay/steps": replay.num_steps,
                "replay/episodes": replay.num_episodes,
                "total_episodes": episodes_total,
                "loss/latent": loss_latent_avg,
                "loss/total": loss_total_avg,
                "train/recall": recall_avg,
                "train/latent_cos_sim": latent_cos_sim_avg,
            })

        if it % 10 == 0:
            print(f"[it {it:05d}] loss={loss_recon_avg:.4f} "
                #   f"recall={obstacle_recall_avg:.1%} "
                #   f"precision={obstacle_precision_avg:.1%} "
                #   f"obs_ratio={obstacle_count_avg:.2%}")
                  f"loss_recon={loss_recon_avg:.4f} "
                  f"loss_latent={loss_latent_avg:.4f} "
                  f"recall={recall_avg:.1%} "
                  f"cos_sim={latent_cos_sim_avg:.3f}")

        # -------- 评估 --------
        if it % cfg.eval_interval == 0 and it > 0:
            print(f"\n[it {it:05d}] ========== EVALUATION ==========")
            
            eval_results = evaluate_with_grid_estimator(
                env, teacher, grid_est, cfg, device,
                num_episodes=cfg.eval_episodes,
                max_steps=cfg.eval_max_steps,
                deterministic=True
            )
            
            print(f"[GridEstimator] return={eval_results['eval/return_mean']:.2f}±{eval_results['eval/return_std']:.2f}, "
                  f"length={eval_results['eval/length_mean']:.1f}, "
                  f"grid_acc={eval_results.get('eval/grid_accuracy_mean', 0):.4f}")
            
            teacher_return = teacher_baseline['teacher/return_mean']
            gridest_return = eval_results['eval/return_mean']
            gap = (teacher_return - gridest_return) / (abs(teacher_return) + 1e-8) * 100
            print(f"[Gap vs Teacher] {gap:.1f}% (Teacher={teacher_return:.2f}, GridEst={gridest_return:.2f})")
            
            eval_results["eval/gap_vs_teacher_pct"] = gap
            
            if run_wandb:
                wandb.log(eval_results)
            
            if gridest_return > best_eval_return:
                best_eval_return = gridest_return
                torch.save(grid_est.state_dict(), "grid_estimator_best.pt")
                print(f"[it {it:05d}] New best! Saved grid_estimator_best.pt")
            
            print("=" * 50 + "\n")
            
            obs, _ = env.reset()

    # -------- 最终保存 --------
    print("[main] Training complete. Saving final model...")
    close_env_safe(env)
    torch.save(grid_est.state_dict(), "grid_estimator_final.pt")
    print("[main] Saved grid_estimator_final.pt")

    # 最终评估
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    env = make_env(args_cli.task)
    
    final_eval = evaluate_with_grid_estimator(
        env, teacher, grid_est, cfg, device,
        num_episodes=10,
        max_steps=cfg.eval_max_steps,
        deterministic=True
    )
    
    print(f"[FINAL GridEstimator] return={final_eval['eval/return_mean']:.2f}±{final_eval['eval/return_std']:.2f}")
    print(f"[FINAL Teacher]       return={teacher_baseline['teacher/return_mean']:.2f}±{teacher_baseline['teacher/return_std']:.2f}")
    
    gap = (teacher_baseline['teacher/return_mean'] - final_eval['eval/return_mean']) / \
          (abs(teacher_baseline['teacher/return_mean']) + 1e-8) * 100
    print(f"[FINAL Gap]           {gap:.1f}%")
    
    close_env_safe(env)

    if run_wandb:
        wandb.save("grid_estimator_best.pt")
        wandb.save("grid_estimator_final.pt")
        wandb.log({
            "final/return_mean": final_eval['eval/return_mean'],
            "final/return_std": final_eval['eval/return_std'],
            "final/gap_vs_teacher_pct": gap,
        })
        wandb.finish()
    
    print("[main] Done!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()