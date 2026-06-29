from __future__ import annotations

"""
可视化 GridEstimator 预测的 grid vs 真实 grid
用法:
    python visualize_grid.py \
        --task Isaac-Navigation-Car-v0 \
        --teacher_checkpoint /path/to/teacher.pt \
        --grid_est_checkpoint grid_estimator_best.pt \
        --num_steps 300 \
        --save_dir vis_output
"""

import argparse
import sys
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--torch_device", type=str, default="cuda:0")
parser.add_argument("--teacher_checkpoint", type=str, required=True)
parser.add_argument("--grid_est_checkpoint", type=str, required=True)
parser.add_argument("--num_steps", type=int, default=300, help="Steps to visualize")
parser.add_argument("--save_dir", type=str, default="vis_output", help="Directory to save images")
parser.add_argument("--save_interval", type=int, default=10, help="Save image every N steps")
parser.add_argument("--show", action="store_true", help="Show realtime plot (may be slow)")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
import matplotlib
if not args_cli.show:
    matplotlib.use('Agg')  # 无显示模式，更快
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import gymnasium as gym
import isaaclab_tasks  # noqa: F401

from isaaclab_tasks.manager_based.gridEstimator.grid_estimator import GridEstimator


# ============================================================
# 从 train_grid_estimator.py 复制必要的类和函数
# ============================================================
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class Config:
    other_obs_size: int = 20
    grid_side: int = 20
    grid_embed_dim: int = 32
    other_embed_dim: int = 64
    num_actions: int = 2
    gru_hidden_dim: int = 256
    device: str = "cuda:0"


class TeacherActor(nn.Module):
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
        
        with torch.no_grad():
            dummy = torch.zeros(1, 1, grid_side, grid_side)
            cnn_out_dim = self.grid_encoder(dummy).shape[-1]
        
        self.grid_proj = nn.Sequential(
            nn.Linear(cnn_out_dim, grid_embed_dim),
            nn.ReLU(),
        )

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

    def act(self, other_obs, grid_flat, deterministic=False):
        other_feat = self.other_encoder(other_obs)
        grid_nchw = grid_flat.view(-1, 1, self.grid_side, self.grid_side)
        grid_feat = self.grid_proj(self.grid_encoder(grid_nchw))
        fused = torch.cat([other_feat, grid_feat], dim=-1)
        action_mean = self.head(fused)
        if deterministic:
            return action_mean
        std = torch.exp(self.log_std_parameter)
        action = action_mean + std * torch.randn_like(action_mean)
        return action.clamp(-1.0, 1.0)

    def act_with_grid_latent(self, other_obs, grid_latent, deterministic=False):
        other_feat = self.other_encoder(other_obs)
        fused = torch.cat([other_feat, grid_latent], dim=-1)
        action_mean = self.head(fused)
        if deterministic:
            return action_mean
        std = torch.exp(self.log_std_parameter)
        action = action_mean + std * torch.randn_like(action_mean)
        return action.clamp(-1.0, 1.0)


def load_teacher(checkpoint_path, cfg, device):
    teacher = TeacherActor(
        other_obs_size=cfg.other_obs_size,
        grid_side=cfg.grid_side,
        grid_embed_dim=cfg.grid_embed_dim,
        other_embed_dim=cfg.other_embed_dim,
        num_actions=cfg.num_actions,
        device=device,
    )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("policy", ckpt.get("model", ckpt))
    teacher.load_state_dict(state_dict, strict=False)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


def make_env(task_name):
    spec = gym.spec(task_name)
    kw = dict(spec.kwargs) if spec.kwargs is not None else {}
    env_cfg_ep = kw.get("env_cfg_entry_point")
    if isinstance(env_cfg_ep, str):
        mod_path, cls_name = env_cfg_ep.rsplit(":", 1)
        import importlib
        mod = importlib.import_module(mod_path)
        env_cfg = getattr(mod, cls_name)()
    else:
        env_cfg = env_cfg_ep()
    return gym.make(task_name, cfg=env_cfg)


def extract_obs(obs, device, other_obs_size, grid_obs_size):
    if isinstance(obs, dict):
        pol = obs.get("policy", list(obs.values())[0])
    else:
        pol = obs
    pol = torch.as_tensor(pol, device=device, dtype=torch.float32)
    if pol.dim() == 1:
        pol = pol.unsqueeze(0)
    other_obs = pol[:, :other_obs_size]
    grid = pol[:, other_obs_size:other_obs_size + grid_obs_size]
    grid = (grid > 0.5).float()
    return other_obs, grid


def format_action(env, action):
    space = env.action_space
    a = action.detach()
    if hasattr(space, "spaces") and isinstance(getattr(space, "spaces"), dict):
        key = next(iter(space.spaces.keys()))
        return {key: a.reshape(space.spaces[key].shape)}
    return a.reshape(space.shape)


# ============================================================
# 可视化函数
# ============================================================
def create_visualization(
    grid_gt: torch.Tensor,
    grid_pred: torch.Tensor,
    grid_logits: torch.Tensor,
    grid_side: int,
    step: int,
    action: torch.Tensor,
    reward: float,
    cumulative_reward: float,
):
    """创建可视化图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Reshape
    gt = grid_gt.view(grid_side, grid_side).cpu().numpy()
    pred = grid_pred.view(grid_side, grid_side).cpu().numpy()
    prob = torch.sigmoid(grid_logits).view(grid_side, grid_side).cpu().numpy()
    
    # 统计
    tp = int(((gt == 1) & (pred == 1)).sum())
    fn = int(((gt == 1) & (pred == 0)).sum())
    fp = int(((gt == 0) & (pred == 1)).sum())
    tn = int(((gt == 0) & (pred == 0)).sum())
    
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # 1. Ground Truth
    ax = axes[0, 0]
    im = ax.imshow(gt, cmap='binary', vmin=0, vmax=1, origin='lower')
    ax.set_title(f'Ground Truth\nObstacles: {int(gt.sum())}', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 2. Prediction (Binary)
    ax = axes[0, 1]
    im = ax.imshow(pred, cmap='binary', vmin=0, vmax=1, origin='lower')
    ax.set_title(f'GRU Prediction (Binary)\nObstacles: {int(pred.sum())}', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    # 3. Probability Map
    ax = axes[1, 0]
    im = ax.imshow(prob, cmap='hot', vmin=0, vmax=1, origin='lower')
    ax.set_title('GRU Prediction (Probability)', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    plt.colorbar(im, ax=ax, fraction=0.046, label='P(obstacle)')
    
    # 4. Difference Map
    ax = axes[1, 1]
    diff = np.zeros_like(gt, dtype=np.int32)
    diff[(gt == 0) & (pred == 0)] = 0  # TN - 白色
    diff[(gt == 1) & (pred == 1)] = 1  # TP - 绿色
    diff[(gt == 1) & (pred == 0)] = 2  # FN - 红色 (漏检)
    diff[(gt == 0) & (pred == 1)] = 3  # FP - 蓝色 (误检)
    
    colors = ['white', 'green', 'red', 'blue']
    cmap = ListedColormap(colors)
    ax.imshow(diff, cmap=cmap, vmin=0, vmax=3, origin='lower')
    ax.set_title('Difference (Green=TP, Red=FN, Blue=FP)', fontsize=12)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3, color='gray', linestyle='--')
    
    # 图例
    patches = [
        mpatches.Patch(color='white', label=f'TN: {tn}', edgecolor='black'),
        mpatches.Patch(color='green', label=f'TP: {tp}'),
        mpatches.Patch(color='red', label=f'FN (Missed): {fn}'),
        mpatches.Patch(color='blue', label=f'FP (False): {fp}'),
    ]
    ax.legend(handles=patches, loc='upper right', fontsize=9)
    
    # 总标题
    act_str = f"[{action[0, 0].item():.2f}, {action[0, 1].item():.2f}]"
    fig.suptitle(
        f'Step {step} | Action: {act_str} | Reward: {reward:.2f} | Cumulative: {cumulative_reward:.2f}\n'
        f'Recall: {recall:.1%} | Precision: {precision:.1%} | Accuracy: {accuracy:.1%}',
        fontsize=14, fontweight='bold'
    )
    
    plt.tight_layout()
    return fig, {"recall": recall, "precision": precision, "accuracy": accuracy}


# ============================================================
# 主函数
# ============================================================
@torch.no_grad()
def main():
    device = torch.device(args_cli.torch_device)
    cfg = Config(device=args_cli.torch_device)
    grid_obs_size = cfg.grid_side * cfg.grid_side
    
    # 创建保存目录
    os.makedirs(args_cli.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("GridEstimator Visualization")
    print("=" * 60)
    print(f"Task: {args_cli.task}")
    print(f"Teacher: {args_cli.teacher_checkpoint}")
    print(f"GridEstimator: {args_cli.grid_est_checkpoint}")
    print(f"Save to: {args_cli.save_dir}")
    print("=" * 60)
    
    # 创建环境
    print("\n[1] Creating environment...")
    env = make_env(args_cli.task)
    
    # 加载 Teacher
    print("[2] Loading Teacher...")
    teacher = load_teacher(args_cli.teacher_checkpoint, cfg, device)
    
    # 加载 GridEstimator
    print("[3] Loading GridEstimator...")
    grid_est = GridEstimator(
        obs_dim=cfg.other_obs_size,
        action_dim=cfg.num_actions,
        grid_obs_size=grid_obs_size,
        latent_dim=cfg.grid_embed_dim,
        hidden_dim=cfg.gru_hidden_dim,
    ).to(device)
    
    ckpt = torch.load(args_cli.grid_est_checkpoint, map_location=device, weights_only=False)
    if "grid_est_state_dict" in ckpt:
        grid_est.load_state_dict(ckpt["grid_est_state_dict"])
    elif isinstance(ckpt, dict) and "latent_head.0.weight" in ckpt:
        grid_est.load_state_dict(ckpt)
    else:
        grid_est.load_state_dict(ckpt)
    grid_est.eval()
    print(f"   Parameters: {sum(p.numel() for p in grid_est.parameters())}")
    
    # 运行可视化
    print("\n[4] Running visualization...")
    
    obs, _ = env.reset()
    other_obs, grid_gt = extract_obs(obs, device, cfg.other_obs_size, grid_obs_size)
    
    hidden = grid_est.init_hidden(batch_size=1, device=device)
    prev_action = torch.zeros(1, cfg.num_actions, device=device)
    
    cumulative_reward = 0.0
    all_recalls = []
    all_precisions = []
    
    if args_cli.show:
        plt.ion()
    
    for step in range(args_cli.num_steps):
        # GRU 推断
        obs_seq = other_obs.unsqueeze(1)
        act_seq = prev_action.unsqueeze(1)
        grid_latent, grid_logits, hidden = grid_est(obs_seq, act_seq, hidden)
        grid_latent = grid_latent[:, -1, :]
        grid_logits_step = grid_logits[:, -1, :]
        
        # 动作
        action = teacher.act_with_grid_latent(other_obs, grid_latent, deterministic=True)
        
        # 预测
        grid_pred = (torch.sigmoid(grid_logits_step) > 0.5).float()
        
        # Step
        obs2, rew, terminated, truncated, info = env.step(format_action(env, action))
        reward = float(rew.item() if hasattr(rew, 'item') else float(rew))
        cumulative_reward += reward
        
        # 可视化
        if step % args_cli.save_interval == 0:
            fig, stats = create_visualization(
                grid_gt, grid_pred, grid_logits_step, cfg.grid_side,
                step, action, reward, cumulative_reward
            )
            
            all_recalls.append(stats["recall"])
            all_precisions.append(stats["precision"])
            
            # 保存
            save_path = os.path.join(args_cli.save_dir, f"step_{step:04d}.png")
            fig.savefig(save_path, dpi=120, bbox_inches='tight')
            
            if args_cli.show:
                plt.draw()
                plt.pause(0.1)
            else:
                plt.close(fig)
            
            print(f"  Step {step:4d} | Reward: {reward:7.2f} | Cumulative: {cumulative_reward:8.2f} | "
                  f"Recall: {stats['recall']:.1%} | Precision: {stats['precision']:.1%}")
        
        # 更新
        prev_action = action.detach()
        other_obs, grid_gt = extract_obs(obs2, device, cfg.other_obs_size, grid_obs_size)
        
        done = terminated or truncated
        if isinstance(done, (torch.Tensor, np.ndarray)):
            done = bool(done.any() if hasattr(done, 'any') else done)
        
        if done:
            print(f"\n  Episode done at step {step}! Resetting...")
            obs, _ = env.reset()
            other_obs, grid_gt = extract_obs(obs, device, cfg.other_obs_size, grid_obs_size)
            hidden = grid_est.init_hidden(batch_size=1, device=device)
            prev_action = torch.zeros(1, cfg.num_actions, device=device)
    
    if args_cli.show:
        plt.ioff()
    
    # 总结
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total steps:      {args_cli.num_steps}")
    print(f"Cumulative reward: {cumulative_reward:.2f}")
    print(f"Avg Recall:       {np.mean(all_recalls):.1%} ± {np.std(all_recalls):.1%}")
    print(f"Avg Precision:    {np.mean(all_precisions):.1%} ± {np.std(all_precisions):.1%}")
    print(f"Images saved to:  {args_cli.save_dir}/")
    print("=" * 60)
    
    # 关闭
    try:
        env.close()
    except:
        pass


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()