"""
阶段二训练：将Teacher（有栅格）蒸馏到Student（仅本体感知）。

用法:
    # 第一步：收集Teacher rollout数据
    python train_stage2.py collect \
        --teacher_ckpt /path/to/agent_XXX.pt \
        --num_episodes 200 \
        --output_dir ./teacher_rollouts

    # 第二步：训练Student
    python train_stage2.py train \
        --teacher_ckpt /path/to/agent_XXX.pt \
        --dataset_dir ./teacher_rollouts \
        --output_dir ./student_output

skrl checkpoint 格式:
    checkpoint
        "policy": state_dict,       # actor网络权重
        ...
    }
"""

from __future__ import annotations

import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

# ============================================================================
# 观测结构常量 —— 根据 car_env_cfg.py 中 ObservationsCfg 的定义
# 修改这里以匹配你的实际obs结构
# ============================================================================
# base_lin_vel(3) + base_ang_vel(3) + projected_gravity(3) + joint_vel(4)
# + last_action(2) + base_forward_dir(2) + target_heading(2) = 19
# 注意：如果other_encoder第一层输入为20，说明proprio实际是20维
PROPRIO_DIM = 20
GRID_SIZE = 20          # num_cells
GRID_CELLS = GRID_SIZE * GRID_SIZE  # 400
GRID_START = PROPRIO_DIM            # 20
GRID_END = GRID_START + GRID_CELLS  # 420
ACTION_DIM = 2                      # (v, w)


# ============================================================================
# skrl Teacher Actor 重建
# ============================================================================

class TeacherActorNetwork(nn.Module):
    """与skrl训练时一致的Teacher actor结构。"""

    def __init__(self, grid_encoder, grid_proj, other_encoder, head, grid_encoder_is_conv=False, grid_size=GRID_SIZE):
        super().__init__()
        self.grid_encoder = grid_encoder
        self.grid_proj = grid_proj
        self.other_encoder = other_encoder
        self.head = head
        self.grid_encoder_is_conv = grid_encoder_is_conv
        self.grid_size = grid_size

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        grid_input = obs[:, GRID_START:GRID_END]       # (B, 400)
        other_input = obs[:, :GRID_START]               # (B, 20)

        if self.grid_encoder_is_conv:
            # reshape成 (B, 1, H, W) 给Conv2d
            B = grid_input.shape[0]
            grid_2d = grid_input.view(B, 1, self.grid_size, self.grid_size)
            grid_feat = self.grid_encoder(grid_2d)  # (B, C, H', W')
            grid_feat = grid_feat.view(B, -1)       # flatten
        else:
            grid_feat = self.grid_encoder(grid_input)

        grid_feat = self.grid_proj(grid_feat)
        other_feat = self.other_encoder(other_input)

        combined = torch.cat([other_feat, grid_feat], dim=-1)
        return self.head(combined)


def build_teacher_actor_from_skrl(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    hidden_sizes: Tuple[int, ...] = (256, 128, 64),
    device: str = "cuda:0",
) -> nn.Module:
    """从skrl checkpoint重建Teacher actor网络。

    根据checkpoint的key自动推断网络结构并加载权重。
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "policy" in ckpt:
        policy_sd = ckpt["policy"]
    elif "model" in ckpt:
        policy_sd = ckpt["model"]
    else:
        policy_sd = ckpt

    # 检查是否是自定义多分支结构
    has_grid_encoder = any(k.startswith("grid_encoder.") for k in policy_sd)

    if has_grid_encoder:
        # ---- 自定义多分支结构 ----
        def get_linear_layers(sd, prefix):
            """提取指定前缀下所有层的(index, in_dim, out_dim)"""
            layers_info = []
            for k, v in sd.items():
                if k.startswith(prefix) and k.endswith(".weight") and v.dim() == 2:
                    idx = int(k[len(prefix):].split(".")[0])
                    out_d, in_d = v.shape
                    layers_info.append((idx, in_d, out_d))
            return sorted(layers_info)

        def build_module(sd, prefix, expected_flat_dim=None):
            """根据checkpoint构建Sequential，支持Linear/Conv2d
            
            Args:
                expected_flat_dim: 如果提供，用于推断Conv2d的stride/padding
            """
            from collections import OrderedDict
            weight_layers = {}
            for k, v in sd.items():
                if k.startswith(prefix) and k.endswith(".weight"):
                    idx = int(k[len(prefix):].split(".")[0])
                    weight_layers[idx] = v

            if not weight_layers:
                return nn.Sequential(), False

            is_conv = False
            modules = OrderedDict()
            sorted_indices = sorted(weight_layers.keys())

            # 先检查是否包含Conv层，如果是则尝试推断stride
            conv_layers_info = []
            for idx in sorted_indices:
                w = weight_layers[idx]
                if w.dim() == 4:
                    out_c, in_c, kh, kw = w.shape
                    conv_layers_info.append((idx, in_c, out_c, kh, kw))

            # 如果有Conv层且提供了expected_flat_dim，推断每层stride/padding
            conv_configs = []  # 每层的 (stride, padding)
            if conv_layers_info and expected_flat_dim is not None:
                last_out_c = conv_layers_info[-1][2]
                n_conv = len(conv_layers_info)
                
                # 暴力搜索每层的stride/padding组合
                from itertools import product
                stride_options = [1, 2]
                padding_options = [0, 1]
                
                found = False
                for combo in product(product(stride_options, padding_options), repeat=n_conv):
                    h = GRID_SIZE
                    valid = True
                    for ci, (_, _, _, kh, kw) in enumerate(conv_layers_info):
                        s, p = combo[ci]
                        h = (h + 2 * p - kh) // s + 1
                        if h <= 0:
                            valid = False
                            break
                    if valid and h * h * last_out_c == expected_flat_dim:
                        conv_configs = list(combo)
                        print(f"[DEBUG] {prefix} Conv推断: {['stride={},pad={}'.format(s,p) for s,p in conv_configs]}")
                        found = True
                        break
                
                if not found:
                    # 默认 stride=1, padding=0
                    conv_configs = [(1, 0)] * n_conv
                    print(f"[WARN] {prefix} 无法推断Conv stride/padding, expected_flat={expected_flat_dim}, 使用默认")
            elif conv_layers_info:
                conv_configs = [(1, 0)] * len(conv_layers_info)

            conv_idx = 0
            for i, idx in enumerate(sorted_indices):
                if i > 0:
                    prev_idx = sorted_indices[i - 1]
                    for gap in range(prev_idx + 1, idx):
                        modules[str(gap)] = nn.ELU()

                w = weight_layers[idx]
                if w.dim() == 2:
                    out_d, in_d = w.shape
                    modules[str(idx)] = nn.Linear(in_d, out_d)
                elif w.dim() == 4:
                    out_c, in_c, kh, kw = w.shape
                    s, p = conv_configs[conv_idx] if conv_idx < len(conv_configs) else (1, 0)
                    modules[str(idx)] = nn.Conv2d(in_c, out_c, (kh, kw), stride=s, padding=p)
                    is_conv = True
                    conv_idx += 1

            return nn.Sequential(modules), is_conv

        # grid_proj的输入维度就是grid_encoder flatten后的维度
        grid_proj_layers = get_linear_layers(policy_sd, "grid_proj.")
        grid_proj_in_dim = grid_proj_layers[0][1] if grid_proj_layers else None

        grid_encoder, grid_encoder_is_conv = build_module(
            policy_sd, "grid_encoder.", expected_flat_dim=grid_proj_in_dim)
        other_encoder, _ = build_module(policy_sd, "other_encoder.")
        grid_proj, _ = build_module(policy_sd, "grid_proj.")
        head, _ = build_module(policy_sd, "head.")

        print(f"[DEBUG] grid_encoder is_conv={grid_encoder_is_conv}")
        print(f"[DEBUG] grid_encoder sd keys: {list(grid_encoder.state_dict().keys())}")

        actor = TeacherActorNetwork(grid_encoder, grid_proj, other_encoder, head,
                                     grid_encoder_is_conv=grid_encoder_is_conv,
                                     grid_size=GRID_SIZE)
        actor.to(device)

        # 加载权重（去掉log_std_parameter）
        filtered_sd = {k: v for k, v in policy_sd.items() if k != "log_std_parameter"}

        model_keys = set(actor.state_dict().keys())
        ckpt_keys = set(filtered_sd.keys())

        if model_keys == ckpt_keys:
            actor.load_state_dict(filtered_sd, strict=True)
            print(f"[Teacher] 多分支结构加载成功")
        else:
            # 按子模块分别加载
            for submodule_name, submodule in [
                ("grid_encoder", grid_encoder),
                ("grid_proj", grid_proj),
                ("other_encoder", other_encoder),
                ("head", head),
            ]:
                prefix = f"{submodule_name}."
                sub_sd = {k[len(prefix):]: v for k, v in filtered_sd.items() if k.startswith(prefix)}
                if sub_sd:
                    submodule.load_state_dict(sub_sd, strict=True)
                    print(f"  [Teacher] {submodule_name} 加载成功 ({len(sub_sd)} tensors)")

            print(f"[Teacher] 多分支结构（分模块）加载成功")

    else:
        # ---- 简单Sequential MLP ----
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ELU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        actor = nn.Sequential(*layers).to(device)

        mapped_sd = {}
        for k, v in policy_sd.items():
            new_k = k
            for prefix in ["net.", "policy.", "actor.", "model.", "a_net."]:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    break
            if new_k != "log_std_parameter":
                mapped_sd[new_k] = v
        actor.load_state_dict(mapped_sd, strict=True)
        print(f"[Teacher] Sequential结构加载成功")

    actor.eval()
    for p in actor.parameters():
        p.requires_grad = False

    print(f"[Teacher] params={sum(p.numel() for p in actor.parameters())}")
    return actor


# ============================================================================
# Teacher Policy Head: 将学生的latent替换Teacher的grid输入
# ============================================================================

class TeacherPolicyHead(nn.Module):
    """将学生的latent_hat注入到Teacher中替代grid分支。"""

    def __init__(
        self,
        teacher_actor: nn.Module,
        latent_dim: int,
        grid_cells: int = GRID_CELLS,
        proprio_dim: int = PROPRIO_DIM,
        grid_start: int = GRID_START,
        grid_end: int = GRID_END,
    ):
        super().__init__()
        self.teacher_actor = teacher_actor
        self.grid_start = grid_start
        self.grid_end = grid_end
        self.proprio_dim = proprio_dim
        self.is_multi_branch = isinstance(teacher_actor, TeacherActorNetwork)

        for p in self.teacher_actor.parameters():
            p.requires_grad = False

        if self.is_multi_branch:
            # 多分支结构：latent -> 替代grid_proj的输出
            # 推断grid_proj输出维度
            grid_proj_out_dim = list(teacher_actor.grid_proj.parameters())[-1].shape[0]
            self.adapter = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ELU(),
                nn.Linear(128, grid_proj_out_dim),
            )
        else:
            # Sequential结构：latent -> 替代grid在obs中的400维
            self.adapter = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ELU(),
                nn.Linear(128, grid_cells),
                nn.Tanh(),
            )

    def forward(self, latent_hat: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        if self.is_multi_branch:
            # 直接注入到Teacher内部：跳过grid_encoder+grid_proj，用adapter输出替代
            grid_feat_hat = self.adapter(latent_hat)
            other_feat = self.teacher_actor.other_encoder(proprio)
            combined = torch.cat([other_feat, grid_feat_hat], dim=-1)
            action = self.teacher_actor.head(combined)
        else:
            grid_replacement = self.adapter(latent_hat)
            full_obs = torch.cat([proprio, grid_replacement], dim=-1)
            action = self.teacher_actor(full_obs)
        return action


# ============================================================================
# Relevance Map Generator (内联版，不依赖外部文件)
# ============================================================================

class RelevanceMapGenerator:
    """通过patch级扰动生成Teacher决策相关性地图。"""

    def __init__(
        self,
        teacher_actor: nn.Module,
        grid_size: int = GRID_SIZE,
        patch_size: int = 5,
        grid_start: int = GRID_START,
        grid_end: int = GRID_END,
        normalize_mode: str = "softmax",
        erase_value: float = 0.0,
        temperature: float = 1.0,
    ):
        self.teacher_actor = teacher_actor
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.grid_start = grid_start
        self.grid_end = grid_end
        self.normalize_mode = normalize_mode
        self.erase_value = erase_value
        self.temperature = temperature

        assert grid_size % patch_size == 0
        self.patches_per_side = grid_size // patch_size
        self.num_patches = self.patches_per_side ** 2
        self._patch_masks = self._build_patch_masks()

    def _build_patch_masks(self) -> torch.Tensor:
        masks = torch.zeros(self.num_patches, self.grid_size, self.grid_size, dtype=torch.bool)
        for p in range(self.num_patches):
            row = p // self.patches_per_side
            col = p % self.patches_per_side
            r0, r1 = row * self.patch_size, (row + 1) * self.patch_size
            c0, c1 = col * self.patch_size, (col + 1) * self.patch_size
            masks[p, r0:r1, c0:c1] = True
        return masks

    @torch.no_grad()
    def compute_relevance(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = obs.device
        B = obs.shape[0]
        P = self.num_patches
        masks = self._patch_masks.to(device)

        a_base = self.teacher_actor(obs)

        # 批量扰动
        obs_exp = obs.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1).clone()
        grid_2d = obs_exp[:, self.grid_start:self.grid_end].view(B * P, self.grid_size, self.grid_size)
        masks_exp = masks.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * P, self.grid_size, self.grid_size)
        grid_2d[masks_exp] = self.erase_value
        obs_exp[:, self.grid_start:self.grid_end] = grid_2d.view(B * P, -1)

        a_pert = self.teacher_actor(obs_exp).view(B, P, -1)
        scores = torch.norm(a_pert - a_base.unsqueeze(1), dim=-1)  # (B, P)

        # 归一化
        if self.normalize_mode == "softmax":
            scores_norm = F.softmax(scores / self.temperature, dim=-1)
        else:
            s_min = scores.min(dim=-1, keepdim=True).values
            s_max = scores.max(dim=-1, keepdim=True).values
            scores_norm = (scores - s_min) / (s_max - s_min).clamp(min=1e-8)

        # 广播到cell
        rel_2d = torch.zeros(B, self.grid_size, self.grid_size, device=device)
        for p in range(P):
            m = masks[p]
            rel_2d[:, m] = scores_norm[:, p:p+1].expand(-1, m.sum().item())

        return rel_2d.view(B, -1), scores_norm

    @torch.no_grad()
    def compute_relevance_batched(self, obs: torch.Tensor, max_batch: int = 512):
        """省内存版本"""
        device = obs.device
        B = obs.shape[0]
        P = self.num_patches
        masks = self._patch_masks.to(device)

        a_base = self.teacher_actor(obs)
        scores = torch.zeros(B, P, device=device)

        chunk = max(1, max_batch // B)
        for p0 in range(0, P, chunk):
            p1 = min(p0 + chunk, P)
            n = p1 - p0
            obs_c = obs.unsqueeze(1).expand(B, n, -1).reshape(B * n, -1).clone()
            g = obs_c[:, self.grid_start:self.grid_end].view(B * n, self.grid_size, self.grid_size)
            m = masks[p0:p1].unsqueeze(0).expand(B, -1, -1, -1).reshape(B * n, self.grid_size, self.grid_size)
            g[m] = self.erase_value
            obs_c[:, self.grid_start:self.grid_end] = g.view(B * n, -1)
            a_p = self.teacher_actor(obs_c).view(B, n, -1)
            scores[:, p0:p1] = torch.norm(a_p - a_base.unsqueeze(1).expand(B, n, -1), dim=-1)

        if self.normalize_mode == "softmax":
            scores_norm = F.softmax(scores / self.temperature, dim=-1)
        else:
            s_min = scores.min(-1, keepdim=True).values
            s_max = scores.max(-1, keepdim=True).values
            scores_norm = (scores - s_min) / (s_max - s_min).clamp(1e-8)

        rel_2d = torch.zeros(B, self.grid_size, self.grid_size, device=device)
        for p in range(P):
            m_p = masks[p]
            rel_2d[:, m_p] = scores_norm[:, p:p+1].expand(-1, m_p.sum().item())

        return rel_2d.view(B, -1), scores_norm


# ============================================================================
# Student Grid Estimator
# ============================================================================

class GridEstimatorV2(nn.Module):
    """学生估计器：proprio + prev_action -> latent_hat + relevance_hat"""

    def __init__(
        self,
        proprio_dim: int = PROPRIO_DIM,
        action_dim: int = ACTION_DIM,
        latent_dim: int = 32,
        grid_cells: int = GRID_CELLS,
        hidden_dim: int = 128,
        encoder_hidden: Tuple[int, ...] = (128, 128),
        num_gru_layers: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.grid_cells = grid_cells
        self.hidden_dim = hidden_dim
        self.num_gru_layers = num_gru_layers

        input_dim = proprio_dim + action_dim
        layers = []
        in_d = input_dim
        for h in encoder_hidden:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ELU())
            in_d = h
        self.input_encoder = nn.Sequential(*layers)

        self.gru = nn.GRU(in_d, hidden_dim, num_gru_layers, batch_first=True)

        self.head_latent = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.head_relevance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, grid_cells),
        )

    def forward(self, proprio, prev_action, hidden=None):
        x = torch.cat([proprio, prev_action], dim=-1)
        x = self.input_encoder(x).unsqueeze(1)
        if hidden is None:
            hidden = torch.zeros(self.num_gru_layers, x.shape[0], self.hidden_dim,
                                 device=x.device, dtype=x.dtype)
        out, hidden_new = self.gru(x, hidden)
        h = out.squeeze(1)
        return self.head_latent(h), self.head_relevance(h), hidden_new

    def init_hidden(self, B, device):
        return torch.zeros(self.num_gru_layers, B, self.hidden_dim, device=device)


# ============================================================================
# Three-part Loss
# ============================================================================

class StudentDistillationLoss(nn.Module):
    def __init__(self, w_behavior=1.0, w_relevance=0.5, w_weighted=0.3,
                 relevance_loss_type="mse", weight_mode="peak"):
        super().__init__()
        self.w1 = w_behavior
        self.w2 = w_relevance
        self.w3 = w_weighted
        self.rel_type = relevance_loss_type
        self.wt_mode = weight_mode

    def forward(self, latent_hat, relevance_hat, teacher_action,
                teacher_relevance, teacher_head, proprio):
        # L1: 行为等价
        pred_action = teacher_head(latent_hat, proprio)
        L1 = F.mse_loss(pred_action, teacher_action)

        # L2: relevance对齐
        rel_prob = torch.sigmoid(relevance_hat)
        if self.rel_type == "mse":
            L2 = F.mse_loss(rel_prob, teacher_relevance)
        elif self.rel_type == "bce":
            L2 = F.binary_cross_entropy_with_logits(relevance_hat, teacher_relevance)
        else:  # kl
            eps = 1e-8
            t = (teacher_relevance.clamp(eps, 1-eps))
            s = (rel_prob.clamp(eps, 1-eps))
            t = t / t.sum(-1, keepdim=True)
            s = s / s.sum(-1, keepdim=True)
            L2 = F.kl_div(s.log(), t, reduction="batchmean")

        # L3: relevance加权
        with torch.no_grad():
            if self.wt_mode == "peak":
                w = teacher_relevance.max(-1).values
            else:
                w = teacher_relevance.sum(-1)
                w = w / w.max().clamp(min=1e-8)
            w = w.clamp(min=0.1)

        per_sample = (pred_action - teacher_action).pow(2).mean(-1)
        L3 = (per_sample * w).mean()

        total = self.w1 * L1 + self.w2 * L2 + self.w3 * L3
        return {"loss": total, "L1_behavior": L1.detach(),
                "L2_relevance": L2.detach(), "L3_weighted": L3.detach()}


# ============================================================================
# 数据收集：跑Teacher环境，保存rollout
# ============================================================================

def collect_data(args):
    """在Isaac Lab环境中跑Teacher，收集数据。"""
    from isaaclab.app import AppLauncher

    import argparse as _argparse
    launcher_parser = _argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(launcher_parser)
    launcher_args = launcher_parser.parse_args(["--headless"])
    
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    import isaaclab_tasks  # noqa: register tasks
    from isaaclab.envs import ManagerBasedRLEnv

    # 导入你的环境配置
    from isaaclab_tasks.manager_based.navigation.config.Car4WD.car_env_cfg import MyCarRoughEnvCfg

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建环境
    env_cfg = MyCarRoughEnvCfg()
    env_cfg.scene.num_envs = args.num_collect_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    obs_dim = PROPRIO_DIM + GRID_CELLS
    teacher = build_teacher_actor_from_skrl(
        args.teacher_ckpt, obs_dim, ACTION_DIM,
        hidden_sizes=tuple(args.teacher_hidden),
        device=device,
    )

    # ====== 验证teacher输出 ======
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"].to(device)
    print(f"[验证] obs shape: {obs.shape}, obs总维度: {obs.shape[-1]}")
    print(f"[验证] obs[:, :GRID_START] range: [{obs[:, :GRID_START].min():.4f}, {obs[:, :GRID_START].max():.4f}]")
    print(f"[验证] obs[:, GRID_START:GRID_END] range: [{obs[:, GRID_START:GRID_END].min():.4f}, {obs[:, GRID_START:GRID_END].max():.4f}]")
    with torch.no_grad():
        test_action = teacher(obs)
    print(f"[验证] teacher action: mean={test_action.mean():.4f}, std={test_action.std():.4f}, "
          f"range=[{test_action.min():.4f}, {test_action.max():.4f}]")
    if test_action.abs().max() > 100:
        print(f"[错误] Teacher输出异常大！请检查网络结构是否正确重建。")
        print(f"[提示] 请检查你训练teacher时的网络定义代码，确认Conv2d的stride和padding参数。")
        env.close()
        return
    # ============================

    print(f"[收集] 开始收集 {args.num_episodes} 个episode...")
    episode_count = 0
    max_steps = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    while episode_count < args.num_episodes:
        obs_dict, _ = env.reset()
        obs = obs_dict["policy"].to(device)  # (num_envs, obs_dim)

        proprios_all = [[] for _ in range(env_cfg.scene.num_envs)]
        grids_all = [[] for _ in range(env_cfg.scene.num_envs)]
        actions_all = [[] for _ in range(env_cfg.scene.num_envs)]
        prev_actions_all = [[] for _ in range(env_cfg.scene.num_envs)]

        prev_action = torch.zeros(env_cfg.scene.num_envs, ACTION_DIM, device=device)

        for step in range(max_steps):
            proprio = obs[:, :GRID_START]
            grid = obs[:, GRID_START:GRID_END]

            with torch.no_grad():
                action = teacher(obs)

            for e in range(env_cfg.scene.num_envs):
                proprios_all[e].append(proprio[e].cpu())
                grids_all[e].append(grid[e].cpu())
                actions_all[e].append(action[e].cpu())
                prev_actions_all[e].append(prev_action[e].cpu())

            obs_dict, _, terminated, truncated, _ = env.step(action)
            obs = obs_dict["policy"].to(device)
            prev_action = action.clone()

            # 检查哪些env结束了
            done = terminated | truncated
            for e in range(env_cfg.scene.num_envs):
                if done[e] and len(proprios_all[e]) > 10:
                    data = {
                        "proprio": torch.stack(proprios_all[e]),
                        "grid": torch.stack(grids_all[e]),
                        "action": torch.stack(actions_all[e]),
                        "prev_action": torch.stack(prev_actions_all[e]),
                    }
                    path = os.path.join(args.output_dir, f"episode_{episode_count:04d}.pt")
                    torch.save(data, path)
                    print(f"  Episode {episode_count}: {len(proprios_all[e])} steps -> {path}")
                    episode_count += 1

                    # 重置该env的缓冲
                    proprios_all[e] = []
                    grids_all[e] = []
                    actions_all[e] = []
                    prev_actions_all[e] = []

                    if episode_count >= args.num_episodes:
                        break

            if episode_count >= args.num_episodes:
                break

    env.close()
    print(f"[收集] 完成! 共 {episode_count} 个episode保存到 {args.output_dir}")


# ============================================================================
# 数据集
# ============================================================================

class RolloutDataset(Dataset):
    def __init__(self, data_files: List[str], seq_len: int = 32):
        self.seq_len = seq_len
        self.segments = []
        for f in data_files:
            data = torch.load(f, map_location="cpu")
            T = data["proprio"].shape[0]
            for start in range(0, max(1, T - seq_len + 1), seq_len // 2):
                end = min(start + seq_len, T)
                if end - start < seq_len:
                    continue
                self.segments.append({k: data[k][start:end] for k in data})
        print(f"[数据集] {len(data_files)} 文件, {len(self.segments)} 片段")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx]


# ============================================================================
# 训练
# ============================================================================

def train(args):
    device = torch.device(args.device)

    # --- 加载Teacher ---
    obs_dim = PROPRIO_DIM + GRID_CELLS
    teacher = build_teacher_actor_from_skrl(
        args.teacher_ckpt, obs_dim, ACTION_DIM,
        hidden_sizes=tuple(args.teacher_hidden), device=device,
    )

    # --- 数据集 ---
    data_files = sorted(glob.glob(os.path.join(args.dataset_dir, "*.pt")))
    assert len(data_files) > 0, f"{args.dataset_dir} 中没有 .pt 文件"
    dataset = RolloutDataset(data_files, seq_len=args.seq_len)

    # ==================== 诊断 ====================
    print("\n" + "=" * 60)
    print("[诊断] 检查数据和Teacher输出")
    seg0 = dataset[0]
    print(f"  proprio shape: {seg0['proprio'].shape}, range: [{seg0['proprio'].min():.4f}, {seg0['proprio'].max():.4f}]")
    print(f"  grid shape: {seg0['grid'].shape}, range: [{seg0['grid'].min():.4f}, {seg0['grid'].max():.4f}]")
    print(f"  action shape: {seg0['action'].shape}, range: [{seg0['action'].min():.4f}, {seg0['action'].max():.4f}]")
    print(f"  prev_action shape: {seg0['prev_action'].shape}")
    print(f"  proprio有NaN: {seg0['proprio'].isnan().any()}")
    print(f"  grid有NaN: {seg0['grid'].isnan().any()}")
    print(f"  action有NaN: {seg0['action'].isnan().any()}")
    
    # 数据质量检查
    if seg0['proprio'].abs().max() > 1000 or seg0['action'].abs().max() > 1000:
        print("\n[错误] 数据异常！proprio或action值极大，说明收集时teacher输出有误。")
        print("[解决] 请执行以下步骤：")
        print("  1. 删除旧数据: rm -rf ./teacher_rollouts/*")
        print("  2. 删除旧缓存: rm -rf ./student_out/relevance_cache.pt")
        print("  3. 重新收集数据: python train_stage2.py collect ...")
        print("  4. 重新训练: python train_stage2.py train ...")
        return
    
    # 测试teacher前向
    test_obs = torch.cat([seg0['proprio'][:1], seg0['grid'][:1]], dim=-1).to(device)
    print(f"  test_obs shape: {test_obs.shape}, 总维度: {test_obs.shape[-1]}")
    with torch.no_grad():
        test_out = teacher(test_obs)
    print(f"  teacher输出: {test_out}, 有NaN: {test_out.isnan().any()}")
    
    # 测试teacher_head
    teacher_head_test = TeacherPolicyHead(
        teacher, latent_dim=args.latent_dim,
        grid_cells=GRID_CELLS, proprio_dim=PROPRIO_DIM,
    ).to(device)
    test_latent = torch.randn(1, args.latent_dim, device=device)
    test_proprio = seg0['proprio'][:1].to(device)
    test_action_out = teacher_head_test(test_latent, test_proprio)
    print(f"  teacher_head输出: {test_action_out}, 有NaN: {test_action_out.isnan().any()}")
    del teacher_head_test
    print("=" * 60 + "\n")
    # ==================== 诊断结束 ====================

    # --- 预计算relevance maps ---
    cache_path = os.path.join(args.output_dir, "relevance_cache.pt")
    os.makedirs(args.output_dir, exist_ok=True)

    if os.path.exists(cache_path) and not args.recompute_relevance:
        print(f"[缓存] 加载 {cache_path}")
        relevance_maps = torch.load(cache_path, map_location="cpu")
    else:
        gen = RelevanceMapGenerator(
            teacher, grid_size=GRID_SIZE, patch_size=args.patch_size,
            grid_start=GRID_START, grid_end=GRID_END,
        )
        relevance_maps = []
        print(f"[Relevance] 预计算中... ({len(dataset)} 片段)")
        for i in range(len(dataset)):
            seg = dataset[i]
            full_obs = torch.cat([seg["proprio"], seg["grid"]], dim=-1).to(device)
            rel, _ = gen.compute_relevance_batched(full_obs, max_batch=1024)
            relevance_maps.append(rel.cpu())
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(dataset)}]")
        torch.save(relevance_maps, cache_path)
        print(f"[缓存] 保存到 {cache_path}")

    # 诊断relevance
    print(f"[诊断] relevance_maps[0] shape: {relevance_maps[0].shape}")
    print(f"[诊断] relevance_maps[0] 有NaN: {relevance_maps[0].isnan().any()}")
    print(f"[诊断] relevance_maps[0] range: [{relevance_maps[0].min():.6f}, {relevance_maps[0].max():.6f}]")
    
    if relevance_maps[0].isnan().any():
        print("\n[错误] relevance_maps 含 NaN！请删除缓存后重新计算：")
        print(f"  rm {cache_path}")
        print("  然后重新运行 train 命令")
        return

    # --- 模型 ---
    student = GridEstimatorV2(
        proprio_dim=PROPRIO_DIM, action_dim=ACTION_DIM,
        latent_dim=args.latent_dim, grid_cells=GRID_CELLS,
        hidden_dim=args.hidden_dim,
        encoder_hidden=(args.hidden_dim, args.hidden_dim),
        num_gru_layers=args.num_gru_layers,
    ).to(device)

    teacher_head = TeacherPolicyHead(
        teacher, latent_dim=args.latent_dim,
        grid_cells=GRID_CELLS, proprio_dim=PROPRIO_DIM,
    ).to(device)

    criterion = StudentDistillationLoss(
        w_behavior=args.w_behavior, w_relevance=args.w_relevance,
        w_weighted=args.w_relevance_weighted,
        relevance_loss_type=args.relevance_loss_type,
    )

    # student参数 + adapter参数一起优化
    all_params = list(student.parameters()) + list(teacher_head.adapter.parameters())
    optimizer = optim.Adam(all_params, lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")

    for epoch in range(args.epochs):
        student.train()
        teacher_head.adapter.train()

        epoch_loss = {"loss": 0, "L1_behavior": 0, "L2_relevance": 0, "L3_weighted": 0}
        n_batch = 0
        indices = torch.randperm(len(dataset))

        for b_start in range(0, len(dataset), args.batch_size):
            b_idx = indices[b_start:b_start + args.batch_size]
            B = len(b_idx)

            proprios = torch.stack([dataset[i]["proprio"] for i in b_idx]).to(device)
            prev_acts = torch.stack([dataset[i]["prev_action"] for i in b_idx]).to(device)
            actions = torch.stack([dataset[i]["action"] for i in b_idx]).to(device)
            rels = torch.stack([relevance_maps[i] for i in b_idx]).to(device)

            hidden = student.init_hidden(B, device)
            batch_loss = 0.0
            batch_metrics = {k: 0.0 for k in ["L1_behavior", "L2_relevance", "L3_weighted"]}

            for t in range(args.seq_len):
                latent_hat, rel_hat, hidden = student(
                    proprios[:, t], prev_acts[:, t], hidden
                )
                hidden = hidden.detach()

                loss_dict = criterion(
                    latent_hat=latent_hat, relevance_hat=rel_hat,
                    teacher_action=actions[:, t], teacher_relevance=rels[:, t],
                    teacher_head=teacher_head, proprio=proprios[:, t],
                )

                batch_loss += loss_dict["loss"]
                for k in batch_metrics:
                    batch_metrics[k] += loss_dict[k].item()

            batch_loss = batch_loss / args.seq_len

            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
            optimizer.step()

            epoch_loss["loss"] += batch_loss.item()
            for k in batch_metrics:
                epoch_loss[k] += batch_metrics[k] / args.seq_len
            n_batch += 1

        scheduler.step()
        for k in epoch_loss:
            epoch_loss[k] /= max(n_batch, 1)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Loss: {epoch_loss['loss']:.4f} | "
            f"L1: {epoch_loss['L1_behavior']:.4f} | "
            f"L2: {epoch_loss['L2_relevance']:.4f} | "
            f"L3: {epoch_loss['L3_weighted']:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if epoch_loss["loss"] < best_loss:
            best_loss = epoch_loss["loss"]
            torch.save({
                "student": student.state_dict(),
                "adapter": teacher_head.adapter.state_dict(),
                "epoch": epoch, "loss": best_loss,
            }, os.path.join(args.output_dir, "student_best.pt"))

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "student": student.state_dict(),
                "adapter": teacher_head.adapter.state_dict(),
                "epoch": epoch,
            }, os.path.join(args.output_dir, f"student_epoch{epoch+1}.pt"))

    print(f"训练完成。最佳损失: {best_loss:.4f}")


# ============================================================================
# 检查工具：打印skrl checkpoint的结构
# ============================================================================

def inspect_checkpoint(args):
    """打印skrl checkpoint的内容，帮助确认网络结构。"""
    ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    print("=" * 60)
    print(f"Checkpoint: {args.teacher_ckpt}")
    print(f"Top-level keys: {list(ckpt.keys())}")
    print("=" * 60)

    for key in ["policy", "model", "actor", "a_net"]:
        if key in ckpt:
            sd = ckpt[key]
            if isinstance(sd, dict):
                print(f"\n[{key}] state_dict keys:")
                for k, v in sd.items():
                    print(f"  {k}: {v.shape}")
            else:
                print(f"\n[{key}] type: {type(sd)}")

    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        print(f"\n整个ckpt就是state_dict:")
        for k, v in ckpt.items():
            print(f"  {k}: {v.shape}")

    print("\n" + "=" * 60)
    print("提示: 根据以上输出确认 --teacher_hidden 参数")
    print("例如: 如果看到 0.weight: [256, 419], 2.weight: [128, 256], 4.weight: [64, 128], 6.weight: [2, 64]")
    print("那么 --teacher_hidden 256 128 64")
    print("=" * 60)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="阶段二蒸馏训练")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ---- inspect ----
    p_inspect = subparsers.add_parser("inspect", help="检查skrl checkpoint结构")
    p_inspect.add_argument("--teacher_ckpt", type=str, required=True)

    # ---- collect ----
    p_collect = subparsers.add_parser("collect", help="收集Teacher rollout数据")
    p_collect.add_argument("--teacher_ckpt", type=str, required=True)
    p_collect.add_argument("--output_dir", type=str, default="./teacher_rollouts")
    p_collect.add_argument("--num_episodes", type=int, default=200)
    p_collect.add_argument("--num_collect_envs", type=int, default=16)
    p_collect.add_argument("--device", type=str, default="cuda:0")
    p_collect.add_argument("--teacher_hidden", type=int, nargs="+", default=[256, 128, 64],
                           help="Teacher MLP隐藏层大小，必须与skrl训练时一致")

    # ---- train ----
    p_train = subparsers.add_parser("train", help="训练Student")
    p_train.add_argument("--teacher_ckpt", type=str, required=True)
    p_train.add_argument("--dataset_dir", type=str, required=True)
    p_train.add_argument("--output_dir", type=str, default="./student_output")
    p_train.add_argument("--device", type=str, default="cuda:0")
    p_train.add_argument("--teacher_hidden", type=int, nargs="+", default=[256, 128, 64])
    # 数据
    p_train.add_argument("--seq_len", type=int, default=32)
    p_train.add_argument("--batch_size", type=int, default=64)
    # Relevance
    p_train.add_argument("--patch_size", type=int, default=5)
    p_train.add_argument("--recompute_relevance", action="store_true")
    # Student架构
    p_train.add_argument("--latent_dim", type=int, default=32)
    p_train.add_argument("--hidden_dim", type=int, default=128)
    p_train.add_argument("--num_gru_layers", type=int, default=1)
    # 损失
    p_train.add_argument("--w_behavior", type=float, default=1.0)
    p_train.add_argument("--w_relevance", type=float, default=0.5)
    p_train.add_argument("--w_relevance_weighted", type=float, default=0.3)
    p_train.add_argument("--relevance_loss_type", type=str, default="mse",
                         choices=["mse", "bce", "kl"])
    # 训练
    p_train.add_argument("--epochs", type=int, default=200)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--save_every", type=int, default=20)

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_checkpoint(args)
    elif args.command == "collect":
        collect_data(args)
    elif args.command == "train":
        train(args)
    else:
        parser.print_help()
