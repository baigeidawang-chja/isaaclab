from __future__ import annotations

import torch
import torch.nn as nn


class GridEstimator(nn.Module):
    """
    从 (policy_obs, action) 序列推断 grid 的隐式表达。
    
    改进：添加碰撞检测特征（obs_diff + collision_indicator）
    """

    def __init__(
        self,
        obs_dim: int,           # policy obs 维度 (other_obs_size)，比如 20
        action_dim: int,        # 动作维度，比如 2
        grid_obs_size: int,     # grid 展平后维度，比如 400 (20x20)
        latent_dim: int = 32,   # 隐式表达维度
        hidden_dim: int = 256,  # GRU hidden 维度
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.grid_obs_size = grid_obs_size
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # 额外特征：obs_diff (obs_dim) + collision_features (3)
        # collision_features: [lin_speed, wheel_speed, collision_indicator]
        extra_feat_dim = obs_dim + 3
        
        # GRU 输入: obs + action + extra_features
        self.gru = nn.GRU(
            input_size=obs_dim + action_dim + extra_feat_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # 从 GRU hidden 输出 latent
        self.latent_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # 从 latent 预测 grid
        self.grid_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, grid_obs_size),
        )

    def compute_extra_features(self, obs_seq: torch.Tensor):
        """
        计算额外特征：obs_diff + collision_features
        
        obs_seq: (B, T, obs_dim=20)
            [0:3]   base_lin_vel
            [3:6]   base_ang_vel
            [6:9]   projected_gravity
            [9:13]  joint_vel (4 wheels)
            [13:15] actions
            [15:17] base_forward_dir
            [17:20] get_target_heading
        
        返回: (B, T, obs_dim + 3)
        """
        B, T, _ = obs_seq.shape
        
        # 1. obs_diff: 观测变化量
        obs_diff = torch.zeros_like(obs_seq)
        if T > 1:
            obs_diff[:, 1:, :] = obs_seq[:, 1:, :] - obs_seq[:, :-1, :]
        
        # 2. 碰撞特征
        base_lin_vel = obs_seq[:, :, 0:3]   # (B, T, 3)
        joint_vel = obs_seq[:, :, 9:13]     # (B, T, 4)
        
        # 线速度大小 (x-y 平面)
        lin_speed = torch.norm(base_lin_vel[:, :, :2], dim=-1, keepdim=True)  # (B, T, 1)
        
        # 轮速均值 (绝对值)
        wheel_speed = torch.abs(joint_vel).mean(dim=-1, keepdim=True)  # (B, T, 1)
        
        # 碰撞指示器: 轮子转但车不动 (软指示器，便于梯度传播)
        collision_indicator = torch.sigmoid(
            (wheel_speed - 1.0) * 3.0   # 轮速 > 1
        ) * torch.sigmoid(
            (0.3 - lin_speed) * 10.0    # 线速度 < 0.3
        )  # (B, T, 1)
        
        # 拼接
        collision_features = torch.cat([lin_speed, wheel_speed, collision_indicator], dim=-1)  # (B, T, 3)
        extra_features = torch.cat([obs_diff, collision_features], dim=-1)  # (B, T, obs_dim + 3)
        
        return extra_features

    def forward(
        self,
        obs_seq: torch.Tensor,      # (B, T, obs_dim)
        action_seq: torch.Tensor,   # (B, T, action_dim)
        hidden: torch.Tensor = None,
    ):
        """序列 forward（训练用）"""
        B, T, _ = obs_seq.shape
        device = obs_seq.device

        # 计算额外特征
        extra_features = self.compute_extra_features(obs_seq)  # (B, T, obs_dim + 3)

        # 拼接所有输入
        x = torch.cat([obs_seq, action_seq, extra_features], dim=-1)

        # GRU forward
        if hidden is None:
            hidden = torch.zeros(1, B, self.hidden_dim, device=device)
        gru_out, hidden = self.gru(x, hidden)

        # 输出
        latent = self.latent_head(gru_out)
        grid_hat = self.grid_decoder(latent)

        return latent, grid_hat, hidden

    def step(
        self,
        obs: torch.Tensor,          # (B, obs_dim)
        action: torch.Tensor,       # (B, action_dim)
        hidden: torch.Tensor,       # (1, B, hidden_dim)
        prev_obs: torch.Tensor = None,  # (B, obs_dim) 上一步的 obs，用于计算 diff
    ):
        """单步推理（部署时用）"""
        B = obs.shape[0]
        device = obs.device
        
        # 计算 obs_diff
        if prev_obs is None:
            obs_diff = torch.zeros_like(obs)
        else:
            obs_diff = obs - prev_obs
        
        # 计算碰撞特征
        base_lin_vel = obs[:, 0:3]
        joint_vel = obs[:, 9:13]
        
        lin_speed = torch.norm(base_lin_vel[:, :2], dim=-1, keepdim=True)
        wheel_speed = torch.abs(joint_vel).mean(dim=-1, keepdim=True)
        collision_indicator = torch.sigmoid(
            (wheel_speed - 1.0) * 3.0
        ) * torch.sigmoid(
            (0.3 - lin_speed) * 10.0
        )
        
        collision_features = torch.cat([lin_speed, wheel_speed, collision_indicator], dim=-1)
        extra_features = torch.cat([obs_diff, collision_features], dim=-1)
        
        # 拼接并 forward
        x = torch.cat([obs, action, extra_features], dim=-1).unsqueeze(1)
        gru_out, hidden = self.gru(x, hidden)
        latent = self.latent_head(gru_out.squeeze(1))
        grid_hat = self.grid_decoder(latent)
        
        return latent, grid_hat, hidden

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)