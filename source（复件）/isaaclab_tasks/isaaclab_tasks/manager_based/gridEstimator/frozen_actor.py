from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple


class FrozenActorHead(nn.Module):
    """
    阶段 1 Actor 的 Head 部分（冻结使用）。
    
    输入: fused = concat(other_feat, grid_latent)  # (B, other_embed_dim + latent_dim) = (B, 96)
    输出: action_mean, log_std
    """

    def __init__(
        self,
        other_embed_dim: int = 64,
        grid_embed_dim: int = 32,
        num_actions: int = 2,
        head_hidden: Tuple[int, int] = (64, 32),
        device: torch.device = None,
    ):
        super().__init__()
        self.other_embed_dim = other_embed_dim
        self.grid_embed_dim = grid_embed_dim
        self.num_actions = num_actions

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

    def forward(self, fused: torch.Tensor):
        """
        fused: (B, other_embed_dim + grid_embed_dim)
        返回: action_mean (B, num_actions), log_std (num_actions,)
        """
        action_mean = self.head(fused)
        return action_mean, self.log_std_parameter


class FrozenOtherEncoder(nn.Module):
    """
    阶段 1 Actor 的 other_encoder 部分（冻结使用）。
    
    输入: other_obs (B, other_obs_size)
    输出: other_feat (B, other_embed_dim)
    """

    def __init__(
        self,
        other_obs_size: int = 20,
        other_embed_dim: int = 64,
        device: torch.device = None,
    ):
        super().__init__()
        self.other_obs_size = other_obs_size
        self.other_embed_dim = other_embed_dim

        self.encoder = nn.Sequential(
            nn.Linear(other_obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, other_embed_dim),
            nn.ReLU(),
        )

        if device is not None:
            self.to(device)

    def forward(self, other_obs: torch.Tensor):
        return self.encoder(other_obs)


def load_frozen_actor_from_skrl_checkpoint(
    checkpoint_path: str,
    other_obs_size: int = 20,
    other_embed_dim: int = 64,
    grid_embed_dim: int = 32,
    num_actions: int = 2,
    head_hidden: Tuple[int, int] = (64, 32),
    device: torch.device = None,
):
    """
    从 skrl 保存的 checkpoint 加载 other_encoder 和 head 权重。
    
    返回:
    - other_encoder: FrozenOtherEncoder（冻结）
    - actor_head: FrozenActorHead（冻结）
    """
    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)

    # skrl agent checkpoint 通常结构是 {"policy": state_dict, ...} 或直接是 state_dict
    if "policy" in ckpt:
        state_dict = ckpt["policy"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        # 假设整个 ckpt 就是 state_dict
        state_dict = ckpt

    # 打印 keys 帮助 debug
    print(f"[load_frozen_actor] checkpoint keys: {list(state_dict.keys())[:20]}...")

    # 创建模块
    other_encoder = FrozenOtherEncoder(
        other_obs_size=other_obs_size,
        other_embed_dim=other_embed_dim,
        device=device,
    )
    actor_head = FrozenActorHead(
        other_embed_dim=other_embed_dim,
        grid_embed_dim=grid_embed_dim,
        num_actions=num_actions,
        head_hidden=head_hidden,
        device=device,
    )

    # 映射权重（需要根据 skrl 保存的 key 名称调整）
    # skrl 的 CNN model 通常 key 格式是 "other_encoder.0.weight" 等
    other_encoder_keys = {
        "encoder.0.weight": "other_encoder.0.weight",
        "encoder.0.bias": "other_encoder.0.bias",
        "encoder.2.weight": "other_encoder.2.weight",
        "encoder.2.bias": "other_encoder.2.bias",
    }
    head_keys = {
        "head.0.weight": "head.0.weight",
        "head.0.bias": "head.0.bias",
        "head.2.weight": "head.2.weight",
        "head.2.bias": "head.2.bias",
        "head.4.weight": "head.4.weight",
        "head.4.bias": "head.4.bias",
        "log_std_parameter": "log_std_parameter",
    }

    # 加载 other_encoder
    other_state = {}
    for new_key, old_key in other_encoder_keys.items():
        if old_key in state_dict:
            other_state[new_key] = state_dict[old_key]
        else:
            print(f"[WARNING] missing key for other_encoder: {old_key}")
    other_encoder.load_state_dict(other_state, strict=False)

    # 加载 head
    head_state = {}
    for new_key, old_key in head_keys.items():
        if old_key in state_dict:
            head_state[new_key] = state_dict[old_key]
        else:
            print(f"[WARNING] missing key for actor_head: {old_key}")
    actor_head.load_state_dict(head_state, strict=False)

    # 冻结
    for p in other_encoder.parameters():
        p.requires_grad = False
    for p in actor_head.parameters():
        p.requires_grad = False

    other_encoder.eval()
    actor_head.eval()

    return other_encoder, actor_head