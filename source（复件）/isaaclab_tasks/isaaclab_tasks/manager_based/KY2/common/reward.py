import torch
import math
from isaaclab.managers import SceneEntityCfg

def horizontal_stability_reward(
    env, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_roll: float = 0.1,
    max_pitch: float = 0.1,
) -> torch.Tensor:
    """奖励函数：保持机器人水平稳定（roll和pitch接近0）。
    
    Args:
        env: 环境对象
        asset_cfg: 机器人资产配置
        max_roll: 允许的最大roll角度（弧度），超过此值奖励为0
        max_pitch: 允许的最大pitch角度（弧度），超过此值奖励为0
    
    Returns:
        奖励值，范围[0, 1]，当roll和pitch都接近0时奖励接近1
    """
    from isaaclab.assets import Articulation
    import isaaclab.utils.math as math_utils
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取机器人朝向（四元数）
    quat_w = asset.data.root_quat_w
    
    # 将四元数转换为欧拉角 (roll, pitch, yaw)
    # 使用ZYX顺序（yaw-pitch-roll）
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(quat_w)
    
    # 计算roll和pitch的绝对值
    roll_abs = torch.abs(roll)
    pitch_abs = torch.abs(pitch)
    
    # 计算奖励：当roll和pitch都接近0时，奖励接近1
    # 使用指数衰减函数
    roll_reward = torch.exp(-roll_abs / max_roll)
    pitch_reward = torch.exp(-pitch_abs / max_pitch)
    
    # 组合奖励（两个角度都需要小）
    reward = roll_reward * pitch_reward
    
    return reward


def angular_velocity_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.5,
) -> torch.Tensor:
    """惩罚函数：惩罚过大的角速度（保持稳定）。
    
    Args:
        env: 环境对象
        asset_cfg: 机器人资产配置
        threshold: 角速度阈值（rad/s）
    
    Returns:
        惩罚值，角速度越大惩罚越大
    """
    from isaaclab.assets import Articulation
    
    asset: Articulation = env.scene[asset_cfg.name]
    
    # 获取角速度
    ang_vel = asset.data.root_ang_vel_w
    
    # 计算角速度的L2范数
    ang_vel_norm = torch.norm(ang_vel, dim=-1)
    
    # 计算惩罚（超过阈值时惩罚增加）
    penalty = torch.clamp(ang_vel_norm / threshold, min=0.0, max=1.0)
    
    return -penalty

def keep_position_reward(
    env,
    asset_cfg: SceneEntityCfg,
    target_pos: tuple[float, float, float] = (0.0, 0.0, 5.0),
    distance_sigma: float = 1.0,
):
    """Reward for keeping the base close to a target position in world frame.

    Uses an RBF reward: exp(-||p - p*||^2 / (2*sigma^2)).
    Range: (0, 1].
    """
    asset = env.scene[asset_cfg.name]
    pos_w = asset.data.root_pos_w  # (num_envs, 3)
    target = torch.tensor(target_pos, device=pos_w.device, dtype=pos_w.dtype).unsqueeze(0)
    d2 = torch.sum((pos_w - target) ** 2, dim=-1)
    sigma2 = (distance_sigma * distance_sigma)
    return torch.exp(-0.5 * d2 / sigma2)
