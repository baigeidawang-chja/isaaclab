import torch
import math
import isaaclab.utils.math as math_utils

from isaaclab.managers import SceneEntityCfg
from isaaclab.assets import Articulation

def check_excessive_tilt(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), max_angle: float = 0.5) -> torch.Tensor:
    """检查机器人是否倾斜过度。
    
    Args:
        env: 环境对象
        asset_cfg: 机器人资产配置，默认为 "robot"
        max_angle: 最大允许角度（弧度），默认0.5（约28.6度）
    
    Returns:
        布尔张量，如果roll或pitch超过阈值则返回True
    """
    asset: Articulation = env.scene[asset_cfg.name]
    quat_w = asset.data.root_quat_w
    roll, pitch, _ = math_utils.euler_xyz_from_quat(quat_w)
    
    roll_abs = torch.abs(roll)
    pitch_abs = torch.abs(pitch)
    
    # 如果roll或pitch超过阈值，则终止
    terminated = (roll_abs > max_angle) | (pitch_abs > max_angle)
    
    return terminated
