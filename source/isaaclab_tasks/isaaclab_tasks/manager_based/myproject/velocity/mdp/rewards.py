# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat, wrap_to_pi


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_rotate_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)

def track_goal_distance_exp(
env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of goal distance (XY) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    # get target position from command manager (shape: [num_envs, 3])
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    # calculate XY distance
    root_pos = asset.data.root_pos_w[:, :2]
    distance_sq = torch.sum(torch.square(root_pos - target_pos), dim=1)
    # apply exponential kernel
    return torch.exp(-distance_sq / std**2)

def position_command_world_tanh(env: ManagerBasedRLEnv, std: float, command_name: str) -> torch.Tensor:
    """Reward position tracking with tanh kernel."""
    command_generator = env.command_manager.get_term(command_name)
    target_pos_w = command_generator._pos_command_w
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[:, :3]
    distance = torch.norm(target_pos_w - robot_pos_w, dim=1)
    # print(f"distance{distance}")
    # des_pos_b = 
    # 新增调试输出（仅打印第一个环境的数据）
    # if env.num_envs > 0:
    #     env_id = 0  # 选择第一个环境
    #     des_pos_debug = des_pos_b[env_id].detach().cpu().numpy()  # 安全转换到CPU
    #     print(f"[Debug] des_pos_b (env {env_id}): x={des_pos_debug[0]:.2f}, y={des_pos_debug[1]:.2f}, z={des_pos_debug[2]:.2f}")
    return 1 - torch.exp(-distance / std)


def heading_command_abs(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Penalize tracking orientation error."""
    command = env.command_manager.get_term(command_name)
    target_heading = command._heading_command_w
    robot = env.scene["robot"]
    current_heading = robot.data.heading_w
    return torch.abs(wrap_to_pi(target_heading - current_heading))
    # heading_b = command[:, 3]
    # return heading_b.abs()

def velocity_direction_reward(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """奖励速度向量指向目标"""
    command = env.command_manager.get_term(command_name)
    target_pos_w = command._pos_command_w
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[:, :3]
    
    # 计算目标方向向量
    target_dir = target_pos_w - robot_pos_w
    target_dir_normalized = target_dir / (torch.norm(target_dir, dim=1, keepdim=True) + 1e-6)
    
    # 获取当前速度向量
    current_vel_w = robot.data.root_lin_vel_w[:, :2]  # 只考虑x-y平面
    current_vel_normalized = current_vel_w / (torch.norm(current_vel_w, dim=1, keepdim=True) + 1e-6)
    
    # 计算余弦相似度
    return torch.einsum("bi,bi->b", target_dir_normalized[:, :2], current_vel_normalized)

def get_desired_position(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """直接获取期望位置 des_pos_b."""
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]  # 提取前3维作为位置
    return des_pos_b

# 目标距离奖励（指数衰减）
def goal_distance_reward(
        env, target_pos, max_distance
) -> torch.Tensor:
    robot = env.scene["robot"] 
    # 获取机器人当前位置
    root_pos = robot.data.root_pos_w

    # print("[DEBUG] root_pos:", root_pos)  # 检查位置是否非零
    # print("[DEBUG] target_pos:", target_pos)  # 确认目标位置有效
    # 计算距离
      # 将 target_pos 转换为 Tensor 并广播到多环境
    target_pos_tensor = torch.tensor(target_pos, device=env.device, dtype=torch.float32)  # 转换为Tensor
    target_pos_tensor = target_pos_tensor.unsqueeze(0).repeat(env.num_envs, 1)  # 形状: (num_envs, 3)
    
    # 计算距离（仅XY平面）
    delta_xy = root_pos[:, :2] - target_pos_tensor[:, :2]  # 形状: (num_envs, 2)
    distance = torch.norm(delta_xy, dim=1)  # 形状: (num_envs,)
    # 归一化奖励
    reward = 1.0 - (distance / max_distance)
    # print(f"[奖励计算] 目标距离: {distance.mean().item():.2f}m | 原始奖励: {reward.mean().item():.4f} | 加权后: {reward.mean().item() * 3:.4f}")
    # 裁剪到 [0, 1] 范围
    return torch.clamp(reward, min=0.0, max=1.0)

# 障碍物惩罚（距离越近惩罚越大）
def obstacle_penalty(env, obstacle_pos, safe_radius):
    robot = env.scene["robot"] 
    # 获取机器人当前位置
    root_pos = robot.data.root_pos_w
    # 计算距离
    # 将障碍物位置转换为Tensor并广播到多环境
    obstacle_pos_tensor = torch.tensor(obstacle_pos, device=env.device, dtype=torch.float32)
    obstacle_pos_tensor = obstacle_pos_tensor.unsqueeze(0).repeat(env.num_envs, 1)  # 形状: (num_envs, 3)
    
    # 计算XY平面距离
    delta_xy = root_pos[:, :2] - obstacle_pos_tensor[:, :2]  # 形状: (num_envs, 2)
    distance = torch.norm(delta_xy, dim=1)  # 形状: (num_envs,)
    # 计算惩罚值（距离<=safe_radius时惩罚从1到0，超出时惩罚0）
    penalty = 1.0 - (distance / safe_radius)
    
    # print(f"[惩罚计算] 障碍物距离: {distance.mean().item():.2f}m | 原始惩罚: {penalty.mean().item():.4f} | 加权后: {penalty.mean().item() * -2:.4f}")
    return torch.clamp(penalty, min=0.0, max=1.0)

# def collision_penalty(env, sensor_cfg: SceneEntityCfg, threshold=0.1):
#     """基于接触力的碰撞检测（修正版）"""
#     # 获取接触传感器数据
#     contact_sensor = env.scene.sensors[sensor_cfg.name]
    
#     # 获取障碍物实体（根据场景配置中的名称）
#     obstacle = env.scene["Obstacle"]  # 注意这里要对应场景配置中的对象名称
    
#     # 关键修正：使用正确的PhysX API获取body索引
#     obstacle_body_ids = obstacle.root_physx_view.env_ids  
    
#     # 计算与障碍物的接触力（添加维度处理）
#     net_contact_forces = contact_sensor.data.net_forces_w[..., obstacle_body_ids, :]
    
#     # 计算接触力强度（添加维度处理）
#     force_norm = torch.norm(net_contact_forces, dim=-1)
    
#     # 生成碰撞标志（需要保持与reset_buf相同的维度）
#     collision_flag = (force_norm > threshold).any(dim=-1).float()  # 添加.any()处理多个刚体
    
#     return collision_flag

def goal_reached(env, target_pos: Tuple[float, float, float], success_radius: float) -> torch.Tensor:
    """检测是否到达目标点"""
    robot = env.scene["robot"] 
    # 获取机器人当前位置
    robot_pos = robot.data.root_pos_w
    # 计算与目标的距离
    distance = torch.norm(robot_pos - torch.tensor(target_pos, device=env.device), dim=1)
    # 判断是否在成功半径内
    return distance < success_radius 


