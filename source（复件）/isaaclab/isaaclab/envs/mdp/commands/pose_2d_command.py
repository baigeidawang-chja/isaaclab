# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause 
"""Sub-module containing command generators for the 2D-pose for locomotion tasks."""

from __future__ import annotations

# from omni.isaac.kit import SimulationApp
# simulation_app = SimulationApp({"headless": True}) 

import torch
import time
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers,VisualizationMarkersCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate_inverse, wrap_to_pi, yaw_quat, quat_apply_inverse, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import TerrainBasedPose2dCommandCfg, UniformPose2dCommandCfg, SequentialWaypointCommandCfg


class UniformPose2dCommand(CommandTerm):
    """Command generator that generates pose commands containing a 3-D position and heading.

    The command generator samples uniform 2D positions around the environment origin. It sets
    the height of the position command to the default root height of the robot. The heading
    command is either set to point towards the target or is sampled uniformly.
    This can be configured through the :attr:`Pose2dCommandCfg.simple_heading` parameter in
    the configuration.
    """

    cfg: UniformPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPose2dCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot and terrain assets
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        # self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

        # 新增三维位置误差计算
        pos_diff_3d = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.metrics["error_pos"] = torch.norm(pos_diff_3d, dim=1)

    def __str__(self) -> str:
        msg = "PositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 4)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)
    # @property
    # def _pos_command_w(self) -> torch.Tensor:
    #     """世界坐标系下的目标位置 (x, y, z)"""
    #     # 动态获取当前设备（自动适配GPU/CPU）
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     return torch.tensor([5.0, 6.0, 0.0], dtype=torch.float32, device=device)
    @property
    def _pos_command_w(self) -> torch.Tensor:
        return self.pos_command_w


    @property
    def _heading_command_w(self) -> torch.Tensor:
        """世界坐标系下的目标航向角"""
        return self.heading_command_w
    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        # 新增三维位置误差计算
        pos_diff_3d = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.metrics["error_pos"] = torch.norm(pos_diff_3d, dim=1)
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))

            # 调试输出（仅针对第一个环境）
        # if self.num_envs > 0:  # 确保至少存在一个环境
        #     env_id = 0  # 查看第一个环境的数据
        # print("\n==== Position Debug ====")
        # print(f"[Env {env_id}] Target Position (World): {self.pos_command_w[env_id].cpu().numpy()}")
        # print(f"[Env {env_id}] Robot Position (World): {self.robot.data.root_pos_w[env_id, :3].cpu().numpy()}")
        # print(f"[Env {env_id}] error_pos: {self.metrics['error_pos'][env_id].item():.4f}")
        # print(f"[Env {env_id}] error_pos_2d: {self.metrics['error_pos_2d'][env_id].item():.4f}")
        # print("=======================\n")

    def _resample_command(self, env_ids: Sequence[int]):
        # obtain env origins for the environments
        self.pos_command_w[env_ids] = self._env.scene.env_origins[env_ids]
        # offset the position command by the current root position
        r = torch.empty(len(env_ids), device=self.device)
        self.pos_command_w[env_ids, 0] += r.uniform_(*self.cfg.ranges.pos_x)
        self.pos_command_w[env_ids, 1] += r.uniform_(*self.cfg.ranges.pos_y)
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)

    def _update_command(self):
        """Re-target the position command to the current root state."""
        target_vec = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        self.pos_command_b[:] = quat_rotate_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec)
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
        )
    


class TerrainBasedPose2dCommand(UniformPose2dCommand):
    """Command generator that generates pose commands based on the terrain.

    This command generator samples the position commands from the valid patches of the terrain.
    The heading commands are either set to point towards the target or are sampled uniformly.

    It expects the terrain to have a valid flat patches under the key 'target'.
    """

    cfg: TerrainBasedPose2dCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: TerrainBasedPose2dCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the terrain asset
        self.terrain: TerrainImporter = env.scene["terrain"]

        # obtain the valid targets from the terrain
        if "target" not in self.terrain.flat_patches:
            raise RuntimeError(
                "The terrain-based command generator requires a valid flat patch under 'target' in the terrain."
                f" Found: {list(self.terrain.flat_patches.keys())}"
            )
        # valid targets: (terrain_level, terrain_type, num_patches, 3)
        self.valid_targets: torch.Tensor = self.terrain.flat_patches["target"]

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new position targets from the terrain
        ids = torch.randint(0, self.valid_targets.shape[2], size=(len(env_ids),), device=self.device)
        self.pos_command_w[env_ids] = self.valid_targets[
            self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids], ids
        ]
        # offset the position command by the current root height
        self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]

        if self.cfg.simple_heading:
            # set heading command to point towards target
            target_vec = self.pos_command_w[env_ids] - self.robot.data.root_pos_w[env_ids]
            target_direction = torch.atan2(target_vec[:, 1], target_vec[:, 0])
            flipped_target_direction = wrap_to_pi(target_direction + torch.pi)

            # compute errors to find the closest direction to the current heading
            # this is done to avoid the discontinuity at the -pi/pi boundary
            curr_to_target = wrap_to_pi(target_direction - self.robot.data.heading_w[env_ids]).abs()
            curr_to_flipped_target = wrap_to_pi(flipped_target_direction - self.robot.data.heading_w[env_ids]).abs()

            # set the heading command to the closest direction
            self.heading_command_w[env_ids] = torch.where(
                curr_to_target < curr_to_flipped_target,
                target_direction,
                flipped_target_direction,
            )
        else:
            # random heading command
            r = torch.empty(len(env_ids), device=self.device)
            self.heading_command_w[env_ids] = r.uniform_(*self.cfg.ranges.heading)


# class SequentialWaypointCommand(CommandTerm):
#     """
#     顺序航点命令生成器 (向量化优化版 - 终版)
    
#     这个版本显式实现了所有父类要求的抽象方法，结构最规范。
#     """
#     cfg: "SequentialWaypointCommandCfg" # 替换为您的实际配置类

#     def __init__(self, cfg: "SequentialWaypointCommandCfg", env: ManagerBasedEnv):
#         super().__init__(cfg, env)
        
#         self.robot: Articulation = env.scene[cfg.asset_name]

#         # 将航点列表转换为Tensor，便于向量化索引
#         self.waypoints_tensor = torch.tensor(cfg.waypoints, dtype=torch.float32, device=self.device)
#         self.num_waypoints = self.waypoints_tensor.shape[0]
#         self.max_waypoint_idx = self.num_waypoints - 1

#         # --- 核心状态变量 ---
#         self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
#         # 世界坐标系下的指令
#         self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
#         self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        
#         # 本体坐标系下的指令 (由 _update_command 负责计算)
#         self.pos_command_b = torch.zeros_like(self.pos_command_w)
#         self.heading_command_b = torch.zeros_like(self.heading_command_w)

#         # “刚刚到达”标志，解耦奖励计算
#         self.newly_reached_waypoint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
#         # 状态追踪（用于指标和复杂逻辑）
#         self.visited_waypoints_mask = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
#         # --- 指标 (Metrics) ---
#         self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
#         self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
#         # ... 其他您需要的指标

#         # 初始化命令
#         self._resample_command(torch.arange(self.num_envs, device=self.device))
#         self._update_command() # 首次计算本体坐标系指令

#     @property
#     def command(self) -> torch.Tensor:
#         """返回在机器人本体坐标系下的指令 [dx, dy, dz, d_yaw]"""
#         return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)


#     def reset(self, env_ids: Sequence[int] | None = None) -> dict:  # 1. 修改返回类型提示为 dict
#         if env_ids is None:
#             env_ids = slice(None)
        
#         self.current_waypoint_idx[env_ids] = 0
#         self.visited_waypoints_mask[env_ids] = 0
        
#         if isinstance(env_ids, slice):
#             self._resample_command(torch.arange(self.num_envs, device=self.device))
#         else:
#             self._resample_command(torch.tensor(env_ids, dtype=torch.long, device=self.device))
        
#         # 2. 捕获父类 reset 方法返回的指标字典
#         extras = super().reset(env_ids)
        
#         # 3. 将这个字典返回，履行与 CommandManager 的“合约”
#         return extras

#     def compute(self, dt: float):
#         """
#         重写父类 compute 方法，整合自定义更新逻辑
#         执行顺序：父类框架逻辑 → 子类业务逻辑（航点判断/切换）
#         """
#         # 1. 先执行父类 compute 的核心逻辑（指标更新、倒计时、通用重采样）
#         super().compute(dt)
#         # 2. 执行子类核心业务逻辑（航点判断、切换、更新指令）
#         self._update()


#     def _update(self):
#         """主更新函数，完全向量化，处理所有逻辑。"""
#         self.newly_reached_waypoint.fill_(False)

#         robot_pos_xy = self.robot.data.root_pos_w[:, :2]
#         target_pos_xy = self.pos_command_w[:, :2]
#         distance_to_target = torch.norm(robot_pos_xy - target_pos_xy, dim=1)
        
#         reached_mask = distance_to_target < self.cfg.success_threshold
#         if torch.any(reached_mask):
#             self.newly_reached_waypoint[reached_mask] = True
            
#             reached_indices = self.current_waypoint_idx[reached_mask]
#             self.visited_waypoints_mask[reached_mask] |= (1 << reached_indices)
            
#             if self.cfg.cyclic:
#                 self.current_waypoint_idx[reached_mask] = (self.current_waypoint_idx[reached_mask] + 1) % self.num_waypoints
#             else:
#                 next_indices = self.current_waypoint_idx[reached_mask] + 1
#                 self.current_waypoint_idx[reached_mask] = torch.clamp(next_indices, max=self.max_waypoint_idx)

#             self._resample_command(torch.where(reached_mask)[0])
        
#         # 在每一步都更新本体坐标系下的指令
#         self._update_command()

#     # vvvvvvvvvvvv 显式实现父类的抽象方法 vvvvvvvvvvvv
    
#     def _update_command(self):
#         """
#         [实现抽象方法]
#         根据世界坐标系下的指令，计算并更新本体坐标系下的指令。
#         """
#         # 计算世界坐标系下的目标向量
#         target_vec_w = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
#         # 转换到机器人本体坐标系，并更新成员变量
#         self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec_w)
#         # 计算朝向误差，并更新成员变量
#         self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

#     def _update_metrics(self):
#         """
#         [实现抽象方法]
#         向量化地更新所有指标。
#         """
#         self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
#         self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
#         # ... 您可以按需添加更多指标 ...

#     def _resample_command(self, env_ids: torch.Tensor):
#         """
#         [实现抽象方法]
#         向量化地为指定环境采样新的航点指令。
#         """
#         if env_ids.numel() == 0:
#             return
            
#         indices_to_sample = self.current_waypoint_idx[env_ids]
#         sampled_waypoints = self.waypoints_tensor[indices_to_sample]
        
#         # 更新世界坐标系下的位置和朝向指令
#         self.pos_command_w[env_ids, :2] = sampled_waypoints[:, :2]
#         self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]
#         self.heading_command_w[env_ids] = sampled_waypoints[:, 2]

#     def _set_debug_vis_impl(self, debug_vis: bool):
#         """[实现] 创建或设置三个独立的 VisualizationMarkers 对象的可见性。"""
#         if debug_vis:
#             # 创建绿色画笔
#             if not hasattr(self, "current_marker"):
#                 self.current_marker = VisualizationMarkers(self.cfg.current_marker_cfg)
#             self.current_marker.set_visibility(True)
#             # 创建蓝色画笔
#             if not hasattr(self, "visited_marker"):
#                 self.visited_marker = VisualizationMarkers(self.cfg.visited_marker_cfg)
#             self.visited_marker.set_visibility(True)
#             # 创建灰色画笔
#             if not hasattr(self, "unvisited_marker"):
#                 self.unvisited_marker = VisualizationMarkers(self.cfg.unvisited_marker_cfg)
#             self.unvisited_marker.set_visibility(True)
#             if not hasattr(self, "goal_pose_visualizer"):
#                 self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
#             self.goal_pose_visualizer.set_visibility(True)
#         else:
#             if hasattr(self, "current_marker"): self.current_marker.set_visibility(False)
#             if hasattr(self, "visited_marker"): self.visited_marker.set_visibility(False)
#             if hasattr(self, "unvisited_marker"): self.unvisited_marker.set_visibility(False)
#             if hasattr(self, "goal_pose_visualizer_cfg"): self.goal_pose_visualizer.set_visibility(False)


#     def _debug_vis_callback(self, event):
#         """[实现] 将航点分组并用对应的“画笔”进行绘制。"""
#         # 1. 准备数据 (这部分逻辑不变)
#         env_origins = self._env.scene.env_origins
#         all_waypoints_w = self.waypoints_tensor.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
#         all_waypoints_w += env_origins.unsqueeze(1)
        
#         waypoint_indices = torch.arange(self.num_waypoints, device=self.device).expand(self.num_envs, -1)
        
#         current_mask = waypoint_indices == self.current_waypoint_idx.unsqueeze(1)
#         visited_mask = ((self.visited_waypoints_mask.unsqueeze(1) >> waypoint_indices) & 1).bool()
#         visited_mask &= ~current_mask
#         unvisited_mask = ~current_mask & ~visited_mask

#         # 2. 过滤出三组坐标
#         current_translations = all_waypoints_w[current_mask]
#         visited_translations = all_waypoints_w[visited_mask]
#         unvisited_translations = all_waypoints_w[unvisited_mask]

#         # 3. 分别调用三个画笔的 visualize 方法
#         if current_translations.numel() > 0:
#             self.current_marker.set_visibility(True)
#             self.current_marker.visualize(translations=current_translations)
#         else:
#             self.current_marker.set_visibility(False)

#         if visited_translations.numel() > 0:
#             self.visited_marker.set_visibility(True)
#             self.visited_marker.visualize(translations=visited_translations)
#         else:
#             self.visited_marker.set_visibility(False)

#         if unvisited_translations.numel() > 0:
#             self.unvisited_marker.set_visibility(True)
#             self.unvisited_marker.visualize(translations=unvisited_translations)
#         else:
#             self.unvisited_marker.set_visibility(False)
        
#         self.goal_pose_visualizer.visualize(
#             translations=self.robot.data.root_pos_w,
#             orientations=quat_unique(self.robot.data.root_quat_w),
#         )
  

#     # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
#     # --- 公共API，供终止条件和奖励函数使用 ---
    
#     def all_waypoints_visited(self) -> torch.Tensor:
#         """检查是否所有航点都已被访问。"""
#         all_visited_mask = (1 << self.num_waypoints) - 1
#         return (self.visited_waypoints_mask & all_visited_mask) == all_visited_mask
        
#     def reached_final_waypoint(self, success_threshold: float = None) -> torch.Tensor:
#         """检查是否到达最后一个航点。"""
#         threshold = success_threshold if success_threshold is not None else self.cfg.success_threshold
        
#         is_targeting_final = self.current_waypoint_idx >= self.max_waypoint_idx
        
#         final_waypoint_pos = self.waypoints_tensor[-1, :2]
#         distance_to_final = torch.norm(self.robot.data.root_pos_w[:, :2] - final_waypoint_pos, dim=1)
#         is_near_final = distance_to_final < threshold
        
#         return is_targeting_final & is_near_final



class SequentialWaypointCommand(CommandTerm):
    """
    顺序航点命令生成器 (向量化优化版 - 终版)
    
    这个版本显式实现了所有父类要求的抽象方法，结构最规范。
    """
    cfg: "SequentialWaypointCommandCfg" # 替换为您的实际配置类

    def __init__(self, cfg: "SequentialWaypointCommandCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        
        self.robot: Articulation = env.scene[cfg.asset_name]

        # 将航点列表转换为Tensor，便于向量化索引
        self.waypoints_tensor = torch.tensor(cfg.waypoints, dtype=torch.float32, device=self.device)
        self.num_waypoints = self.waypoints_tensor.shape[0]
        self.max_waypoint_idx = self.num_waypoints - 1

        # --- 核心状态变量 ---
        self.current_waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # 世界坐标系下的指令
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        
        # 本体坐标系下的指令 (由 _update_command 负责计算)
        self.pos_command_b = torch.zeros_like(self.pos_command_w)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)

        # “刚刚到达”标志，解耦奖励计算
        self.newly_reached_waypoint = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # 状态追踪（用于指标和复杂逻辑）
        self.visited_waypoints_mask = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        
        # --- 指标 (Metrics) ---
        self.metrics["error_pos_2d"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)
        # ... 其他您需要的指标

        # # 初始化命令
        # self._resample_command(torch.arange(self.num_envs, device=self.device))
        # self._update_command() # 首次计算本体坐标系指令

    @property
    def command(self) -> torch.Tensor:
        """返回在机器人本体坐标系下的指令 [dx, dy, dz, d_yaw]"""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)


    def reset(self, env_ids: Sequence[int] | None = None) -> dict:  # 1. 修改返回类型提示为 dict
        if env_ids is None:
            env_ids = slice(None)
        
        self.current_waypoint_idx[env_ids] = 0
        self.visited_waypoints_mask[env_ids] = 0
        
        if isinstance(env_ids, slice):
            self._resample_command(torch.arange(self.num_envs, device=self.device))
        else:
            self._resample_command(torch.tensor(env_ids, dtype=torch.long, device=self.device))
        
        # 2. 捕获父类 reset 方法返回的指标字典
        extras = super().reset(env_ids)
        
        # 3. 将这个字典返回，履行与 CommandManager 的“合约”
        return extras

    def compute(self, dt: float):
        """
        重写父类 compute 方法，整合自定义更新逻辑
        执行顺序：父类框架逻辑 → 子类业务逻辑（航点判断/切换）
        """
        # 1. 先执行父类 compute 的核心逻辑（指标更新、倒计时、通用重采样）
        super().compute(dt)
        # 2. 执行子类核心业务逻辑（航点判断、切换、更新指令）
        self._update()


    def _update(self):
        """主更新函数，完全向量化，处理所有逻辑。"""
        self.newly_reached_waypoint.fill_(False)

        robot_pos_xy = self.robot.data.root_pos_w[:, :2]
        target_pos_xy = self.pos_command_w[:, :2]
        distance_to_target = torch.norm(robot_pos_xy - target_pos_xy, dim=1)
        
        reached_mask = distance_to_target < self.cfg.success_threshold
        if torch.any(reached_mask):
            self.newly_reached_waypoint[reached_mask] = True
            
            reached_indices = self.current_waypoint_idx[reached_mask]
            self.visited_waypoints_mask[reached_mask] |= (1 << reached_indices)
            
            if self.cfg.cyclic:
                self.current_waypoint_idx[reached_mask] = (self.current_waypoint_idx[reached_mask] + 1) % self.num_waypoints
            else:
                next_indices = self.current_waypoint_idx[reached_mask] + 1
                self.current_waypoint_idx[reached_mask] = torch.clamp(next_indices, max=self.max_waypoint_idx)

            self._resample_command(torch.where(reached_mask)[0])
        
        # 在每一步都更新本体坐标系下的指令
        self._update_command()

    # vvvvvvvvvvvv 显式实现父类的抽象方法 vvvvvvvvvvvv
    
    def _update_command(self):
        """
        [实现抽象方法]
        根据世界坐标系下的指令，计算并更新本体坐标系下的指令。
        """
        # 计算世界坐标系下的目标向量
        target_vec_w = self.pos_command_w - self.robot.data.root_pos_w[:, :3]
        # 转换到机器人本体坐标系，并更新成员变量
        self.pos_command_b[:] = quat_apply_inverse(yaw_quat(self.robot.data.root_quat_w), target_vec_w)
        # 计算朝向误差，并更新成员变量
        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - self.robot.data.heading_w)

    def _update_metrics(self):
        """
        [实现抽象方法]
        向量化地更新所有指标。
        """
        self.metrics["error_pos_2d"] = torch.norm(self.pos_command_w[:, :2] - self.robot.data.root_pos_w[:, :2], dim=1)
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(self.heading_command_w - self.robot.data.heading_w))
        # ... 您可以按需添加更多指标 ...

    def _resample_command(self, env_ids: torch.Tensor):
        """
        [实现抽象方法]
        向量化地为指定环境采样新的航点指令。
        """
        if env_ids.numel() == 0:
            return
            
        indices_to_sample = self.current_waypoint_idx[env_ids]
        sampled_waypoints = self.waypoints_tensor[indices_to_sample]

        env_origins = self._env.scene.env_origins[env_ids]
        # print(f"env_origins for env_ids {env_ids}: {env_origins.detach().cpu().numpy()}")
                
        # 更新世界坐标系下的位置和朝向指令
        self.pos_command_w[env_ids, :2] = sampled_waypoints[:, :2] + env_origins[:, :2]
        self.pos_command_w[env_ids, 2] = self.robot.data.default_root_state[env_ids, 2]
        self.heading_command_w[env_ids] = sampled_waypoints[:, 2]

        # print(
        # "env 0/1/2 cmd:",
        # self.pos_command_w[:3, :2].detach().cpu())

    def _set_debug_vis_impl(self, debug_vis: bool):
        """[实现] 创建或设置三个独立的 VisualizationMarkers 对象的可见性。"""
        if debug_vis:
            # 创建绿色画笔
            if not hasattr(self, "current_marker"):
                self.current_marker = VisualizationMarkers(self.cfg.current_marker_cfg)
            self.current_marker.set_visibility(True)
            # 创建蓝色画笔
            if not hasattr(self, "visited_marker"):
                self.visited_marker = VisualizationMarkers(self.cfg.visited_marker_cfg)
            self.visited_marker.set_visibility(True)
            # 创建灰色画笔
            if not hasattr(self, "unvisited_marker"):
                self.unvisited_marker = VisualizationMarkers(self.cfg.unvisited_marker_cfg)
            self.unvisited_marker.set_visibility(True)
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "current_marker"): self.current_marker.set_visibility(False)
            if hasattr(self, "visited_marker"): self.visited_marker.set_visibility(False)
            if hasattr(self, "unvisited_marker"): self.unvisited_marker.set_visibility(False)
            if hasattr(self, "goal_pose_visualizer_cfg"): self.goal_pose_visualizer.set_visibility(False)


    def _debug_vis_callback(self, event):
        """[实现] 将航点分组并用对应的“画笔”进行绘制。"""
        # 1. 准备数据 (这部分逻辑不变)
        env_origins = self._env.scene.env_origins
        all_waypoints_w = self.waypoints_tensor.unsqueeze(0).expand(self.num_envs, -1, -1).clone()
        all_waypoints_w += env_origins.unsqueeze(1)
        
        # (num_envs,num_waypoints)        
        waypoint_indices = torch.arange(self.num_waypoints, device=self.device).expand(self.num_envs, -1)
        
        current_mask = waypoint_indices == self.current_waypoint_idx.unsqueeze(1)
        visited_mask = ((self.visited_waypoints_mask.unsqueeze(1) >> waypoint_indices) & 1).bool()
        visited_mask &= ~current_mask
        unvisited_mask = ~current_mask & ~visited_mask

        # 2. 过滤出三组坐标
        current_translations = all_waypoints_w[current_mask]
        visited_translations = all_waypoints_w[visited_mask]
        unvisited_translations = all_waypoints_w[unvisited_mask]

        # 3. 分别调用三个画笔的 visualize 方法
        if current_translations.numel() > 0:
            self.current_marker.set_visibility(True)
            self.current_marker.visualize(translations=current_translations)
        else:
            self.current_marker.set_visibility(False)

        if visited_translations.numel() > 0:
            self.visited_marker.set_visibility(True)
            self.visited_marker.visualize(translations=visited_translations)
        else:
            self.visited_marker.set_visibility(False)

        if unvisited_translations.numel() > 0:
            self.unvisited_marker.set_visibility(True)
            self.unvisited_marker.visualize(translations=unvisited_translations)
        else:
            self.unvisited_marker.set_visibility(False)
        
        self.goal_pose_visualizer.visualize(
            translations=self.robot.data.root_pos_w,
            orientations=quat_unique(self.robot.data.root_quat_w),
        )
  

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    # --- 公共API，供终止条件和奖励函数使用 ---
    
    def all_waypoints_visited(self) -> torch.Tensor:
        """检查是否所有航点都已被访问。"""
        all_visited_mask = (1 << self.num_waypoints) - 1
        return (self.visited_waypoints_mask & all_visited_mask) == all_visited_mask
        
    def reached_final_waypoint(self, success_threshold: float = None) -> torch.Tensor:
        """检查是否到达最后一个航点。"""
        threshold = success_threshold if success_threshold is not None else self.cfg.success_threshold
        
        is_targeting_final = self.current_waypoint_idx >= self.max_waypoint_idx
        final_waypoint_pos_local = self.waypoints_tensor[-1, :2]
        env_origns = self._env.scene.env_origins[:, :2]
        final_waypoint_pos = final_waypoint_pos_local + env_origns
        distance_to_final = torch.norm(self.robot.data.root_pos_w[:, :2] - final_waypoint_pos, dim=1)
        is_near_final = distance_to_final < threshold
        
        return is_targeting_final & is_near_final
