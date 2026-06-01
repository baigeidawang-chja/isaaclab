# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import math 
from dataclasses import MISSING, field
from typing import Callable, List
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from pxr import UsdPhysics, Gf

from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG, POSITION_GOAL_MARKER_CFG, CUBOID_MARKER_CFG, SPHERE_MARKER_CFG


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class BodyForceAction(ActionTerm):
    """Action term that applies forces along specified Xform axes to bodies.
    
    这个ActionTerm接收N个力值，并将它们沿指定body的Xform的指定轴方向施加。
    支持：
    - 直接指定RigidBody名称（如 'propeller01'）
    - 指定Xform节点名称（如 'Xform_propeller02'），会自动找到父RigidBody并转换坐标系
    """

    cfg: "BodyForceActionCfg"
    """The configuration of the action term."""
    _asset: Articulation
    """The articulation asset on which the action term is applied."""

    def __init__(self, cfg: "BodyForceActionCfg", env: "ManagerBasedEnv"):
        # initialize the action term
        super().__init__(cfg, env)
        
        # get the asset
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Resolve body names and Xform transforms
        # For each name, try to find as RigidBody first, then as Xform child
        resolved_body_names = []
        xform_transforms = []  # Store (pos_offset, quat_offset) for each Xform
        
        template_prim = sim_utils.find_first_matching_prim(self._asset.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self._asset.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString
        stage = template_prim.GetStage()
        
        for xform_name in cfg.body_names:
            # Try to find as RigidBody first
            try:
                body_ids, body_names = self._asset.find_bodies([xform_name])
                if len(body_names) > 0:
                    resolved_body_names.append(body_names[0])
                    xform_transforms.append((None, None))  # No transform needed
                    continue
            except:
                pass
            
            # If not found, try to find as Xform and get its parent RigidBody
            xform_prim = sim_utils.find_first_matching_prim(f"{template_prim_path}/{xform_name}")
            if xform_prim is not None:
                # Find parent RigidBody by traversing up the hierarchy
                current_prim = xform_prim
                parent_rigid_body = None
                
                while current_prim is not None:
                    if current_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        parent_rigid_body = current_prim
                        break
                    current_prim = current_prim.GetParent()
                
                if parent_rigid_body is not None:
                    # Get the body name
                    parent_body_name = parent_rigid_body.GetPath().pathString.split("/")[-1]
                    # Verify this body exists
                    try:
                        body_ids, body_names = self._asset.find_bodies([parent_body_name])
                        if len(body_names) > 0:
                            resolved_body_names.append(parent_body_name)
                            
                            # Get Xform transform relative to parent RigidBody
                            # Get world transform of parent RigidBody
                            parent_world_xform = parent_rigid_body.GetLocalToWorldTransform()
                            # Get world transform of Xform
                            xform_world_xform = xform_prim.GetLocalToWorldTransform()
                            # Compute relative transform: Xform w.r.t. parent
                            relative_xform = xform_world_xform * parent_world_xform.GetInverse()
                            
                            # Extract position and rotation
                            relative_pos = relative_xform.ExtractTranslation()
                            relative_rot = relative_xform.ExtractRotationQuat()
                            
                            # Convert to torch tensors (will be broadcasted later)
                            pos_offset = torch.tensor(
                                [relative_pos[0], relative_pos[1], relative_pos[2]], 
                                device=self.device
                            )
                            # Convert quaternion from (x, y, z, w) to (w, x, y, z)
                            quat_offset = torch.tensor(
                                [relative_rot.real, relative_rot.imaginary[0], relative_rot.imaginary[1], relative_rot.imaginary[2]],
                                device=self.device
                            )
                            
                            xform_transforms.append((pos_offset, quat_offset))
                            continue
                    except:
                        pass
            
            # If all methods fail, raise error
            raise ValueError(
                f"Could not find body or Xform '{xform_name}'. "
                f"Available bodies: {self._asset.body_names}"
            )
        
        # Now resolve all body IDs
        body_ids, body_names = self._asset.find_bodies(resolved_body_names, preserve_order=True)
        
        self._body_ids = body_ids
        self._body_names = body_names
        self._num_bodies = len(body_ids)
        self._xform_transforms = xform_transforms  # Store transforms for later use
        
        # validate that we have the correct number of bodies
        if self._num_bodies != len(cfg.force_axes):
            raise ValueError(
                f"Number of bodies ({self._num_bodies}) must match number of force axes ({len(cfg.force_axes)}). "
                f"Bodies: {body_names}, Axes: {cfg.force_axes}"
            )
        
        # create buffers for forces
        self._raw_actions = torch.zeros(self.num_envs, self._num_bodies, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._forces = torch.zeros(self.num_envs, self._num_bodies, 3, device=self.device)
        
        # parse scale
        if isinstance(cfg.scale, (float, int)):
            self._scale = float(cfg.scale)
        elif isinstance(cfg.scale, (list, tuple)):
            if len(cfg.scale) != self._num_bodies:
                raise ValueError(f"Scale list length ({len(cfg.scale)}) must match number of bodies ({self._num_bodies})")
            self._scale = torch.tensor(cfg.scale, device=self.device).unsqueeze(0)  # (1, num_bodies)
        else:
            raise ValueError(f"Unsupported scale type: {type(cfg.scale)}. Supported types are float, int, list, or tuple.")
        
        # parse offset
        if isinstance(cfg.offset, (float, int)):
            self._offset = float(cfg.offset)
        elif isinstance(cfg.offset, (list, tuple)):
            if len(cfg.offset) != self._num_bodies:
                raise ValueError(f"Offset list length ({len(cfg.offset)}) must match number of bodies ({self._num_bodies})")
            self._offset = torch.tensor(cfg.offset, device=self.device).unsqueeze(0)  # (1, num_bodies)
        else:
            raise ValueError(f"Unsupported offset type: {type(cfg.offset)}. Supported types are float, int, list, or tuple.")
        
        # parse force axes: 'x', 'y', 'z', or indices 0, 1, 2
        self._force_axis_indices = []
        for axis in cfg.force_axes:
            if isinstance(axis, str):
                axis_lower = axis.lower()
                if axis_lower == 'x':
                    self._force_axis_indices.append(0)
                elif axis_lower == 'y':
                    self._force_axis_indices.append(1)
                elif axis_lower == 'z':
                    self._force_axis_indices.append(2)
                else:
                    raise ValueError(f"Invalid force axis '{axis}'. Must be 'x', 'y', 'z', or 0, 1, 2.")
            elif isinstance(axis, int):
                if axis not in [0, 1, 2]:
                    raise ValueError(f"Invalid force axis index {axis}. Must be 0, 1, or 2.")
                self._force_axis_indices.append(axis)
            else:
                raise ValueError(f"Invalid force axis type {type(axis)}. Must be str or int.")
        
        self._force_axis_indices = torch.tensor(self._force_axis_indices, device=self.device)

    @property
    def action_dim(self) -> int:
        return self._num_bodies

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """Process raw actions: apply scale and offset."""
        # store raw actions
        self._raw_actions[:] = actions
        
        # apply scale and offset
        if isinstance(self._scale, (float, int)):
            self._processed_actions[:] = self._raw_actions * self._scale + self._offset
        else:
            self._processed_actions[:] = self._raw_actions * self._scale + self._offset
        
        # apply clip if specified
        if self.cfg.clip is not None:
            clip_min, clip_max = self.cfg.clip
            self._processed_actions[:] = torch.clamp(self._processed_actions, min=clip_min, max=clip_max)

    def apply_actions(self):
        """Apply forces along specified axes in each body's local frame (or Xform frame if specified)."""
        # Initialize force vectors
        self._forces.zero_()
        
        # Get body orientations in world frame (for coordinate transformation)
        body_quats_w = self._asset.data.body_quat_w[:, self._body_ids]  # (num_envs, num_bodies, 4)
        
        # For each body, compute force in the appropriate frame
        for i, axis_idx in enumerate(self._force_axis_indices):
            force_magnitude = self._processed_actions[:, i]  # (num_envs,)
            
            # Get Xform transform if available
            pos_offset, quat_offset = self._xform_transforms[i]
            
            if pos_offset is not None and quat_offset is not None:
                # Force is specified in Xform's local frame
                # Create force vector in Xform's local frame
                force_xform_local = torch.zeros(self.num_envs, 3, device=self.device)
                force_xform_local[:, axis_idx] = force_magnitude
                
                # Transform force from Xform's local frame to body's local frame
                # Xform's orientation relative to body: quat_offset
                # Force in body's local frame = quat_offset^-1 * force_xform_local * quat_offset
                # For vectors, we use quat_rotate
                force_body_local = math_utils.quat_rotate(
                    math_utils.quat_inv(quat_offset.unsqueeze(0).expand(self.num_envs, -1)),
                    force_xform_local
                )
                
                self._forces[:, i] = force_body_local
            else:
                # Force is directly in body's local frame
                self._forces[:, i, axis_idx] = force_magnitude
        
        # Set external forces (torques are zero)
        torques = torch.zeros_like(self._forces)  # (num_envs, num_bodies, 3)
        self._asset.set_external_force_and_torque(
            forces=self._forces,
            torques=torques,
            body_ids=self._body_ids,
        )

    def reset(self, env_ids: torch.Tensor | None = None):
        """Reset action buffers."""
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._forces[env_ids] = 0.0


@configclass
class BodyForceActionCfg(ActionTermCfg):
    """Configuration for body force action term."""

    class_type: type[ActionTerm] = BodyForceAction

    asset_name: str = "robot"
    """Name of the articulation asset."""

    body_names: list[str] = None
    """List of body names or Xform names to apply forces to.
    
    Can be:
    - Direct RigidBody names (e.g., 'propeller01')
    - Xform node names (e.g., 'Xform_propeller02') - will find parent RigidBody and transform coordinates
    """

    force_axes: list[str | int] = None
    """List of axes along which to apply forces for each body/Xform.
    
    Each element can be:
    - 'x', 'y', 'z' (case-insensitive)
    - 0, 1, 2 (for x, y, z respectively)
    
    Length must match the number of bodies/Xforms.
    
    If a Xform is specified, the axis is in the Xform's local frame.
    """

    scale: float | list[float] = 1.0
    """Scale factor for the force actions. Can be a single value or a list (one per body)."""

    offset: float | list[float] = 0.0
    """Offset factor for the force actions. Can be a single value or a list (one per body)."""

    clip: tuple[float, float] | None = None
    """Clip range for processed actions. If None, no clipping is applied."""


class GearRackActionTerm(ActionTerm):
    """齿轮齿条联动动作项：主动齿轮→齿条→从动齿轮的联动控制"""

    cfg: GearRackActionCfg
    """配置实例"""
    _asset: Articulation
    """绑定的机器人（强制为Articulation类型，确保关节控制接口可用）"""

    def __init__(self, cfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._asset = env.scene[self.cfg.asset_name]
        # ------------------------------------------------
        # 关节索引
        # ------------------------------------------------
        self._active_idx, _ = self._asset.find_joints([self.cfg.active_gear_joint])
        self._rack_left_idx, _ = self._asset.find_joints([self.cfg.rack_left_joint])
        self._rack_right_idx, _ = self._asset.find_joints([self.cfg.rack_right_joint])
        self._slave_0103_idx, _ = self._asset.find_joints(self.cfg.slave_0103_joints)
        self._slave_0204_idx, _ = self._asset.find_joints(self.cfg.slave_0204_joints)
        self._servo, _ = self._asset.find_joints(self.cfg.servo_joints)

        self._active_idx = self._active_idx[0]
        self._rack_left_idx = self._rack_left_idx[0]
        self._rack_right_idx = self._rack_right_idx[0]

        assert len(self._slave_0103_idx) == 2
        assert len(self._slave_0204_idx) == 2

        print("\n" + "="*50)
        print("[GearRackActionTerm] Joint ID Mapping Debug:")
        print(f"  Active Gear : ID {self._active_idx:<3} -> {self.cfg.active_gear_joint}")
        print(f"  Rack Left   : ID {self._rack_left_idx:<3} -> {self.cfg.rack_left_joint}")
        print(f"  Rack Right  : ID {self._rack_right_idx:<3} -> {self.cfg.rack_right_joint}")

        for i, (idx, name) in enumerate(zip(self._slave_0103_idx, self.cfg.slave_0103_joints)):
            print(f"  Slave 0103 #{i}: ID {idx:<3} -> {name}")

        for i, (idx, name) in enumerate(zip(self._slave_0204_idx, self.cfg.slave_0204_joints)):
            print(f"  Slave 0204 #{i}: ID {idx:<3} -> {name}")

        for i, (idx, name) in enumerate(zip(self._servo, self.cfg.servo_joints)):
            print(f"  Servo      #{i}: ID {idx:<3} -> {name}")
        print("="*50 + "\n")

        # ------------------------------------------------
        # 传动参数
        # ------------------------------------------------
        device = self._asset.device

        eps = 1e-6
        r_active = max(self.cfg.module * self.cfg.z_active / 2000.0, eps)
        r_slave = max(self.cfg.module * self.cfg.z_slave / 2000.0, eps)
        device = self._asset.device
        self._r_active = torch.tensor(r_active, device=device)
        self._r_slave = torch.tensor(r_slave, device=device)

        # ------------------------------------------------
        # 缓存
        # ------------------------------------------------
        self._raw_actions = torch.zeros((env.num_envs, self.action_dim), device=device)
        self._processed_actions = torch.zeros(
            (env.num_envs, self.action_dim), device=device
        )
        self._prev_processed_actions = torch.zeros(env.num_envs, self.action_dim, device=device)
        self._final_actions = torch.zeros((env.num_envs, 11), device=device)
        # self._joint_pos_min = self._asset.data.joint_limits[:, :, 0].to(device)
        # self._joint_pos_max = self._asset.data.joint_limits[:, :, 1].to(device)


    # ------------------------------------------------
    # 必须接口
    # ------------------------------------------------

    @property
    def action_dim(self) -> int:
        return 5

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions):

        # if torch.isnan(actions).any():
        #     # print("DEBUG: 策略输出包含 NaN，已重置为 0")
        #     actions = torch.nan_to_num(actions, nan=0.0)
        self._raw_actions[:] = actions
        processed = actions.clone()
        processed[:, 0] = torch.clamp(processed[:, 0], min=0.0, max=1.0)
        processed[:, 1:5] = torch.clamp(processed[:, 1:5], min=-1.0, max=1.0)

        action_change = processed - self._prev_processed_actions
        action_change[:, 0] = torch.clamp(action_change[:, 0], min=-self.cfg.max_action_change, max=self.cfg.max_action_change)
        action_change[:, 1:5] = torch.clamp(action_change[:, 1:5], min=-self.cfg.max_servo_change, max=self.cfg.max_servo_change)
        self._processed_actions = self._prev_processed_actions + action_change
        self._prev_processed_actions[:] = self._processed_actions

    def apply_actions(self):
        """Apply joint position targets to the articulation."""
        self._calculate_action()
        joint_ids = [self._active_idx, self._rack_left_idx, self._rack_right_idx] + \
                    self._slave_0103_idx + self._slave_0204_idx + self._servo
        
        current_pos = self._asset.data.joint_pos
        actual_pos = current_pos[:, joint_ids]

        targets = self._final_actions[:, joint_ids]
        self._asset.set_joint_position_target(targets, joint_ids = joint_ids)

        
        # --- debug: print 4 servo action values (env0) ---
        env_id = 0
        # processed servo actions are columns 1:5 in self._processed_actions
        servo_actions = self._processed_actions[env_id, 1:5].detach().cpu().tolist()
        print(
            "\r[GearRackActionTerm] servo_actions(a1..a4) = "
            f"{[f'{x:+.4f}' for x in servo_actions]}   ",
            end="",
        )
        # print("\n" + "="*30 + " [Apply Actions Debug] " + "="*30)
        # # 只打印第一个环境 (Env 0) 的数据，避免刷屏
        # env_id = 0
        # rad2deg = 180.0 / math.pi
        # # 打印主动齿轮 (Active Gear)
        # idx_in_list = 0 # joint_ids[0] 是 active_idx
        # t_active = targets[env_id, idx_in_list].item()
        # a_active = actual_pos[env_id, idx_in_list].item()
        # print(f"Active Gear : Target={t_active*rad2deg: .4f} | Actual={a_active*rad2deg: .4f} | Diff={(t_active-a_active)*rad2deg: .4f}")
        
        # # 打印左齿条 (Rack Left)
        # idx_in_list = 1 # joint_ids[1] 是 rack_left
        # t_rack = targets[env_id, idx_in_list].item()
        # a_rack = actual_pos[env_id, idx_in_list].item()
        # print(f"Rack Left   : Target={t_rack: .4f} | Actual={a_rack: .4f} | Diff={t_rack-a_rack: .4f}")

        # # 打印第一个从动齿轮 (Slave 01)
        # idx_in_list = 3 # joint_ids[3] 是 slave_0103[0]
        # t_slave = targets[env_id, idx_in_list].item()
        # a_slave = actual_pos[env_id, idx_in_list].item()
        # print(f"Slave 01    : Target={t_slave*rad2deg: .4f} | Actual={a_slave*rad2deg: .4f} | Diff={(t_slave-a_slave)*rad2deg: .4f}")
        
        # print("="*80 + "\n")

        # if hasattr(self._asset.data, "joint_pos_target"):
        #     tgt_read = self._asset.data.joint_pos_target[:, joint_ids]
        #     print("[DEBUG] target write vs readback (env0):")
        #     print("  write :", targets[0].detach().cpu())
        #     print("  read  :", tgt_read[0].detach().cpu())


    def _calculate_action(self):
        deg2rad = math.pi /180
        a = self._processed_actions[:, 0]
        # 主动齿轮角度目标（rad）
        q_active = a * float(self.cfg.active_angle_scale) * deg2rad
        self._final_actions[:, self._active_idx] = q_active
        # 齿条位移（m）：x = q * r
        x_l = q_active * self._r_active * float(self.cfg.rack_scale)
        self._final_actions[:, self._rack_left_idx] = x_l
        self._final_actions[:, self._rack_right_idx] = -x_l
        # 从动齿轮角度（rad）：q = x / r
        q_slave0103 = x_l / self._r_slave
        q_slave0204 = -x_l / self._r_slave
        self._final_actions[:, self._slave_0103_idx[0]] = q_slave0103
        self._final_actions[:, self._slave_0103_idx[1]] = q_slave0103
        self._final_actions[:, self._slave_0204_idx[0]] = q_slave0204
        self._final_actions[:, self._slave_0204_idx[1]] = q_slave0204

        
        servo_rad = (self._processed_actions[:, 1:5] * 180.0) * deg2rad
        self._final_actions[:, self._servo[0]] = servo_rad[:, 0]
        self._final_actions[:, self._servo[1]] = servo_rad[:, 1]
        self._final_actions[:, self._servo[2]] = servo_rad[:, 2]
        self._final_actions[:, self._servo[3]] = servo_rad[:, 3]

    """
    调试可视化（官方接口要求，可选实现）
    """
    def _set_debug_vis_impl(self, debug_vis: bool):
        raise NotImplementedError(f"{self.__class__.__name__} 未实现调试可视化")

    def _debug_vis_callback(self, event):
        raise NotImplementedError(f"{self.__class__.__name__} 未实现调试可视化回调")


@configclass
class GearRackActionCfg(ActionTermCfg):
    """齿轮齿条联动动作配置类"""
    # 基础配置（ActionTerm基类要求）
    class_type: type[ActionTerm] = GearRackActionTerm
    asset_name: str = "robot"  # 绑定的机器人资产名称
    debug_vis: bool = False    # 调试可视化开关
    # 机械参数配置
    active_gear_joint : str = "joint_revolute_baselink_gear00"    # 主动齿轮关节名
    rack_left_joint : str = "joint_prismatic_gear00_rack01"
    rack_right_joint : str = "joint_prismatic_gear00_rack02"
    slave_0103_joints = [
        "joint_revolute_rack01_gear01",
        "joint_revolute_rack03_gear03",
    ]

    slave_0204_joints = [
        "joint_revolute_rack02_gear02",
        "joint_revolute_rack04_gear04",
    ]
    
    servo_joints = [
        "joint_revolute_servo01_propeller01",
        "joint_revolute_servo02_propeller02",
        "joint_revolute_servo03_propeller03",
        "joint_revolute_servo04_propeller04",
    ]

    module: float = 2        # 齿轮模数
    z_active: int = 20         # 主动齿轮齿数
    z_slave: int = 20          # 从动齿轮齿数
    active_angle_scale = 90
    rack_scale = 1.0
    max_action_change: float = 0.01
    max_servo_change: float = 0.001


class XformZForceActionTerm(ActionTerm):
    """Action term that applies forces along local Z axis to specified prims."""

    cfg: XformZForceActionCfg
    _asset: Articulation

    def __init__(self, cfg: XformZForceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self._asset = env.scene[self.cfg.asset_name]

        if len(self.cfg.body_names) != 4:
            raise ValueError(f"body_names must contain 4 prim paths, got {len(self.cfg.body_names)}")
        if len(self.cfg.force_scale) != 4:
            raise ValueError(f"force_scale must contain 4 values, got {len(self.cfg.force_scale)}")

        body_ids, body_names = self._asset.find_bodies(self.cfg.body_names, preserve_order=True)
        self._body_ids = body_ids
        device = self._asset.device

        self._raw_actions = torch.zeros((env.num_envs, self.action_dim), device=device)
        self._processed_actions = torch.zeros((env.num_envs, self.action_dim), device=device)
        self._force_scale = torch.tensor(self.cfg.force_scale, device=device).view(1, 4)
        self._force_offset = torch.tensor(self.cfg.force_offset, device=device).view(1, 4)
        self._forces_b = torch.zeros((env.num_envs, 4, 3), device=device)
        self._torques_b = torch.zeros((env.num_envs, 4, 3), device=device)
        self._prev_f = torch.zeros((env.num_envs, 4, 1), device=device)
        self._force_markers_storage: VisualizationMarkers | None = None

    @property
    def action_dim(self) -> int:
        return 4

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = torch.clamp(actions, 0.0, 1.0)

    def apply_actions(self):
        f = (self._processed_actions * self._force_scale+ self._force_offset).unsqueeze(-1)  # (num_envs, 4, 1)

        action_change = f - self._prev_f
        action_change = torch.clamp(action_change, min = -self.cfg.max_change, max = self.cfg.max_change)
        f = self._prev_f + action_change
        self._prev_f = f

        self._forces_b.zero_()
        self._torques_b.zero_()
        self._forces_b[:, :, 2:3] = f  # set Z component
        total_mass = self._asset.data.default_mass[0].sum().item()
        gravity = total_mass * 9.81
        current_forces = f[0, :, 0].detach().cpu().tolist()
        print(f"\r[Debug] Mass: {total_mass:.2f}kg | Gravity: {gravity:.1f}N | Total Thrust: {sum(current_forces):.1f}N | Forces: {[f'{x:.1f}' for x in current_forces]}   ", end="")


        self._asset.set_external_force_and_torque(
            forces=self._forces_b,
            torques=self._torques_b,
            body_ids=self._body_ids,
        )
    
    def _set_debug_vis_impl(self, debug_vis):
            fm = getattr(self, "_force_markers_storage", None)
            if fm is not None:
                fm.set_visibility(debug_vis)

    def _debug_vis_callback(self, event):
        dbg_vis = getattr(self.cfg, "debug_vis", False)
        if not dbg_vis:
            return

        # 2. 懒加载：如果 markers 不存在（被重置了），现在立刻创建
        fm = getattr(self, "_force_markers_storage", None)
        if fm is None:
            # print(f"[XformZForceActionTerm] Lazy creating markers in callback (self_id={id(self)})")
            cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
                prim_path=self.cfg.debug_vis_prim_path
            )
            cfg.markers["arrow"].scale = (0.1, 0.1, 1) 
            self._force_markers_storage = VisualizationMarkers(cfg)
            fm = self._force_markers_storage
            fm.set_visibility(True)

        env_id = 0

        # world pose of all bodies (env0)
        pos_w_all = self._asset.data.body_pos_w[env_id]    # (num_bodies, 3)
        quat_w_all = self._asset.data.body_quat_w[env_id]  # (num_bodies, 4) (w,x,y,z)

        translations = pos_w_all[self._body_ids].clone()   # (4,3)  -> body base point (COM in IsaacLab data)
        body_quat = quat_w_all[self._body_ids].clone()     # (4,4)
        # print("\n[XformZForceActionTerm Debug] env0")
        # for i, body_id in enumerate(self._body_ids):
        #     p = pos_w_all[body_id].detach().cpu().tolist()
        #     qbw = quat_w_all[body_id].detach().cpu().tolist()
        #     print(
        #         f"  #{i} body_id={int(body_id):>3} name='{self._asset.body_names[int(body_id)]}'"
        #         f" | pos_w={p}"
        #         f" | quat_w(wxyz)={qbw}"
        #     )

        # force magnitudes (along each body's LOCAL +Z), N
        fz = self._forces_b[env_id, :, 2]  # (4,)

        # Marker prototype is an +X arrow (RED_ARROW_X_MARKER_CFG).
        # We want arrow +X to align with body local +Z => extra rotation +90deg about Y (X->Z)
        q_x_to_z = torch.tensor(
            [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0],  # (w,x,y,z)
            device=translations.device,
            dtype=translations.dtype,
        ).view(1, 4).expand(4, 4)

        # For negative fz, reverse arrow direction: 180deg about Y flips +X to -X
        q_flip_x = torch.tensor(
            [0.0, 0.0, 1.0, 0.0],  # (w,x,y,z)
            device=translations.device,
            dtype=translations.dtype,
        ).view(1, 4).expand(4, 4)

        # orientation: body orientation then map marker axis
        q = math_utils.quat_mul(body_quat, q_x_to_z)

        neg = (fz < 0.0).view(4, 1)
        if neg.any():
            q_neg = math_utils.quat_mul(q, q_flip_x)
            q = torch.where(neg.expand(4, 4), q_neg, q)

        # Length proportional to |F|. Marker stretches along its +X axis -> scales[:,0]
        vis_scale = float(self.cfg.debug_vis_scale)  # meters per Newton
        length = (fz.abs() * vis_scale).clamp(min=1e-6)    # (4,)

        scales = torch.ones((4, 3), device=translations.device, dtype=translations.dtype)
        scales[:, 0] = length

        # Ensure arrow STARTS at body point:
        # shift marker origin by +0.5*length along arrow direction so that tail aligns with translations.
        dir_w = math_utils.quat_rotate(
            q, torch.tensor([1.0, 0.0, 0.0], device=translations.device, dtype=translations.dtype).expand(4, 3)
        )
        translations = translations + dir_w * (0.25 * length).unsqueeze(-1)

        # RED_ARROW_X_MARKER_CFG has a single prototype -> indices all zeros
        marker_indices = torch.zeros((4,), device=translations.device, dtype=torch.int64)

        fm.visualize(
            translations=translations,
            orientations=q,
            scales=scales,
            marker_indices=marker_indices,
        )

@configclass
class XformZForceActionCfg(ActionTermCfg):
    """Apply forces along each prim's local +Z axis.

    The action space is 4D:
      actions[:, i] in [-1, 1] -> force magnitude scaled by `force_scale[i]` (N).
    """
    class_type: type[ActionTerm] = XformZForceActionTerm
    asset_name: str = MISSING
    """"Name of the asset in the scene (kept for interface consistency; not required for applying forces)."""

    body_names: list[str] = MISSING
    """List of 4 prim paths (Xforms / rigid bodies) to apply forces on."""

    force_scale: list[float] = field(default_factory=lambda: [70.0, 70.0, 70.0, 70.0])
    force_offset: list[float] = field(default_factory=lambda: [20.0, 20.0, 20.0, 20.0])

    """Force scale (N) per prim."""
    max_change: float = 20

    debug_vis: bool = True
    debug_vis_scale: float = 0.08  # meters per Newton
    debug_vis_prim_path: str = "/World/Visuals/KY2/Forces"
