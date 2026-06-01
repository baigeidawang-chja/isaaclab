# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import torch
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg,TerrainGeneratorCfg,MeshPlaneTerrainCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ImuCfg

import isaaclab.envs.mdp as mdp

from .common import reward, termination, action
from .assets import KY2_CFG

##
# Pre-defined configs
##



@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path="/home/chja/myproject/no_water.usd",
        )


    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     env_spacing=1.0,
    #     terrain_generator=TerrainGeneratorCfg(
    #         seed=42,
    #         curriculum=False,
    #         size=(800.0, 800.0),
    #         border_width=0.0,
    #         border_height=0.0,
    #         num_rows=1,
    #         num_cols=1,
    #         horizontal_scale=0.1,
    #         vertical_scale=0.005,
    #         color_scheme="none",
    #         sub_terrains={
    #             "flat": MeshPlaneTerrainCfg(),
    #         },
    #         difficulty_range=(0.0, 1.0),
    #         use_cache=True,
    #         cache_dir="/tmp/isaaclab/terrains",
    #     ),
    #     visual_material=sim_utils.PreviewSurfaceCfg(
    #         diffuse_color=(1.0, 1.0, 1.0),
    #         emissive_color=(0.2, 0.2, 0.2),
    #         metallic=0.0,
    #         roughness=0.5,
    #     ),
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    # )

    # robot
    robot: ArticulationCfg = KY2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # lights - 使用合理的配置
    # 方案1：使用DistantLightCfg（推荐，最简单）
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(0.95, 0.95, 0.95),
            intensity=3000.0,  # 合理的强度
        ),
    )

    # IMU sensor - 添加到机器人基座
    imu = ImuCfg(
        prim_path="{ENV_REGEX_NS}/Robot/KY205/base_link",
        gravity_bias=(0.0, 0.0, 0.0),  # 禁用重力偏置，因为场景中已禁用重力
        debug_vis=False,  # 可以设为True来可视化IMU
    )
        

##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP.
    
    动作空间包含：
    1. 一个齿轮（revolute）的旋转 - gear_joint
    2. 四个 revolute 旋转 - revolute_joint_1, revolute_joint_2, revolute_joint_3, revolute_joint_4
    3. 沿着四个Xform指定轴的力 - 作用在四个body上，沿各自的Xform指定轴
    """

    # 1. 齿轮旋转（位置控制）
    gear_joint_pos = action.GearRackActionCfg(
        asset_name="robot",  # 绑定到上面配置的机器人
        active_gear_joint  = "joint_revolute_baselink_gear00",    # 主动齿轮关节名
        rack_left_joint = "joint_prismatic_gear00_rack01",
        rack_right_joint = "joint_prismatic_gear00_rack03",
        module=2.0,
        z_active=20,
        z_slave=20,
        debug_vis=False
    )

    # 推进器1-4
    propeller_force = action.XformZForceActionCfg(
        asset_name="robot",
        body_names=["Xform_propeller01",
                    "Xform_propeller02",
                    "Xform_propeller03",
                    "Xform_propeller04"],
        debug_vis=True,
    )
    

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # IMU观测
        imu_orientation = ObsTerm(
            func=mdp.imu_orientation,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        imu_ang_vel = ObsTerm(
            func=mdp.imu_ang_vel,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        imu_lin_acc = ObsTerm(
            func=mdp.imu_lin_acc,
            params={"asset_cfg": SceneEntityCfg("imu")},
        )
        
        # 机器人状态观测
        base_height = ObsTerm(func=mdp.base_pos_z)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        # joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        actions = ObsTerm(func=mdp.last_action)


        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "z": (5, 5),"yaw": (0, 0)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # wave_impulse = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="interval",
    #     interval_range_s=(0.3, 1.2),
    #     params={
    #         # 标量范围：会对 (x,y,z) 每个分量独立采样同样的范围
    #         "force_range": (-80.0, 80.0),     # N
    #         "torque_range": (-30.0, 30.0),    # N·m
    #         # 只作用在 base_link（如果名字不对就改）
    #         "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
    #     },
    # )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) 水平稳定奖励 - 保持roll和pitch接近0
    horizontal_stability = RewTerm(
        func=reward.horizontal_stability_reward,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_roll": 0.1,  # 最大允许roll角度（弧度），约5.7度
            "max_pitch": 0.1,  # 最大允许pitch角度（弧度），约5.7度
        },
    )
    
    keep_position = RewTerm(
        func=reward.keep_position_reward,
        weight=5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_pos": (0.0, 0.0, 5.0),
            "distance_sigma": 1.0,
        },
    )

    # (2) 角速度惩罚 - 惩罚过大的角速度
    ang_vel_penalty = RewTerm(
        func=reward.angular_velocity_penalty,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "threshold": 0.5,  # 角速度阈值（rad/s）
        },
    )
    
    # (3) 动作惩罚 - 惩罚过大的动作
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.01)
    
    # (4) 存活奖励 - 保持存活
    alive = RewTerm(func=mdp.is_alive, weight=1)



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    
    # # (2) Terminate if the robot tilts too much (roll or pitch > threshold)
    # excessive_tilt = DoneTerm(
    #     func=termination.check_excessive_tilt,
    #     params={"max_angle": math.pi /2},  # 最大允许角度（弧度），约28.6度
    # )
    base_out_of_bounds = DoneTerm(
        func=mdp.root_pos_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "x": (-3.0, 3.0),  # x轴允许范围 (米)
            "y": (-3.0, 3.0),  # y轴允许范围 (米)
            "z": ( 1.0, 10.0),    # z轴允许范围 (米)，防止飞太高或掉到地底
        },
    )

@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):  # 将 AntEnvCfg 改为 KY2EnvCfg
    """Configuration for the KY2 horizontal stability environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=128, env_spacing=0.0)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 300.0
        # simulation settings
        self.sim.dt = 0.01  # 减小时间步长，提高碰撞稳定性（从 0.05 改为 0.01）
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.observations.policy.enable_corruption = True
        self.sim.gravity = (0.0, 0.0, -4.9) 
        self.observations.policy.enable_corruption = True


class KY2EnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.device = "cuda:0"
        
