# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import time
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollection, RigidObjectCfg
from isaaclab.utils import ArticulationActions
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg,TerrainGeneratorCfg,MeshPlaneTerrainCfg,HfRandomUniformTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from wheeledlab_assets.mushr import MUSHR_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg

import isaaclab_tasks.manager_based.myproject.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path = "/World/ground",
        terrain_type = "plane",
        # usd_path = "/media/chja/CE54D158C9599027/Assets/Isaac/4.5/Isaac/Environments/Terrains/rough_plane.usd",
        env_spacing = 1.0,
        # visual_material=sim_utils.MdlFileCfg(
        # mdl_path="/media/chja/CE54D158C9599027/Assets/Isaac/4.5/Isaac/IsaacLab/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        # project_uvw=True,
        # texture_scale=(0.25, 0.25),
        # ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        # terrain_generator=TerrainGeneratorCfg(
        # seed=42,                                  # 固定种子使生成结果确定；如不需要可设为 None
        # curriculum=False,                         # 是否使用渐进式难度生成
        # size=(100.0, 100.0),                        # 每个子地形的尺寸 (宽, 长)
        # border_width=0.5,                         # 地形边界宽度
        # border_height=1.0,                        # 地形边界高度
        # num_rows=1,                               # 子地形行数
        # num_cols=1,                               # 子地形列数
        # color_scheme="none",                      # 无特殊色彩方案，可选 "height", "random", "none"
        # horizontal_scale=0.1,                       # x, y 方向离散化尺度
        # vertical_scale=0.005,                       # z 方向离散化尺度
        # slope_threshold=0.75,                     # 斜率阈值
        # sub_terrains={
        #     "flat": MeshPlaneTerrainCfg(),
        # },
        # difficulty_range=(0.0, 1.0),              # 地形难度范围
        # use_cache=True,                           # 是否使用缓存
        # cache_dir="/tmp/isaaclab/terrains"          # 缓存目录
        # ),
    )

    #cubes
    # Obstacle = RigidObjectCfg(
    #     prim_path = "{ENV_REGEX_NS}/Obstacle",
    #     spawn = sim_utils.CuboidCfg(
    #         size = (0.1, 0.1, 0.1),
    #         mass_props = sim_utils.MassPropertiesCfg(mass = 10000),
    #         rigid_props = sim_utils.RigidBodyPropertiesCfg(
    #             rigid_body_enabled = False,
    #             disable_gravity = False,
    #         ),
    #         collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled = False),
    #         ),
    #     collision_group = -1,

    # )

    robot: ArticulationCfg = MISSING
    #robot: ArticulationCfg = MUSHR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # AAA: AssetBaseCfg = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/AAA",
    # )
    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/robot/Chassis",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     attach_yaw_only=True,
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )



    #contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True,filter_prim_paths_expr=["/World/ground"] )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"/media/chja/CE54D158C95990271/Assets/Isaac/4.5/Isaac/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(10.0, 10.0),
    #     rel_standing_envs=0.02,
    #     rel_heading_envs=1.0,
    #     heading_command=True,
    #     heading_control_stiffness=2,
    #     debug_vis=True,
    #     ranges=mdp.UniformVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
    #     ),
    # )
    pose_command = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        simple_heading=True,
        resampling_time_range=(20.0, 40.0),
        debug_vis=True,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(5.0, 7.0), pos_y=(5.0, 7.0), heading=(math.pi/4, math.pi/4)),
        #ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(-7.0, 7.0), pos_y=(-7.0, 7.0), heading=(math.pi/4, math.pi/4)),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    
    throttle_steer = mdp.RCCar4WDActionCfg(
        wheel_joint_names=[
            "RevoluteJoint07",
            "RevoluteJoint14",
            "RevoluteJoint21",
            "RevoluteJoint28"
        ],
        steering_joint_names=[
            "front_left_wheel_steer",
            "front_right_wheel_steer",
        ],
        base_length=0.325,
        base_width=0.2,
        wheel_radius=0.05,
        scale=(3.0, 0.488),
        no_reverse=True,
        bounding_strategy="clip",
        asset_name="robot",
    )

    # joint_pos = mdp.JointVelocityActionCfg(
    #     asset_name="robot",
    #     joint_names=[
    #                  "RevoluteJoint07",
    #                  "RevoluteJoint14",
    #                  "RevoluteJoint21",
    #                  "RevoluteJoint24"
    #                  ],
    #     scale=0.05,
    #     offset = 0.0,
    #     clip={".*": (0.0, 0.1)},
    #     )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        # velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "pose_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        # height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        #     noise=Unoise(n_min=-0.1, n_max=0.1),
        #     clip=(-1.0, 1.0),
        # )

        # 修改后的目标相对位置观测
        target_rel_pos = ObsTerm(
            func=mdp.get_target_rel_pos,  # 直接引用模块函数
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        # 修改后的目标航向观测
        target_heading = ObsTerm(
            func=mdp.get_target_heading,  # 直接引用模块函数
            noise=Unoise(n_min=-0.1, n_max=0.1)
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Chassis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Chassis"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


# def straight_line_reward(env, min_speed=0.1, angle_threshold=0.1):

#     cmd = env.command_manager.get_term("pose_command")
#     lin_vel = env.scene["robot"].data.root_lin_vel_w[:, :2]
#     speed = torch.norm(lin_vel, dim=1)
#     current_heading = torch.atan2(lin_vel[:, 1], lin_vel[:, 0])
#     heading_diff = torch.abs(torch.remainder(current_heading - cmd._heading_command_w + math.pi, 2 * math.pi) - math.pi)
#     reward = torch.where((speed > min_speed) & (heading_diff < angle_threshold), 1.0, 0.0)
#     return reward

def position_distance(env, command_name: str):
    """Reward position tracking with tanh kernel."""
    command_generator = env.command_manager.get_term(command_name)
    target_pos_w = command_generator._pos_command_w
    robot = env.scene["robot"]
    robot_pos_w = robot.data.root_pos_w[:, :3]
    distance = torch.norm(target_pos_w - robot_pos_w, dim=1)
    # 使用向量化操作生成奖励张量
    reward = torch.where(
        distance >= 5,
        torch.full_like(distance, -10.0),  # 距离≥5时返回-10
        torch.full_like(distance, 0.0)     # 否则返回0
    )
        # print(f"distance{distance}")
    # des_pos_b = distance
    # #新增调试输出（仅打印第一个环境的数据）
    # if env.num_envs > 0:
    #     env_id = 0  # 选择第一个环境
    #     des_pos_debug = des_pos_b[env_id].detach().cpu().numpy()  # 安全转换到CPU
    #     print(f"[Debug] des_pos_b (env {env_id}): x={des_pos_debug[0]:.2f}, y={des_pos_debug[1]:.2f}, z={des_pos_debug[2]:.2f}")
    # # return 1 - torch.exp(-distance / std)
    return reward

def straight_line_reward(env, min_speed=0.1, angle_threshold=0.1):
    """
    当车辆以足够速度前进并且行驶方向与命令目标接近时，给予正奖励
    """
    cmd = env.command_manager.get_term("pose_command")
    lin_vel = env.scene["robot"].data.root_lin_vel_w[:, :2]
    speed = torch.norm(lin_vel, dim=1)
    current_heading = torch.atan2(lin_vel[:, 1], lin_vel[:, 0])
    # cmd._heading_command_w 为命令中期望的航向（需确保该字段已正确设置）
    heading_diff = torch.abs(torch.remainder(current_heading - cmd._heading_command_w + math.pi, 2 * math.pi) - math.pi)
    reward = torch.where((speed > min_speed) & (heading_diff < angle_threshold), 1.0, 0.0)
    return reward

def angular_velocity_penalty(env, threshold=0.1):
    """
    当车辆角速度超过阈值时，给予负向惩罚，抑制过度旋转
    """
    ang_vel = env.scene["robot"].data.root_ang_vel_w[:, 2]
    penalty = torch.where(torch.abs(ang_vel) > threshold,
                          -(torch.abs(ang_vel) - threshold),
                          torch.zeros_like(ang_vel))
    return penalty

def progress_reward(env):
    # current_time = time.time()
    cmd = env.command_manager.get_term("pose_command")
    robot = env.scene["robot"]
    # 假设目标位置存储在 cmd._pos_command_w，且两者均为 [num_envs, 3] 张量
    current_dist = torch.norm(cmd._pos_command_w - robot.data.root_pos_w[:, :3], dim=1)
    # 如果第一次调用，则初始化前一距离
    if not hasattr(env, '_prev_distance'):
        env._prev_distance = current_dist.clone()
        # env._prev_time = current_time
        return torch.zeros_like(current_dist)
    # 计算进展：前一时刻距离减去当前距离，若有正进展则获得正奖励
    # time_interval = current_time - env._prev_time
    progress = env._prev_distance - current_dist
    # env._prev_distance = current_dist.clone()
    # env._prev_time = current_time
    # 放大奖励系数（根据需要调整系数）
    reward = progress * 2.0
    return reward

def goal_reached_reward(env, success_threshold: float = 1, reward_value: float = 1.0):
    cmd = env.command_manager.get_term("pose_command")
    robot = env.scene["robot"]


    target_pos_w = cmd.pos_command_w  # 目标位置
    robot_pos_w = robot.data.root_pos_w[:, :3]  # 机器人位置
    distance = torch.norm(target_pos_w - robot_pos_w, dim=1)  # [B]

    # 创建奖励张量：到达目标时返回1.0，否则返回0.0
    reward = torch.where(
        distance <= success_threshold,
        torch.full_like(distance, reward_value),  # 成功奖励
        torch.zeros_like(distance)                # 未到达无奖励
    )

    return reward

def stationary_penalty(env, speed_threshold=0.01, penalty_value=-1.0):
    lin_vel = env.scene["robot"].data.root_lin_vel_w[:, :2]
    # print(f"当前机器人的线速度 lin_vel: {lin_vel}")
    speed = torch.norm(lin_vel, dim=1)
    # print(f"当前机器人的speed: {speed}")
    penalty = torch.where(speed < speed_threshold,
                          torch.full_like(speed, penalty_value),
                          torch.zeros_like(speed))
    return penalty

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # position_tracking = RewTerm(
    #     func=mdp.position_command_world_tanh,
    #     weight=0.3,
    #     params={"std": 3.0, "command_name": "pose_command"},
    # )
    # position_tracking_fine_grained = RewTerm(
    #     func=mdp.position_command_world_tanh,
    #     weight=0.5,
    #     params={"std": 0.2, "command_name": "pose_command"},
    # )
    # orientation_tracking = RewTerm(
    #     func=mdp.heading_command_abs,
    #     weight=-6,
    #     params={"command_name": "pose_command"},
    # )

    # straight_line_reward = RewTerm(
    #     func=straight_line_reward,
    #     weight=2.0,  # 奖励权重，可根据实际效果适当调整
    #     params={"min_speed": 0.1, "angle_threshold": 0.1},
    # )
    # 新增进展奖励：鼓励车辆实际向目标前进
    progress_reward = RewTerm(
        func=progress_reward,
        weight=4.0,  # 奖励系数，数值可调以确保真正靠近目标能获得显著奖励
        params={},
    )
    stationary_penalty = RewTerm(
        func=stationary_penalty,
        weight=1.0,
        params={"speed_threshold":0.5,"penalty_value":-1.0},
    )
    # 角速度惩罚项：防止车辆通过过快旋转“刷奖励”
    angular_velocity_penalty = RewTerm(
        func=angular_velocity_penalty,
        weight=2,  # 负权值，用于惩罚过大角速度
        params={"threshold": 0.2},
    )

    goal_reached_reward = RewTerm(
        func=goal_reached_reward,
        weight=10,
        params={"success_threshold":0.5,"reward_value":20}
    )
    # position_distance = RewTerm(
    #     func=position_distance,
    #     weight=1,
    #     params={"command_name": "pose_command"},
    # )
    # 移除原position_tracking_fine_grained
    # 新增速度方向奖励
    # velocity_alignment = RewTerm(
    #     func=mdp.velocity_direction_reward,
    #     weight=0.5,
    #     params={"command_name": "pose_command"}
    # )

def goal_reached(env):
    """获取目标相对位置 (世界坐标系)"""
    cmd = env.command_manager.get_term("pose_command")
    robot = env.scene["robot"]
    distance = torch.norm(cmd.pos_command_w - robot.data.root_pos_w[:, :3])
    return torch.logical_or(distance <= 0.5, distance <= 0.5)


def distance_out(env):
    cmd = env.command_manager.get_term("pose_command")
    robot = env.scene["robot"]
    distance = torch.norm(cmd.pos_command_w - robot.data.root_pos_w[:, :3])
    return torch.logical_or(distance >= 20, distance >= 20)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # 新增成功条件
    # goal_reached = DoneTerm(
    #     func=goal_reached,
    # )

    # distance_out = DoneTerm(
    #     func=distance_out,
    # )




@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=10)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 100.0
        # simulation settings
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # if self.scene.height_scanner is not None:
        #     self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.observations.policy.enable_corruption = True
