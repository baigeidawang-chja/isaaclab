# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import torch
import time
from dataclasses import MISSING
 
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCollection, RigidObjectCfg, RigidObjectCollectionCfg, Articulation
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns, ImuCfg
from isaaclab.terrains import TerrainImporterCfg,TerrainGeneratorCfg,MeshPlaneTerrainCfg,HfRandomUniformTerrainCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from wheeledlab_assets.mushr import MUSHR_CFG
from wheeledlab_assets.hound import HOUND_SUS_ACTUATOR_CFG
from wheeledlab_tasks.common import Mushr4WDActionCfg

import isaaclab.envs.mdp as mdp
# import isaaclab_tasks.manager_based.myproject.velocity.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from ...mdp import rewards, observation, terminations_user
from ...mdp.actions import Car4WDActionCfg, CarVWActionCfg
from ...mdp.events import generate_random_grid_obstacle_positions , randomize_obstacles_per_env
from isaaclab_assets import CAR_CFG

##
# Scene definition
##

# OBSTACLE_POSITIONS = [
#     [ 0.0,  -6.2473],
#     [ 38.0, -6.2473],
#     [ 5.0,  -3.0],
#     [ 7.0,  12.7527],
#     [ 8.0, -5.2473],
#     [ 11.0,  1.7527],
#     [ 16.0, 10.7527],
#     [ 15.0,  -6.2473],
#     [ 17.0,  6.7527],
#     [ 20.0, -5.2473],
#     [ 12.0,  -11.2473],
#     [ 21.0, -12.2473],
#     [ 22.0,  15.7527],
#     [ 22.0,  7.7527],
#     [ 28.0, 10.51739],
#     [ 4.0,  8.0],
#     [ 27.0, -14.2473],
#     [ 29.0,  12.7527],
#     [ 27.0,  -9.2473],
#     [ 15.0, 14.7527],
#     [ 44.04432,  1.16013],
#     [ 32.0, -15.2473],
#     [ 32.0,  15.7527],
#     [ 32.0,  -5.2473],
#     [ 38.0,  8.8],
#     [ 38.0, 10.7527],
#     [ 0.0,  18.7527],
#     [ 5.0, -16.2473],
#     [ 11.0,  19.7527],
#     [ 39.0,  21.7527],
#     [ 39.0, -17.2473],
#     [ 26.75, 1.8], 
#     #添加
#     [ 26.75, 0],
#     [ 38, 6.6],

# ]

OBSTACLE_POSITIONS = [
    [ 5.0, 0.0],
    [ 5.0, 0.6],
    [ 5.0, 1.2],
    [ 5.0, 1.8],
    [ 5.0, 2.4],
    [ 5.0, 3.0],
    [ 5.0, 3.6],
    [ 5.0, 4.2],
    [ 5.0, -0.6],
    [ 5.0, -1.2],
    [ 5.0, -1.8],
    [ 5.0, -2.4],
    [ 5.0, -3.0],
    [ 5.0, -3.6],
    [ 5.0, -4.2],
    [ 10.0, 0.3],
    [ 10.0, 0.9],
    [ 10.0, 1.5],
    [ 10.0, 2.1],
    [ 10.0, 2.7],
    [ 10.0, 3.3],
    [ 10.0, 3.9],
    [ 10.0, -0.3],
    [ 10.0, -0.9],
    [ 10.0, -1.5],
    [ 10.0, -2.1],
    [ 10.0, -2.7],
    [ 10.0, -3.3],
    [ 10.0, -3.9],
    [1.0, 0.8],      # 上方
    [1.0, -0.8],     # 下方
    [1.5, 1.5],      # 上侧
    [0.5, -1.5],     # 下侧
    
    # ===== 第二段: (2,0) → (4,2) 之间 =====
    [2.8, 0.5],      # 路径中间偏下
    [3.2, 1.2],      # 路径中间
    [2.5, -0.8],     # 下方阻挡
    [3.5, -0.5],     # 下方
    [3.0, 2.5],      # 上方
    
    # ===== 第三段: (4,2) → (6,0) 之间 =====
    [4.8, 1.5],      # 路径上方
    [5.2, 0.8],      # 路径中间
    [4.5, 0.0],      # 中间阻挡
    [5.5, -0.5],     # 下方
    [5.0, 2.8],      # 上侧
    
    # ===== 第四段: (6,0) → (8,-2) 之间 =====
    [6.8, -0.5],     # 路径上方
    [7.2, -1.2],     # 路径中间
    [6.5, 0.8],      # 上方阻挡
    [7.5, 0.5],      # 上方
    [7.0, -2.8],     # 下侧
    
    # ===== 第五段: (8,-2) → (10,0) 之间 =====
    [8.8, -1.5],     # 路径下方
    [9.2, -0.8],     # 路径中间
    [8.5, 0.0],      # 上方阻挡
    [9.5, 0.5],      # 上方
    [9.0, -3.0],     # 下侧
    
    # ===== 散布在区域各处的额外障碍物 =====
    # 左侧区域
    [0.5, 2.5],
    [1.5, -2.5],
    [0.8, 3.5],
    [1.2, -3.5],
    
    # 中间区域
    [3.5, 3.5],
    [4.5, -2.5],
    [5.5, 3.0],
    [6.5, -3.0],
    
    # 右侧区域
    [7.5, 2.5],
    [8.5, -3.5],
    [9.5, 2.0],
    [9.0, 1.5],
    
    # 边界附近（稀疏）
    [2.0, 4.0],
    [4.0, -4.0],
    [6.0, 4.0],
    [8.0, -4.0],
]


# OBSTACLE_POSITIONS = [
#     ===== 第一段: (0,0) → (2,0) 之间 =====
#     [1.0, 0.8],      # 上方
#     [1.0, -0.8],     # 下方
#     [1.5, 1.5],      # 上侧
#     [0.5, -1.5],     # 下侧
    
#     ===== 第二段: (2,0) → (4,2) 之间 =====
#     [2.8, 0.5],      # 路径中间偏下
#     [3.2, 1.2],      # 路径中间
#     [2.5, -0.8],     # 下方阻挡
#     [3.5, -0.5],     # 下方
#     [3.0, 2.5],      # 上方
    
#     ===== 第三段: (4,2) → (6,0) 之间 =====
#     [4.8, 1.5],      # 路径上方
#     [5.2, 0.8],      # 路径中间
#     [4.5, 0.0],      # 中间阻挡
#     [5.5, -0.5],     # 下方
#     [5.0, 2.8],      # 上侧
    
#     ===== 第四段: (6,0) → (8,-2) 之间 =====
#     [6.8, -0.5],     # 路径上方
#     [7.2, -1.2],     # 路径中间
#     [6.5, 0.8],      # 上方阻挡
#     [7.5, 0.5],      # 上方
#     [7.0, -2.8],     # 下侧
    
#     ===== 第五段: (8,-2) → (10,0) 之间 =====
#     [8.8, -1.5],     # 路径下方
#     [9.2, -0.8],     # 路径中间
#     [8.5, 0.0],      # 上方阻挡
#     [9.5, 0.5],      # 上方
#     [9.0, -3.0],     # 下侧
    
#     ===== 散布在区域各处的额外障碍物 =====
#     左侧区域
#     [0.5, 2.5],
#     [1.5, -2.5],
#     [0.8, 3.5],
#     [1.2, -3.5],
    
#     中间区域
#     [3.5, 3.5],
#     [4.5, -2.5],
#     [5.5, 3.0],
#     [6.5, -3.0],
    
#     右侧区域
#     [7.5, 2.5],
#     [8.5, -3.5],
#     [9.5, 2.0],
#     [9.0, 1.5],
    
#     边界附近（稀疏）
#     [2.0, 4.0],
#     [4.0, -4.0],
#     [6.0, 4.0],
#     [8.0, -4.0],
# ]

NUM_OBSTACLES = len(OBSTACLE_POSITIONS)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

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

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            restitution=0.0,
            friction_combine_mode="min",
            restitution_combine_mode="min",
        ),
        visual_material= sim_utils.PreviewSurfaceCfg(

        ),
    )

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="usd",
    #     env_spacing=0.0,
    #     usd_path=f"/home/chja/myproject/ground2.usd"
    # )

    obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            f"cube_{i:02d}": RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/cube_{i:02d}",
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(OBSTACLE_POSITIONS[i][0], OBSTACLE_POSITIONS[i][1], 0.5)
                ),
                spawn=sim_utils.CuboidCfg(
                    size=(0.2, 0.2, 7.0),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,  # static obstacle (won't be pushed)
                    ),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.2, 0.2),
                        roughness=0.4,
                        metallic=0.0,
                    ),
                ),
            )
            for i in range(NUM_OBSTACLES)
        }
    )
    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/Car/base_link", debug_vis=False)

    robot: ArticulationCfg = CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot") # type: ignore[attr-defined]


    # contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/Car/base_link",
    #     track_air_time=False,
    #     debug_vis=False,
    # )

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
    sequential_waypoints = mdp.SequentialWaypointCommandCfg(
        asset_name="robot",
        debug_vis=True,
        waypoints=[
            [0.0, 0.0, 0.0],       # 起点
            [2.5, 0.0, 0.0], 
            [5.0, 1.0, 0.0],       # 第一个障碍墙前（稍微向上绕）
            [7.5, 2.0, 0.0],       # 绕过第一个障碍墙
            [10.0, 0.0, 0.0],      # 第二个障碍墙前
            [12.5, -1.0, 0.0],     # 绕过第二个障碍墙
            [15.0, 0.0, 0.0],      # 终点
        ],

        # waypoints=[
        #     [0.0, 0.0, 0.0],      # 起点
        #     [2.0, 0.0, 0.0],      # 直行
        #     [4.0, 2.0, 0.0],      # 向上转
        #     [6.0, 0.0, 0.0],      # 向下转
        #     [8.0, -2.0, 0.0],     # 继续向下
        #     [10.0, 0.0, 0.0],     # 终点，回到中线
        # ],
        simple_heading=True,
        success_threshold=0.8,     # 根据车身宽度加大距离阈值
        resampling_time_range=(30.0, 30.0),
        cyclic=False,
    )

# class CommandsCfg:
#     """Command specifications for the MDP."""

#     sequential_waypoints = mdp.SequentialWaypointCommandCfg(
#         asset_name="robot",
#         debug_vis=True,
#         waypoints=[
#             # [0.0, 0.0, 0.0],       # 起点

#             # 第一段：从起点到第一个转弯（更密集）
#             # [2.0, 0.93, 0.0],      # 中间点1
#             [4.0, 1.87, 0.0],      # 中间点2
#             # [6.0, 2.8, 0.0],       # 中间点3
#             # [8.0, 3.73, 0.0],      # 中间点4
#             # [10.0, 4.67, 0.0],     # 中间点5
#             [12.0, 5.6, 0.0],      # 第一个转弯点

#             # 第二段：向左回折（更密集）
#             # [13.5, 4.7, 0.0],      # 中间点6
#             # [15.0, 3.8, 0.0],      # 中间点7
#             # [17.0, 1.9, 0.0],      # 中间点8
#             [19.5, -0.25, 0.0],    # 中间点9
#             # [22.0, -2.4, 0.0],     # 向左回折点

#             # 第三段：S形第二段（更密集）
#             # [24.0, -1.2, 0.0],     # 中间点10
#             [25.0, 0.0, 0.0],      # 中间点11
#             # [26.5, 1.6, 0.0],      # 中间点12
#             # [28.0, 3.2, 0.0],      # 中间点13
#             # [29.5, 4.8, 0.0],      # 中间点14
#             [31.0, 6.4, 0.0],      # 中间点15
#             # [32.5, 8.0, 0.0],      # 中间点16
#             # [34.0, 9.6, 0.0],      # S形第二段点

#             # 第四段：接近终点前的调整（更密集）
#             # [35.5, 8.8, 0.0],      # 中间点17
#             # [37.0, 8.0, 0.0],      # 中间点18
#             [38.5, 7.0, 0.0],      # 中间点19
#             # [40.5, 6.0, 0.0],      # 中间点20
#             # [42.25, 5.0, 0.0],     # 中间点21
#             # [44.0, 4.0, 0.0],      # 接近终点前的调整点

#             # 第五段：到终点（更密集）
#             # [45.5, 6.2, 0.0],      # 中间点22
#             # [47.0, 8.4, 0.0],      # 中间点23
#             [48.5, 10.6, 0.0],     # 中间点24
#             #[50.5, 12.5, 0.0],     # 中间点25
#             [52.25, 13.45, 0.0],   # 中间点26
#             # [54.0, 14.4, 0.0],     # 终点
#         ],
#         simple_heading=True,
#         success_threshold=0.8,     # 根据车身宽度加大距离阈值
#         resampling_time_range=(30.0, 30.0),
#         cyclic=False,
#     )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # throttle_steer = Car4WDActionCfg(
    #     wheel_joint_names=[
    #         "joint_front_right_wheel_link_wheel",
    #         "joint_front_left_wheel_link_wheel",
    #         "joint_back_right_wheel_link_wheel",
    #         "joint_back_left_wheel_link_wheel"
    #     ],
    #     steering_joint_names=[
    #         "joint_front_right_steer",
    #         "joint_front_left_steer",
    #     ],
    #     base_length=2.035 / 5,
    #     base_width=1.1673 / 5,
    #     # wheel_radius=0.675,
    #     scale=(5.0, 5.0, 5.0, 5.0, 1.0),
    #     #scale=(0.0, 0.0, 0.0, 0.0, 0.0),
    #     offset=(0.0, 0.0, 0.0, 0.0, 0.0),
    #     # no_reverse = False,
    #     bounding_strategy="clip",
    #     asset_name="robot",
    #     max_action_change=(1.0, 1.0, 1.0, 1.0, 0.5),
    # )


    throttle_steer = CarVWActionCfg(
        wheel_joint_names=[
            "joint_front_right_wheel_link_wheel",
            "joint_front_left_wheel_link_wheel",
            "joint_back_right_wheel_link_wheel",
            "joint_back_left_wheel_link_wheel"
        ],
        steering_joint_names=[
            "joint_front_right_steer",
            "joint_front_left_steer",
        ],
        base_length=2.035 / 5,
        base_width=1.1673 / 5,
        wheel_radius=0.035,
        scale=(1.5, 1.0),
        #scale=(0.0, 0.0, 0.0, 0.0, 0.0),
        offset=(0.0, 0.0),
        # no_reverse = False,
        bounding_strategy="clip",
        asset_name="robot",
        no_reverse= False,
        # max_action_change=(0.2, 0.2, 0.2, 0.2, 0.5),
    )




@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=[
                    "joint_front_left_wheel_link_wheel",
                    "joint_front_right_wheel_link_wheel",
                    "joint_back_left_wheel_link_wheel",
                    "joint_back_right_wheel_link_wheel"])
            },
            # noise=Unoise(n_min=-1.5, n_max=1.5)
        )

        actions = ObsTerm(func=mdp.last_action)

        # IMU_line_acc_axis_x = ObsTerm(
        #     func=mdp.imu_lin_acc_axis_x,
        # )

        # desired_dir = ObsTerm(
        #     func=observation.desired_world_dir,
        #     params={"x": 1.0, "y": 0.0}, 
        # )

        base_forward_dir = ObsTerm(
            func=observation.base_forward_dir_w,
        )

        get_target_heading = ObsTerm(
            func=observation.get_target_heading,
        )

        # obstacle_grid_map = ObsTerm(
        #     func=observation.get_obstacle_grid_map_scene,
        #     params={
        #         "grid_size": 1.5,  # 3x3米区域
        #         "num_cells": 20,  # 3x3网格，共9个格子
        #         "obstacle_size": 0.2,  # 障碍物半径（米）
        #         "debug_vis": True,  # 启用可视化
        #     },
            # 不需要噪声，因为已经是二值化的
            # noise=Unoise(n_min=-0.01, n_max=0.01),
        #)

        obstacle_grid_map = ObsTerm(
            func=observation.get_obstacle_grid_map_scene,
            params={
                "grid_size": 1.5,  # 3x3米区域
                "num_cells": 20,  # 3x3网格，共9个格子
                "obstacle_size": 0.2,  # 障碍物半径（米）
                "debug_vis": True,  # 启用可视化
            },
        )



    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    # physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
    #         "static_friction_range": (0.8, 0.8),
    #         "dynamic_friction_range": (0.6, 0.6),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )


    randomize_obstacles = EventTerm(
        func=randomize_obstacles_per_env,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("obstacles"),
            "x_range": (2.0, 14.0),      # X范围改为15以内（留出起点和终点空间）
            "y_range": (-5.0, 5.0),      # Y范围适当缩小
            "z_height": 0.5,
            "min_dist_between_obstacles": 0.4,
            "min_dist_from_robot": 1.5,
            "robot_asset_name": "robot",
            "max_tries": 2000,
        },
    )

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "mass_distribution_params": (0.5, 1.0),
    #         "operation": "add",
    #     },
    # )

    # randomize_obstacle_positions = EventTerm(
    #     func=generate_random_grid_obstacle_positions,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("obstacles"),
    #         "grid_rows": 5,  # 网格行数（use_random_distribution=False 时使用）
    #         "grid_cols": 5,  # 网格列数（use_random_distribution=False 时使用）
    #         "base_spacing": 4.0,  # 基础间距（米）
    #         "spacing_variation": 2.0,  # 间距变化范围（±2米）
    #         "x_range": (0.0, 54.0),  # 从0开始，覆盖整个S型轨迹范围（起点0到终点54）
    #         "y_range": (-8, 10.0),  # 扩大y范围，覆盖轨迹的y范围（-2.4到14.4）
    #         "z_height": 0.5,  # Z方向高度
    #         避开机器人初始位置的参数
    #         "robot_init_x_range": (-1.5, 1.5),  # 机器人初始X位置范围
    #         "robot_init_y_range": (-1.5, 1.5),  # 机器人初始Y位置范围
    #         "min_distance_from_robot": 3.0,  # 障碍物距离机器人初始位置的最小距离（米）- 避开初始位置
    #         "min_distance_between_obstacles": 3.0,  # 障碍物之间的最小距离（米）
    #         "use_random_distribution": True,  # 使用完全随机分布，更乱
    #         "max_sample_tries": 1000,  # 每个障碍物位置的最大采样尝试次数
    #         "only_one_set": True,
    #     },
    # )

    # 在 reset 时设置悬架目标位置
    # set_suspension_target = EventTerm(
    #     func=mdp.reset_joints_by_offset,  # 或使用自定义函数
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg(
    #             "robot",
    #             joint_names=[
    #                 "joint_front_right_damper",
    #                 "joint_front_left_damper",
    #                 "joint_back_right_damper",
    #                 "joint_back_left_damper"
    #             ]
    #         ),
    #         "position_range": (0.0, 0.0),  # 固定压缩位置
    #         "velocity_range": (0.0, 0.0),
    #     },
    # )

    # reset
    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
    #         "force_range": (-1.0, 1.0),
    #         "torque_range": (-1.0, 1.0),
    #     },
    # )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0,0), "y": (0, 0), "yaw": (-1, 1)},
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
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(10.0, 15.0),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )

@configclass
class RewardsCfg:

    track_forward_speed_reward = RewTerm(
        func=rewards.track_forward_speed_reward,
        weight=3,
        params={
            "target_speed": 1.5,
            "std": 1.0,
        },
    )

    # heading_align_dir = RewTerm(
    #     func=rewards.heading_align_world_dir_reward,
    #     weight=4,
    #     params={"dir_x": 1.0, "dir_y": 0.0, "scale": 1.0},
    # )

    heading_align_waypoint = RewTerm(
        func=rewards.heading_align_waypoint_reward,
        weight=4,
        params={
            "scale": 1.0,
        },
    )

    # obstacle_repulsion = RewTerm(
    #     func=rewards.obstacle_proximity_repulsion_penalty,
    #     weight=5.0,
    #     params={
    #         "grid_size": 1.5,
    #         "num_cells": 20,
    #         "obstacle_size": 0.2,
    #         "falloff": 1.0,
    #         "max_range": 1.5,      # 只对4m内的占据格敏感（可调/可去掉）
    #         "front_only": True,    # 只看前方
    #         "front_x_min": 0.0,
    #         "scale": 0.5,
    #     },
    # )

    # flip_penalty = RewTerm(
    #     func=rewards.flipped_penalty,
    #     weight=5,
    #     params={
    #         "up_z_threshold": 0.5,
    #         "penalty": -20,
    #     },
    # )

    time_penalty = RewTerm(
        func=rewards.time_penalty,
        weight=1,
        params={"penalty_per_step": -5}
    )


def all_waypoints_completed(env):
    """
    成功条件：检查是否所有航点都被访问且到达最后一个航点

    成功条件：
    1. 所有航点都被访问过（不漏掉任何一个点）
    2. 当前到达最后一个航点（距离足够近）

    Returns:
        Tensor of shape (num_envs,): True表示成功完成任务，False表示未完成
    """
    cmd = env.command_manager.get_term("sequential_waypoints")

    # 检查是否所有航点都被访问
    all_visited = cmd.all_waypoints_visited()

    # 检查是否到达最后一个航点
    reached_final = cmd.reached_final_waypoint(success_threshold=0.8)

    # 两个条件都满足才算成功
    success = all_visited & reached_final

    return success

@configclass
class TerminationsCfg:
    # 时间超时：episode达到最大长度时终止
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    out_of_bounds = DoneTerm(
        func=terminations_user.out_of_bounds,
        params={
            "x_min": -10,
            "x_max": 50,
            "y_min": -10,
            "y_max": 10,
        },
    )

    flip = DoneTerm(
        func=terminations_user.is_flipped,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "up_z_threshold": 0.2,
        },
    )

    # 成功条件：所有航点都被访问且到达最后一个航点
    all_waypoints_completed = DoneTerm(
        func=all_waypoints_completed,
        time_out=False,  # 这是成功终止，不是超时
    )


##
# Environment configuration
##


@configclass
class LocomotionVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=80)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    #actions: Mushr4WDActionCfg = Mushr4WDActionCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 5
        self.episode_length_s = 300.0
        # simulation settings
        self.sim.dt = 0.01  # 减小时间步长，提高碰撞稳定性（从 0.05 改为 0.01）
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material

        self.observations.policy.enable_corruption = True
class MyCarRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.sim.device = "cuda:0"

        # self.events.push_robot = None
        self.events.reset_base.params = {

            "pose_range": {
                "x": (0, 0),  # x方向范围，确保有足够的距离
                "y": (0, 0),  # y方向范围
                "z": (1, 1),   # z方向固定
                "yaw": (-1, 1),  # 初始朝向，面向第一个航点方向
            },
            "velocity_range": {
                "x": (0.0, 0.0),  # 初始速度设为0，让机器人从静止开始
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

class MyCarRoughEnvCfg_PLAY(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 58
        self.scene.env_spacing = 0
