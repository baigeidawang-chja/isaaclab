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
from isaaclab.sensors import ContactSensorCfg, ImuCfg
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
from ..mdp import rewards, observation, terminations_user
from ..mdp.actions import Car4WDActionCfg, CarVWActionCfg
from ..mdp.events import reset_local_nav_task
from isaaclab_assets import CAR_CFG

##
# Scene definition
##


OBSTACLE_POSITIONS = [
    [0.6 + 1.15 * col, -3.2 + 1.25 * row]
    for row in range(10)
    for col in range(10)
]

NUM_OBSTACLES = len(OBSTACLE_POSITIONS)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

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

    robot: ArticulationCfg = CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore[attr-defined]

    robot_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Car/contact_.*",
        history_length=1,
        track_air_time=False,
        update_period=0.0,
        debug_vis=True,
    )

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
    """No command terms are used in the proprio-only local trajectory task."""

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

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
        scale=(1.2, 0.6),
        offset=(0.0, 0.0),
        bounding_strategy="clip",
        use_rate_limit=True,
        max_speed_rate=0.2,
        max_steer_rate=0.3,
        asset_name="robot",
        no_reverse=False,
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

        imu_state = ObsTerm(func=observation.get_imu_state)

        # contact_feedback = ObsTerm(func=observation.get_contact_sensor_feedback)

        # derived_proprio = ObsTerm(func=observation.get_derived_proprio_features)

        # nextprop_target = ObsTerm(func=observation.get_next_proprio_target)

        stuck_label = ObsTerm(func=observation.get_stuck_label)

        # local_tracking_state = ObsTerm(func=observation.get_local_tracking_state)

        # Current target + next waypoint for earlier turn preparation in S-curves.
        future_waypoint_preview = ObsTerm(
            func=observation.get_future_waypoint_preview,
            params={"num_waypoints": 1, "activate_next_after_dist": 0.8},
        )

        # mode_label = ObsTerm(func=observation.get_mode_label)

        # interaction_label = ObsTerm(
        #     func=observation.get_interaction_label,
        #     params={"num_dirs": 12},
        # )

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_local_nav = EventTerm(
        func=reset_local_nav_task,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_asset_cfg": SceneEntityCfg("obstacles"),
            "s_margin_start": 0.0,
            "s_margin_end": 0.0,
            "lateral_offset_range": (-0.08, 0.08),
            "heading_offset_range": (-0.08, 0.08),
            "start_speed_range": (0.0, 0.08),
            "waypoint_reach_thresh": 0.65,
            # 15m x 15m local navigation square.
            "square_x": (0.5, 14.5),
            "square_y": (-7.0, 7.0),
            "obstacle_path_clearance": 0.5,
            "obstacle_spacing": 0.55,
            "obstacle_start_clearance": 1.0,
            "obstacle_goal_clearance": 1.0,
            "debug_vis": True,
            "debug_vis_num_points": 16,
            "debug_vis_ds": 0.5,
        },
    )

    reset_base = None

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    progress_forward = RewTerm(
        func=rewards.local_forward_progress_reward,
        # weight=2.0 -> 4.0
        weight=1.0,
        # scale 6.0 -> 8.0
        # max_delta 0.2 -> 0.12
        params={"scale": 8.0, "max_delta": 0.12},
    )

    waypoint_reached = RewTerm(
        func=rewards.local_waypoint_reached_bonus,
        # weight=8.0 -> 2.5
        weight=8.0,
        # bonus 0.5 -> 0.25
        params={"bonus": 0.25},
    )

    # waypoint_exp_progress = RewTerm(
    #     func=rewards.waypoint_exp_progress_reward,
    #     weight=2.0,
    #     params={"alpha": 2.0, "scale": 1.0, "clip": 0.2},
    # )

    heading_align = RewTerm(
        func=rewards.local_target_heading_alignment_reward,
        weight=0.1,
        params={
            "std": 0.40,
            "directional": True,
            "reverse_floor": 0.15,
            "reverse_speed_thresh": 0.05,
        },
    )

    speed_tracking = RewTerm(
        func=rewards.local_speed_tracking_reward,
        weight=0.4,
        params={"target_speed": 0.45, "std": 0.5},
    )

    time_penalty = RewTerm(
        func=rewards.time_penalty,
        weight=10.0,
        params={"penalty_per_step": -0.001}
    )

@configclass
class TerminationsCfg:
    progress_state_tick = DoneTerm(
        func=terminations_user.progress_state_tick,
        time_out=False,
    )

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    target_too_far = DoneTerm(
        func=terminations_user.target_distance_exceeded,
        params={"max_target_distance": 5.5},
    )

    local_goal_reached = DoneTerm(
        func=terminations_user.local_goal_reached,
        time_out=False,
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
        self.episode_length_s = 200.0
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
        # Enable obstacle-rich training to produce stuck / near-stuck situations.
        self.events.reset_local_nav.params["disable_obstacles"] = False
        self.events.reset_local_nav.params["obstacle_path_clearance"] = 0.4
        self.events.reset_local_nav.params["obstacle_spacing"] = 0.45
        self.events.reset_local_nav.params["obstacle_start_clearance"] = 0.8
        self.events.reset_local_nav.params["obstacle_goal_clearance"] = 0.8
        self.terminations.target_too_far.params["max_target_distance"] = 5.5


class MyCarSimpleEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Minimal straight-line task for validating Dreamer training flow."""

    def __post_init__(self):
        super().__post_init__()
        self.sim.device = "cuda:0"
        self.episode_length_s = 12.0

        self.events.reset_local_nav.params["fixed_path_id"] = 0
        self.events.reset_local_nav.params["disable_obstacles"] = True
        self.events.reset_local_nav.params["lateral_offset_range"] = (-0.05, 0.05)
        self.events.reset_local_nav.params["heading_offset_range"] = (-0.05, 0.05)
        self.events.reset_local_nav.params["start_speed_range"] = (0.0, 0.05)
        self.events.reset_local_nav.params["waypoint_reach_thresh"] = 0.35

        # self.rewards.heading_align.weight = 2.5
        self.rewards.waypoint_reached.weight = 2.0
        self.rewards.waypoint_reached.params["bonus"] = 1.0
        self.rewards.speed_tracking.weight = 1.5
        self.rewards.time_penalty.params["penalty_per_step"] = -0.002

        self.terminations.target_too_far.params["max_target_distance"] = 2.5

class MyCarRoughEnvCfg_PLAY(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 58
        self.scene.env_spacing = 0


class MyCarSimpleEnvCfg_PLAY(MyCarSimpleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 0
