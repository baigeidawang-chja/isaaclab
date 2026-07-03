from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets import CAR_CFG

from ..mdp import observation, recovery_rewards, recovery_terminations
from ..mdp.actions import RecoveryPrimitiveActionCfg
from ..mdp.events import reset_blocked_recovery_task


@configclass
class BlockedRecoverySceneCfg(InteractiveSceneCfg):
    """Minimal blocked-recovery scene: plane, car, and two recovery obstacles."""

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
        visual_material=sim_utils.PreviewSurfaceCfg(),
    )

    robot: ArticulationCfg = CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore[attr-defined]

    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/Car/base_link", debug_vis=False)

    robot_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Car/.*wheel",
        history_length=1,
        track_air_time=False,
        update_period=0.0,
        debug_vis=False,
    )

    obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            "object_0": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/object_0",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -10.0)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.03, 2.5, 0.03),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.55, 0.42, 0.26),
                        roughness=0.6,
                        metallic=0.0,
                    ),
                ),
            ),
            "object_1": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/object_1",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, -10.0)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.3, 0.25, 0.10),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35, 0.35, 0.35),
                        roughness=0.8,
                        metallic=0.0,
                    ),
                ),
            ),
        }
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


@configclass
class BlockedRecoveryActionsCfg:
    """Box(6) recovery primitive selector."""

    recovery_primitive = RecoveryPrimitiveActionCfg(
        wheel_joint_names=[
            "joint_front_right_wheel_link_wheel",
            "joint_front_left_wheel_link_wheel",
            "joint_back_right_wheel_link_wheel",
            "joint_back_left_wheel_link_wheel",
        ],
        steering_joint_names=[
            "joint_front_right_steer",
            "joint_front_left_steer",
        ],
        base_length=2.035 / 5,
        base_width=1.1673 / 5,
        wheel_radius=0.035,
        asset_name="robot",
        no_reverse=False,
        min_steps_between_switch=1,
        max_speed=1.55,
        max_steer=0.45,
        max_speed_rate=1.0,
        max_steer_rate=0.8,
        continue_speed=0.45,
        slow_speed=0.18,
        reverse_speed=0.35,
        escape_speed=0.25,
        escape_steer=0.42,
    )


@configclass
class BlockedRecoveryObservationsCfg:
    """Proprioceptive policy observations only."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=observation.get_base_lin_vel_safe,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel = ObsTerm(
            func=observation.get_base_ang_vel_safe,
            noise=Unoise(n_min=-0.08, n_max=0.08),
        )
        projected_gravity = ObsTerm(
            func=observation.get_projected_gravity_safe,
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        joint_vel = ObsTerm(
            func=observation.get_joint_vel_safe,
            params={
                "asset_cfg": SceneEntityCfg(
                    name="robot",
                    joint_names=[
                        "joint_front_left_wheel_link_wheel",
                        "joint_front_right_wheel_link_wheel",
                        "joint_back_left_wheel_link_wheel",
                        "joint_back_right_wheel_link_wheel",
                    ],
                )
            },
        )
        actions = ObsTerm(func=mdp.last_action)
        imu_state = ObsTerm(func=observation.get_imu_state)
        wheel_slip_proxy = ObsTerm(func=observation.get_wheel_slip_proxy)
        wheel_current_proxy = ObsTerm(func=observation.get_wheel_current_torque_proxy)
        front_rear_slip_diff = ObsTerm(func=observation.get_front_rear_slip_difference)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class BlockedRecoveryEventCfg:
    reset_blocked_recovery = EventTerm(
        func=reset_blocked_recovery_task,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_asset_cfg": SceneEntityCfg("obstacles"),
            "scenario_weights": (1.0, 0.0, 0.0),
            "start_x_range": (-0.08, 0.08),
            "start_y_range": (-0.05, 0.05),
            "heading_range": (-0.04, 0.04),
            "start_speed_range": (0.0, 0.03),
            "root_z": 0.18,
            "curb_x_range": (0.0, 1.30),
            "belly_x_range": (0.30, 0.45),
            "success_distance": 1.15,
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
class BlockedRecoveryRewardsCfg:
    effective_displacement = RewTerm(
        func=recovery_rewards.effective_displacement,
        weight=1.0,
        params={"scale": 8.0, "max_delta": 0.08},
    )
    success_bonus = RewTerm(
        func=recovery_rewards.success_bonus,
        weight=1.0,
        params={"scale": 4.0, "success_distance": 1.15},
    )
    time_penalty = RewTerm(
        func=recovery_rewards.time_penalty,
        weight=1.0,
        params={"penalty_per_step": -0.003},
    )
    wheel_spin_penalty = RewTerm(
        func=recovery_rewards.wheel_spin_penalty,
        weight=1.0,
        params={"scale": 0.08},
    )
    slip_penalty = RewTerm(
        func=recovery_rewards.slip_penalty,
        weight=1.0,
        params={"scale": 0.015},
    )
    torque_proxy_penalty = RewTerm(
        func=recovery_rewards.torque_proxy_penalty,
        weight=1.0,
        params={"scale": 0.001},
    )
    action_switch_penalty = RewTerm(
        func=recovery_rewards.action_switch_penalty,
        weight=1.0,
        params={"scale": 0.04},
    )
    invalid_action_penalty = RewTerm(
        func=recovery_rewards.invalid_action_penalty,
        weight=1.0,
        params={"scale": 0.08},
    )
    retreat_penalty = RewTerm(
        func=recovery_rewards.retreat_penalty,
        weight=1.0,
        params={"scale": 0.2, "allowed_retreat": 0.35},
    )


@configclass
class BlockedRecoveryTerminationsCfg:
    success = DoneTerm(
        func=recovery_terminations.blocked_recovery_success,
        time_out=False,
        params={"success_distance": 1.15},
    )
    failure = DoneTerm(
        func=recovery_terminations.blocked_recovery_failure,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_retreat": 0.65,
            "max_no_progress_steps": 120,
            "min_episode_steps": 20,
            "flip_up_z_threshold": 0.2,
        },
    )
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CommandsCfg:
    """BlockedRecovery does not use command terms."""


@configclass
class MyCarBlockedRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """First-stage blocked recovery task: curb momentum loss only."""

    scene: BlockedRecoverySceneCfg = BlockedRecoverySceneCfg(num_envs=1, env_spacing=5.0)
    observations: BlockedRecoveryObservationsCfg = BlockedRecoveryObservationsCfg()
    actions: BlockedRecoveryActionsCfg = BlockedRecoveryActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: BlockedRecoveryRewardsCfg = BlockedRecoveryRewardsCfg()
    terminations: BlockedRecoveryTerminationsCfg = BlockedRecoveryTerminationsCfg()
    events: BlockedRecoveryEventCfg = BlockedRecoveryEventCfg()

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 10.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = "cuda:0"
        self.observations.policy.enable_corruption = True


@configclass
class MyCarBlockedRecoveryEnvCfg_PLAY(MyCarBlockedRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.0
