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
from isaaclab.sensors import ImuCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets import CAR_CFG

from isaaclab_tasks.manager_based.navigation.mdp.actions import CarVWActionCfg
from ...mdp import FailureAwareCommandCfg
from ...mdp import events, observations, rewards, terminations


WHEEL_JOINTS = [
    "joint_front_left_wheel_link_wheel",
    "joint_front_right_wheel_link_wheel",
    "joint_back_left_wheel_link_wheel",
    "joint_back_right_wheel_link_wheel",
]


@configclass
class FailureAwareSceneCfg(InteractiveSceneCfg):
    """Minimal scene: plane, Car4WD, curb placeholder, and low-mu marker."""

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
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.47, 0.42)),
    )

    robot: ArticulationCfg = CAR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")  # type: ignore[attr-defined]

    imu = ImuCfg(prim_path="{ENV_REGEX_NS}/Robot/Car/base_link", debug_vis=False)

    obstacles = RigidObjectCollectionCfg(
        rigid_objects={
            "curb": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/curb",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(1.2, 0.0, 0.025)),
                spawn=sim_utils.CuboidCfg(
                    size=(0.08, 1.8, 0.05),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.35, 0.34, 0.32),
                        roughness=0.8,
                        metallic=0.0,
                    ),
                ),
            ),
            "low_traction_region": RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/low_traction_region",
                init_state=RigidObjectCfg.InitialStateCfg(pos=(2.4, 0.0, 0.001)),
                spawn=sim_utils.CuboidCfg(
                    size=(1.0, 1.8, 0.002),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.10, 0.22, 0.26),
                        opacity=0.35,
                        roughness=0.95,
                        metallic=0.0,
                    ),
                ),
            ),
        }
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg:
    planner_command = FailureAwareCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        v_plan_range=(0.45, 1.20),
        heading_plan_range=(-0.18, 0.18),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    throttle_steer = CarVWActionCfg(
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
        scale=(1.5, 0.8),
        offset=(0.0, 0.0),
        bounding_strategy="clip",
        asset_name="robot",
        no_reverse=False,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        planner_command = ObsTerm(func=observations.planner_command)
        base_lin_vel = ObsTerm(func=observations.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=observations.base_ang_vel, noise=Unoise(n_min=-0.08, n_max=0.08))
        projected_gravity = ObsTerm(func=observations.projected_gravity, noise=Unoise(n_min=-0.03, n_max=0.03))
        wheel_vel = ObsTerm(
            func=observations.wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
        )
        imu_state = ObsTerm(func=observations.imu_state)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class LabelsCfg(ObsGroup):
        stuck_label = ObsTerm(func=observations.stuck_label)
        slip_label = ObsTerm(func=observations.slip_label)
        front_traction_label = ObsTerm(func=observations.front_traction_label)
        rear_traction_label = ObsTerm(func=observations.rear_traction_label)
        abort_required_label = ObsTerm(func=observations.abort_required_label)
        continue_feasible_label = ObsTerm(func=observations.continue_feasible_label)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    labels: LabelsCfg = LabelsCfg()


@configclass
class EventCfg:
    reset_task = EventTerm(
        func=events.reset_failure_aware_task,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_asset_cfg": SceneEntityCfg("obstacles"),
            "start_x_range": (-0.05, 0.05),
            "start_y_range": (-0.08, 0.08),
            "heading_range": (-0.08, 0.08),
            "root_z": 0.18,
            "curb_x_range": (1.10, 1.45),
            "curb_height_range": (0.025, 0.07),
            "low_mu_x_range": (2.00, 2.50),
            "low_mu_length_range": (0.8, 1.4),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    progress_along_plan = RewTerm(func=rewards.progress_along_plan, weight=1.5, params={"scale": 1.0})
    heading_following = RewTerm(func=rewards.heading_following, weight=0.8, params={"std": 0.45})
    no_progress_penalty = RewTerm(func=rewards.no_progress_penalty, weight=0.2, params={"threshold": 0.03})
    slip_penalty = RewTerm(func=rewards.slip_penalty, weight=1.0, params={"scale": 0.03})
    stuck_penalty = RewTerm(func=rewards.stuck_penalty, weight=0.5, params={"scale": 1.0})
    action_rate_penalty = RewTerm(func=rewards.action_rate_penalty, weight=1.0, params={"scale": 0.01})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(func=terminations.success_distance, time_out=False, params={"distance": 4.0})
    flip = DoneTerm(
        func=terminations.is_flipped,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "up_z_threshold": 0.25},
    )
    no_progress_failure = DoneTerm(
        func=terminations.no_progress_failure,
        time_out=False,
        params={"min_progress_speed": 0.005, "max_no_progress_steps": 220, "min_episode_steps": 60},
    )


@configclass
class FailureAwareExecutionEnvCfg(ManagerBasedRLEnvCfg):
    """Failure-aware execution layer skeleton for Car4WD."""

    scene: FailureAwareSceneCfg = FailureAwareSceneCfg(num_envs=1, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 12.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = "cuda:0"


@configclass
class FailureAwareExecutionEnvCfg_PLAY(FailureAwareExecutionEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
