from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
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

from ...mdp import FailureAwareCommandCfg
from ...mdp import events, observations, rewards, terminations
from ...mdp.actions import RateLimitedCarVWActionCfg


WHEEL_JOINTS = [
    "joint_front_left_wheel_link_wheel",
    "joint_front_right_wheel_link_wheel",
    "joint_back_left_wheel_link_wheel",
    "joint_back_right_wheel_link_wheel",
]


@configclass
class CommandFollowingSceneCfg(InteractiveSceneCfg):
    """Flat command-following scene for the Car4WD robot."""

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

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.75, 0.75, 0.75)),
    )


@configclass
class CommandsCfg:
    """Nominal planner command. First version fixes heading to +x."""

    planner_command = FailureAwareCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 8.0),
        v_plan_range=(0.35, 1.20),
        heading_plan_range=(0.0, 0.0),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    """Car action interface: forward velocity command and steering command."""

    throttle_steer = RateLimitedCarVWActionCfg(
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
        max_v_rate=2.0,
        max_steer_rate=2.0,
        reset_to_zero=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        planner_command = ObsTerm(func=observations.planner_command)
        base_lin_vel = ObsTerm(func=observations.base_lin_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        base_ang_vel = ObsTerm(func=observations.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        wheel_vel = ObsTerm(
            func=observations.wheel_vel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINTS)},
        )
        projected_gravity = ObsTerm(func=observations.projected_gravity, noise=Unoise(n_min=-0.02, n_max=0.02))
        imu_state = ObsTerm(func=observations.imu_state)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.18, 0.18),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )

    reset_runtime_buffers = EventTerm(
        func=events.reset_runtime_buffers,
        mode="reset",
        params={},
    )


@configclass
class RewardsCfg:
    speed_tracking = RewTerm(
        func=rewards.plan_speed_tracking,
        weight=2.0,
        params={"command_name": "planner_command", "std": 0.35},
    )
    heading_tracking = RewTerm(
        func=rewards.heading_tracking,
        weight=1.0,
        params={"command_name": "planner_command", "std": 0.35},
    )
    yaw_rate = RewTerm(func=rewards.yaw_rate_penalty, weight=0.1, params={"scale": 0.25})
    lateral_velocity = RewTerm(func=rewards.lateral_velocity_penalty, weight=0.1, params={"scale": 0.5})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    flip = DoneTerm(
        func=terminations.is_flipped,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "up_z_threshold": 0.25},
    )
    out_of_bounds = DoneTerm(
        func=terminations.out_of_bounds,
        time_out=False,
        params={"x_min": -2.0, "x_max": 12.0, "y_min": -3.0, "y_max": 3.0},
    )


@configclass
class CommandFollowingEnvCfg(ManagerBasedRLEnvCfg):
    """First-version flat speed and heading command-following task."""

    scene: CommandFollowingSceneCfg = CommandFollowingSceneCfg(num_envs=1, env_spacing=3.0)
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
class CommandFollowingEnvCfg_PLAY(CommandFollowingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
