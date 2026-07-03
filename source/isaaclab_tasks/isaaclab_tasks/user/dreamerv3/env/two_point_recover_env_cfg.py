# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from ..mdp import rewards, terminations_user
from ..mdp.events import reset_local_nav_task
from .car_env_cfg import ActionsCfg, CommandsCfg, MySceneCfg, ObservationsCfg


@configclass
class TwoPointRecoverEventCfg:
    """Reset into a short start-goal recovery task with controlled obstacle templates."""

    reset_local_nav = EventTerm(
        func=reset_local_nav_task,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "obstacle_asset_cfg": SceneEntityCfg("obstacles"),
            "path_mode": "two_point",
            "goal_distance_range": (1.5, 3.0),
            "goal_lateral_range": (-0.3, 0.3),
            "disable_obstacles": False,
            "obstacle_mode": "recover_template",
            "recover_template_set": "two_point",
            "recover_obstacle_distance_range": (0.45, 0.8),
            "lateral_offset_range": (-0.05, 0.05),
            "heading_offset_range": (-0.1, 0.1),
            "start_speed_range": (0.0, 0.08),
            "waypoint_reach_thresh": 0.30,
            "contact_memory_num_sectors": 8,
            "contact_memory_decay": 0.995,
            "debug_vis": True,
            "debug_vis_num_points": 8,
            "debug_vis_ds": 0.25,
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
class TwoPointRecoverRewardsCfg:
    """Rewards focused on local recovery instead of long-horizon waypoint tracking."""

    progress_to_goal = RewTerm(
        func=rewards.local_forward_progress_reward,
        weight=1.4,
        params={"scale": 8.0, "max_delta": 0.10},
    )

    recovery_progress = RewTerm(
        func=rewards.local_recovery_progress_bonus,
        weight=1.0,
        params={"stuck_steps": 10, "scale": 2.0},
    )

    recover_success = RewTerm(
        func=rewards.local_escape_contact_bonus,
        weight=0.8,
        params={"min_speed": 0.08, "stuck_steps": 10, "scale": 1.5},
    )

    goal_success = RewTerm(
        func=rewards.local_success_reward,
        weight=0.8,
        params={"scale": 3.0},
    )

    heading_align = RewTerm(
        func=rewards.local_target_heading_alignment_reward,
        weight=0.04,
        params={
            "std": 0.55,
            "directional": True,
            "reverse_floor": 0.25,
            "reverse_speed_thresh": 0.04,
        },
    )

    weak_speed_prior = RewTerm(
        func=rewards.local_speed_tracking_reward,
        weight=0.05,
        params={"target_speed": 0.25, "std": 0.45},
    )

    stuck_duration = RewTerm(
        func=rewards.local_stuck_penalty,
        weight=1.0,
        params={"patience": 12, "scale": 0.25},
    )

    action_oscillation = RewTerm(
        func=rewards.local_action_smoothness_penalty,
        weight=1.0,
        params={"scale": 0.01},
    )

    time_penalty = RewTerm(
        func=rewards.time_penalty,
        weight=1.0,
        params={"penalty_per_step": -0.002},
    )


@configclass
class TwoPointRecoverTerminationsCfg:
    """Short local-recovery terminations."""

    progress_state_tick = DoneTerm(
        func=terminations_user.progress_state_tick,
        time_out=False,
    )

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    target_too_far = DoneTerm(
        func=terminations_user.target_distance_exceeded,
        params={"max_target_distance": 4.0},
    )

    local_goal_reached = DoneTerm(
        func=terminations_user.local_goal_reached,
        time_out=False,
    )


@configclass
class MyCarTwoPointRecoverEnvCfg(ManagerBasedRLEnvCfg):
    """Two-point short-distance local stuck-recovery training environment."""

    scene: MySceneCfg = MySceneCfg(num_envs=1, env_spacing=8)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: TwoPointRecoverRewardsCfg = TwoPointRecoverRewardsCfg()
    terminations: TwoPointRecoverTerminationsCfg = TwoPointRecoverTerminationsCfg()
    events: TwoPointRecoverEventCfg = TwoPointRecoverEventCfg()

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 12.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.device = "cuda:0"
        self.observations.policy.enable_corruption = True


class MyCarTwoPointRecoverEnvCfg_PLAY(MyCarTwoPointRecoverEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 0
