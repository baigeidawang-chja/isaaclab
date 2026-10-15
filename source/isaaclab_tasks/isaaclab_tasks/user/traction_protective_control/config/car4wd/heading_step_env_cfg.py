"""Constant-speed, heading-step isolation task for Phase 1 CommandFollow."""

from isaaclab.utils import configclass

from ...mdp import MultiStepSpeedHeadingCommandCfg
from .command_follow_env_cfg import CommandFollowEnvCfg


CONSTANT_SPEED = 0.8
HEADING_OFFSETS = (0.0, 0.2, -0.2, 0.4, -0.4, 0.0)
SPEED_LEVELS = (CONSTANT_SPEED,) * len(HEADING_OFFSETS)
STEP_DURATION_S = 5.0
BASE_HEADING_RANGE = (-0.2, 0.2)


@configclass
class HeadingStepCommandsCfg:
    """Constant speed with staged world-heading offsets."""

    planner_command = MultiStepSpeedHeadingCommandCfg(
        asset_name="robot",
        resampling_time_range=(STEP_DURATION_S, STEP_DURATION_S),
        desired_speed_range=(CONSTANT_SPEED, CONSTANT_SPEED),
        desired_heading_range=BASE_HEADING_RANGE,
        speed_levels=SPEED_LEVELS,
        heading_offsets=HEADING_OFFSETS,
        debug_vis=False,
    )


@configclass
class HeadingStepEnvCfg(CommandFollowEnvCfg):
    """Isolate steering dynamics with constant speed and heading steps."""

    commands: HeadingStepCommandsCfg = HeadingStepCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = len(HEADING_OFFSETS) * STEP_DURATION_S


@configclass
class HeadingStepEnvCfg_PLAY(HeadingStepEnvCfg):
    """Deterministic visualization configuration for heading-step evaluation."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        self.commands.planner_command.desired_heading_range = (0.0, 0.0)
        self.commands.planner_command.debug_vis = True
        self.observations.policy.enable_corruption = False
        self.terminations.out_of_bounds.params = {
            "x_min": -2000.0,
            "x_max": 2000.0,
            "y_min": -1000.0,
            "y_max": 1000.0,
        }
