"""Variable-heading extension of the multi-level speed CommandFollow task."""

from isaaclab.utils import configclass

from ...mdp import MultiStepSpeedHeadingCommandCfg
from .command_follow_env_cfg import CommandFollowEnvCfg
from .multi_step_speed_env_cfg import SPEED_LEVELS, STEP_DURATION_S


HEADING_OFFSETS = (0.0, 0.2, -0.2, 0.4, -0.4, 0.0)
BASE_HEADING_RANGE = (-0.2, 0.2)


@configclass
class MultiStepSpeedHeadingCommandsCfg:
    """Synchronized speed and heading profiles for one complete episode."""

    planner_command = MultiStepSpeedHeadingCommandCfg(
        asset_name="robot",
        resampling_time_range=(STEP_DURATION_S, STEP_DURATION_S),
        desired_speed_range=(min(SPEED_LEVELS), max(SPEED_LEVELS)),
        desired_heading_range=BASE_HEADING_RANGE,
        speed_levels=SPEED_LEVELS,
        heading_offsets=HEADING_OFFSETS,
        debug_vis=False,
    )


@configclass
class MultiStepSpeedHeadingEnvCfg(CommandFollowEnvCfg):
    """CommandFollow with synchronized speed and world-heading steps."""

    commands: MultiStepSpeedHeadingCommandsCfg = MultiStepSpeedHeadingCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = len(SPEED_LEVELS) * STEP_DURATION_S


@configclass
class MultiStepSpeedHeadingEnvCfg_PLAY(MultiStepSpeedHeadingEnvCfg):
    """Deterministic visualization configuration for speed and heading steps."""

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
