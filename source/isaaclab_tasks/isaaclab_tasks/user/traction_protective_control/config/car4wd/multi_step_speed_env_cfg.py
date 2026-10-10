"""Multi-level speed-step variant of the Phase 1 CommandFollow task."""

from isaaclab.utils import configclass

from ...mdp import MultiStepSpeedCommandCfg
from .command_follow_env_cfg import CommandFollowEnvCfg


SPEED_LEVELS = (0.3, 0.6, 0.9, 1.2, 0.6, 0.3)
STEP_DURATION_S = 5.0


@configclass
class MultiStepSpeedCommandsCfg:
    """Planner command with a fixed episode heading and ordered speed steps."""

    planner_command = MultiStepSpeedCommandCfg(
        asset_name="robot",
        resampling_time_range=(STEP_DURATION_S, STEP_DURATION_S),
        desired_speed_range=(min(SPEED_LEVELS), max(SPEED_LEVELS)),
        desired_heading_range=(-0.4, 0.4),
        speed_levels=SPEED_LEVELS,
        debug_vis=False,
    )


@configclass
class MultiStepSpeedEnvCfg(CommandFollowEnvCfg):
    """CommandFollow with multiple speed steps and one fixed heading per episode."""

    commands: MultiStepSpeedCommandsCfg = MultiStepSpeedCommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = len(SPEED_LEVELS) * STEP_DURATION_S


@configclass
class MultiStepSpeedEnvCfg_PLAY(MultiStepSpeedEnvCfg):
    """Small deterministic visualization configuration for the speed-step task."""

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
