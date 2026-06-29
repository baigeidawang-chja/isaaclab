# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG, POSITION_GOAL_MARKER_CFG, CUBOID_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg

from .null_command import NullCommand
from .pose_2d_command import TerrainBasedPose2dCommand, UniformPose2dCommand, SequentialWaypointCommand
from .pose_command import UniformPoseCommand
from .velocity_command import NormalVelocityCommand, UniformVelocityCommand

from isaaclab.sim import PreviewSurfaceCfg



@configclass
class NullCommandCfg(CommandTermCfg):
    """Configuration for the null command generator."""

    class_type: type = NullCommand

    def __post_init__(self):
        """Post initialization."""
        # set the resampling time range to infinity to avoid resampling
        self.resampling_time_range = (math.inf, math.inf)


@configclass
class UniformVelocityCommandCfg(CommandTermCfg):
    """Configuration for the uniform velocity command generator."""

    class_type: type = UniformVelocityCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    heading_command: bool = False
    """Whether to use heading command or angular velocity command. Defaults to False.

    If True, the angular velocity command is computed from the heading error, where the
    target heading is sampled uniformly from provided range. Otherwise, the angular velocity
    command is sampled uniformly from provided range.
    """

    heading_control_stiffness: float = 1.0
    """Scale factor to convert the heading error to angular velocity command. Defaults to 1.0."""

    rel_standing_envs: float = 0.0
    """The sampled probability of environments that should be standing still. Defaults to 0.0."""

    rel_heading_envs: float = 1.0
    """The sampled probability of environments where the robots follow the heading-based angular velocity command
    (the others follow the sampled angular velocity command). Defaults to 1.0.

    This parameter is only used if :attr:`heading_command` is True.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the velocity commands."""

        lin_vel_x: tuple[float, float] = MISSING
        """Range for the linear-x velocity command (in m/s)."""

        lin_vel_y: tuple[float, float] = MISSING
        """Range for the linear-y velocity command (in m/s)."""

        ang_vel_z: tuple[float, float] = MISSING
        """Range for the angular-z velocity command (in rad/s)."""

        heading: tuple[float, float] | None = None
        """Range for the heading command (in rad). Defaults to None.

        This parameter is only used if :attr:`~UniformVelocityCommandCfg.heading_command` is True.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    """The configuration for the goal velocity visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    """The configuration for the current velocity visualization marker. Defaults to BLUE_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.5, 0.5, 0.5)
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class NormalVelocityCommandCfg(UniformVelocityCommandCfg):
    """Configuration for the normal velocity command generator."""

    class_type: type = NormalVelocityCommand
    heading_command: bool = False  # --> we don't use heading command for normal velocity command.

    @configclass
    class Ranges:
        """Normal distribution ranges for the velocity commands."""

        mean_vel: tuple[float, float, float] = MISSING
        """Mean velocity for the normal distribution (in m/s).

        The tuple contains the mean linear-x, linear-y, and angular-z velocity.
        """

        std_vel: tuple[float, float, float] = MISSING
        """Standard deviation for the normal distribution (in m/s).

        The tuple contains the standard deviation linear-x, linear-y, and angular-z velocity.
        """

        zero_prob: tuple[float, float, float] = MISSING
        """Probability of zero velocity for the normal distribution.

        The tuple contains the probability of zero linear-x, linear-y, and angular-z velocity.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the velocity commands."""


@configclass
class UniformPoseCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = UniformPoseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    body_name: str = MISSING
    """Name of the body in the asset for which the commands are generated."""

    make_quat_unique: bool = False
    """Whether to make the quaternion unique or not. Defaults to False.

    If True, the quaternion is made unique by ensuring the real part is positive.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the pose commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        pos_z: tuple[float, float] = MISSING
        """Range for the z position (in m)."""

        roll: tuple[float, float] = MISSING
        """Range for the roll angle (in rad)."""

        pitch: tuple[float, float] = MISSING
        """Range for the pitch angle (in rad)."""

        yaw: tuple[float, float] = MISSING
        """Range for the yaw angle (in rad)."""

    ranges: Ranges = MISSING
    """Ranges for the commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/goal_pose")
    """The configuration for the goal pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    current_pose_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(
        prim_path="/Visuals/Command/body_pose"
    )
    """The configuration for the current pose visualization marker. Defaults to FRAME_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.1, 0.1, 0.1)
    goal_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    current_pose_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)


@configclass
class UniformPose2dCommandCfg(CommandTermCfg):
    """Configuration for the uniform 2D-pose command generator."""

    class_type: type = UniformPose2dCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    simple_heading: bool = MISSING
    """Whether to use simple heading or not.

    If True, the heading is in the direction of the target position.
    """

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        pos_x: tuple[float, float] = MISSING
        """Range for the x position (in m)."""

        pos_y: tuple[float, float] = MISSING
        """Range for the y position (in m)."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the position commands."""

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""

    # Set the scale of the visualization markers to (0.2, 0.2, 0.8)
    goal_pose_visualizer_cfg.markers["sphere"].radius = 0.1


@configclass
class TerrainBasedPose2dCommandCfg(UniformPose2dCommandCfg):
    """Configuration for the terrain-based position command generator."""

    class_type = TerrainBasedPose2dCommand

    @configclass
    class Ranges:
        """Uniform distribution ranges for the position commands."""

        heading: tuple[float, float] = MISSING
        """Heading range for the position commands (in rad).

        Used only if :attr:`simple_heading` is False.
        """

    ranges: Ranges = MISSING
    """Distribution ranges for the sampled commands."""

@configclass
class SequentialWaypointCommandCfg(CommandTermCfg):
    
    class_type: type = SequentialWaypointCommand

    asset_name: str = "robot"

    debug_vis: bool = True

    waypoints: list = MISSING

    success_threshold: float = 0.5

    cyclic: bool = True

    simple_heading: bool = MISSING

    # goal_pose_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
    #     prim_path="/Visuals/Command/pose_goal"
    # )
    # 1. 为“当前”目标定义的绿色画笔
    current_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/CurrentWaypoint", # 必须是唯一路径
        markers={ "sphere": UsdFileCfg(
            usd_path=f"/media/chja/CE54D158C95990271/Assets/Isaac/4.5/Isaac/Props/Shapes/sphere.usd",
            scale=(0.35, 0.35, 0.35),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
        )}
    )
    # 2. 为“已访问”航点定义的蓝色画笔
    visited_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/VisitedWaypoints", # 必须是唯一路径
        markers={ "sphere": UsdFileCfg(
            usd_path=f"/media/chja/CE54D158C95990271/Assets/Isaac/4.5/Isaac/Props/Shapes/sphere.usd",
            scale=(0.3, 0.3, 0.3),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9))
        )}
    )
    # 3. 为“未访问”航点定义的灰色画笔
    unvisited_marker_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/UnvisitedWaypoints", # 必须是唯一路径
        markers={ "sphere": UsdFileCfg(
            usd_path=f"/media/chja/CE54D158C95990271/Assets/Isaac/4.5/Isaac/Props/Shapes/sphere.usd",
            scale=(0.3, 0.3, 0.3),
            visual_material=PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4))
        )}
    )

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/World/Visuals/pose"
    )
    """The configuration for the goal pose visualization marker. Defaults to GREEN_ARROW_X_MARKER_CFG."""
