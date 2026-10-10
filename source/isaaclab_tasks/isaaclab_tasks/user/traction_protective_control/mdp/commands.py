from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi


class PlannerCommand(CommandTerm):
    """Upper-level planner command: desired speed and world-frame heading."""

    cfg: "PlannerCommandCfg"

    def __init__(self, cfg: "PlannerCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["speed_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return [desired_speed, desired_heading]."""
        return self._command

    @property
    def desired_speed(self) -> torch.Tensor:
        return self._command[:, 0]

    @property
    def desired_heading(self) -> torch.Tensor:
        return self._command[:, 1]

    @property
    def heading_error(self) -> torch.Tensor:
        return wrap_to_pi(self.desired_heading - self.robot.data.heading_w)

    def _update_metrics(self):
        max_command_time = max(float(self.cfg.resampling_time_range[1]), 1.0e-6)
        max_command_step = max_command_time / self._env.step_dt
        plan_dir = torch.stack([torch.cos(self.desired_heading), torch.sin(self.desired_heading)], dim=-1)
        speed_along_plan = torch.sum(self.robot.data.root_lin_vel_w[:, :2] * plan_dir, dim=-1)
        self.metrics["heading_error"] += torch.abs(self.heading_error) / max_command_step
        self.metrics["speed_error"] += torch.abs(self.desired_speed - speed_along_plan) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        random_values = torch.empty(len(env_ids), device=self.device)
        self._command[env_ids, 0] = random_values.uniform_(*self.cfg.desired_speed_range)
        self._command[env_ids, 1] = random_values.uniform_(*self.cfg.desired_heading_range)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        if not self._env.sim.has_gui():
            return

        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.target_visualizer_cfg)
                self.actual_visualizer = VisualizationMarkers(self.cfg.actual_visualizer_cfg)
            self.target_visualizer.set_visibility(True)
            self.actual_visualizer.set_visibility(True)
            self._set_telemetry_window_visible(True)
        elif hasattr(self, "target_visualizer"):
            self.target_visualizer.set_visibility(False)
            self.actual_visualizer.set_visibility(False)
            self._set_telemetry_window_visible(False)

    def _debug_vis_callback(self, event):
        if not self._env.sim.has_gui() or not self.robot.is_initialized:
            return

        target_pos_w = self.robot.data.root_pos_w.clone()
        actual_pos_w = target_pos_w.clone()
        target_pos_w[:, 2] += self.cfg.target_arrow_height
        actual_pos_w[:, 2] += self.cfg.actual_arrow_height

        actual_speed = self.robot.data.root_lin_vel_b[:, 0]
        target_quat = self._heading_to_quaternion(self.desired_heading)
        actual_quat = self._heading_to_quaternion(self.robot.data.heading_w)
        target_scale = self._speed_to_arrow_scale(self.desired_speed, self.target_visualizer)
        actual_scale = self._speed_to_arrow_scale(actual_speed, self.actual_visualizer)

        self.target_visualizer.visualize(target_pos_w, target_quat, target_scale)
        self.actual_visualizer.visualize(actual_pos_w, actual_quat, actual_scale)
        self._update_telemetry_labels(actual_speed)

    def _heading_to_quaternion(self, heading: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(heading)
        return math_utils.quat_from_euler_xyz(zeros, zeros, heading)

    def _speed_to_arrow_scale(
        self, speed: torch.Tensor, visualizer: VisualizationMarkers
    ) -> torch.Tensor:
        default_scale = visualizer.cfg.markers["arrow"].scale
        scale = torch.tensor(default_scale, device=self.device).repeat(speed.shape[0], 1)
        speed_scale = torch.clamp(torch.abs(speed) * self.cfg.speed_visualizer_scale, min=0.1)
        scale[:, 0] *= speed_scale
        return scale

    def _set_telemetry_window_visible(self, visible: bool):
        if not hasattr(self, "_telemetry_window"):
            if not visible:
                return
            import omni.ui as ui

            self._telemetry_window = ui.Window(
                "Command Follow Telemetry",
                width=310,
                height=180,
                visible=True,
                dock_preference=ui.DockPreference.RIGHT_TOP,
            )
            with self._telemetry_window.frame:
                with ui.VStack(spacing=6):
                    ui.Label("Environment 0", height=24)
                    self._target_speed_label = ui.Label("Target speed: -- m/s")
                    self._target_heading_label = ui.Label("Target heading: -- deg")
                    self._actual_speed_label = ui.Label("Actual speed: -- m/s")
                    self._actual_heading_label = ui.Label("Actual heading: -- deg")
        self._telemetry_window.visible = visible

    def _update_telemetry_labels(self, actual_speed: torch.Tensor):
        if not hasattr(self, "_telemetry_window") or not self._telemetry_window.visible:
            return
        values = torch.stack(
            (
                self.desired_speed[0],
                torch.rad2deg(self.desired_heading[0]),
                actual_speed[0],
                torch.rad2deg(self.robot.data.heading_w[0]),
            )
        ).tolist()
        target_speed_value, target_heading_value, actual_speed_value, actual_heading_value = values
        self._target_speed_label.text = f"Target speed (green): {target_speed_value:+.2f} m/s"
        self._target_heading_label.text = f"Target heading (green): {target_heading_value:+.1f} deg"
        self._actual_speed_label.text = f"Actual speed (blue): {actual_speed_value:+.2f} m/s"
        self._actual_heading_label.text = f"Actual heading (blue): {actual_heading_value:+.1f} deg"


class MultiStepSpeedCommand(PlannerCommand):
    """Fixed-heading planner command with an episode-long speed-step profile."""

    cfg: "MultiStepSpeedCommandCfg"

    def __init__(self, cfg: "MultiStepSpeedCommandCfg", env):
        if not cfg.speed_levels:
            raise ValueError("MultiStepSpeedCommandCfg.speed_levels must contain at least one level.")
        super().__init__(cfg, env)
        if cfg.resampling_time_range[0] != cfg.resampling_time_range[1]:
            raise ValueError("MultiStepSpeedCommand requires a fixed resampling interval.")
        self._speed_levels = torch.tensor(cfg.speed_levels, device=self.device, dtype=torch.float32)
        step_duration = float(cfg.resampling_time_range[0])
        self._steps_per_level = round(step_duration / float(env.step_dt))
        realized_duration = self._steps_per_level * float(env.step_dt)
        if self._steps_per_level < 1 or abs(step_duration - realized_duration) > 1.0e-6:
            raise ValueError("Speed-step duration must be an integer multiple of the environment step time.")
        self._steps_in_level = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._steps_in_level[env_ids] = 0
        self.time_left[env_ids] = self._steps_per_level * float(self._env.step_dt)
        return extras

    def compute(self, dt: float):
        self._update_metrics()
        self._steps_in_level += 1
        resample_env_ids = (self._steps_in_level >= self._steps_per_level).nonzero().flatten()
        if len(resample_env_ids) > 0:
            self._steps_in_level[resample_env_ids] = 0
            self.command_counter[resample_env_ids] += 1
            self._resample_command(resample_env_ids)
        self.time_left[:] = (self._steps_per_level - self._steps_in_level) * dt
        self._update_command()

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        stage_indices = torch.remainder(self.command_counter[env_ids] - 1, len(self.cfg.speed_levels))
        self._command[env_ids, 0] = self._speed_levels[stage_indices]

        new_episode_env_ids = env_ids[self.command_counter[env_ids] == 1]
        if len(new_episode_env_ids) > 0:
            headings = torch.empty(len(new_episode_env_ids), device=self.device)
            self._command[new_episode_env_ids, 1] = headings.uniform_(*self.cfg.desired_heading_range)


class MultiStepSpeedHeadingCommand(MultiStepSpeedCommand):
    """Planner command with synchronized speed and world-heading steps."""

    cfg: "MultiStepSpeedHeadingCommandCfg"

    def __init__(self, cfg: "MultiStepSpeedHeadingCommandCfg", env):
        if len(cfg.heading_offsets) != len(cfg.speed_levels):
            raise ValueError("heading_offsets and speed_levels must have the same length.")
        super().__init__(cfg, env)
        self._heading_offsets = torch.tensor(cfg.heading_offsets, device=self.device, dtype=torch.float32)
        self._base_heading = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        new_episode_mask = self.command_counter[env_ids] == 1
        super()._resample_command(env_ids)

        new_episode_env_ids = env_ids[new_episode_mask]
        if len(new_episode_env_ids) > 0:
            self._base_heading[new_episode_env_ids] = self._command[new_episode_env_ids, 1]

        stage_indices = torch.remainder(self.command_counter[env_ids] - 1, len(self.cfg.heading_offsets))
        self._command[env_ids, 1] = wrap_to_pi(
            self._base_heading[env_ids] + self._heading_offsets[stage_indices]
        )


@configclass
class PlannerCommandCfg(CommandTermCfg):
    """Configuration for planner speed and heading commands."""

    class_type: type[CommandTerm] = PlannerCommand
    asset_name: str = MISSING
    desired_speed_range: tuple[float, float] = (0.3, 1.2)
    desired_heading_range: tuple[float, float] = (-0.4, 0.4)
    speed_visualizer_scale: float = 2.0
    target_arrow_height: float = 0.65
    actual_arrow_height: float = 0.45
    target_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/CommandFollow/target"
    )
    actual_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/CommandFollow/actual"
    )

    target_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    actual_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)


@configclass
class MultiStepSpeedCommandCfg(PlannerCommandCfg):
    """Configuration for a discrete speed profile with one heading per episode."""

    class_type: type[CommandTerm] = MultiStepSpeedCommand
    speed_levels: tuple[float, ...] = (0.3, 0.6, 0.9, 1.2, 0.6, 0.3)
    """Speed levels visited in order, one level per command resampling interval."""


@configclass
class MultiStepSpeedHeadingCommandCfg(MultiStepSpeedCommandCfg):
    """Configuration for synchronized speed and heading step profiles."""

    class_type: type[CommandTerm] = MultiStepSpeedHeadingCommand
    heading_offsets: tuple[float, ...] = (0.0, 0.2, -0.2, 0.4, -0.4, 0.0)
    """World-heading offsets from the episode's sampled base heading, in radians."""
