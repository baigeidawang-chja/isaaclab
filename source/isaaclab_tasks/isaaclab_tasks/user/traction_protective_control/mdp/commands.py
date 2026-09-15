from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
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


@configclass
class PlannerCommandCfg(CommandTermCfg):
    """Configuration for planner speed and heading commands."""

    class_type: type[CommandTerm] = PlannerCommand
    asset_name: str = MISSING
    desired_speed_range: tuple[float, float] = (0.3, 1.2)
    desired_heading_range: tuple[float, float] = (-0.4, 0.4)
