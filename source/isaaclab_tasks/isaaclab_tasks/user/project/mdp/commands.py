from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi


class FailureAwareCommand(CommandTerm):
    """Nominal planner command: planned speed and world-frame heading."""

    cfg: "FailureAwareCommandCfg"

    def __init__(self, cfg: "FailureAwareCommandCfg", env):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["heading_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["speed_error"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return [v_plan, heading_plan]."""
        return self._command

    @property
    def v_plan(self) -> torch.Tensor:
        return self._command[:, 0]

    @property
    def heading_plan(self) -> torch.Tensor:
        return self._command[:, 1]

    @property
    def heading_error(self) -> torch.Tensor:
        return wrap_to_pi(self.heading_plan - self.robot.data.heading_w)

    def _update_metrics(self):
        max_command_time = max(float(self.cfg.resampling_time_range[1]), 1.0e-6)
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["heading_error"] += torch.abs(self.heading_error) / max_command_step
        self.metrics["speed_error"] += torch.abs(self.v_plan - self.robot.data.root_lin_vel_b[:, 0]) / max_command_step

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        r = torch.empty(len(env_ids), device=self.device)
        self._command[env_ids, 0] = r.uniform_(*self.cfg.v_plan_range)
        self._command[env_ids, 1] = r.uniform_(*self.cfg.heading_plan_range)

    def _update_command(self):
        pass


@configclass
class FailureAwareCommandCfg(CommandTermCfg):
    """Configuration for nominal planner commands."""

    class_type: type[CommandTerm] = FailureAwareCommand
    asset_name: str = MISSING
    v_plan_range: tuple[float, float] = (0.4, 1.2)
    heading_plan_range: tuple[float, float] = (-0.15, 0.15)

