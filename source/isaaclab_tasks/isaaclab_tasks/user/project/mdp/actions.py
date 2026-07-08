from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass
from WheeledLab.source.wheeledlab.wheeledlab.envs.mdp.actions import AckermannAction, AckermannActionCfg


class RateLimitedCarVWAction(AckermannAction):
    """Ackermann [forward, steering] action with physical target rate limits."""

    cfg: "RateLimitedCarVWActionCfg"

    def __init__(self, cfg: "RateLimitedCarVWActionCfg", env):
        super().__init__(cfg, env)
        self._processed_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)
        self._previous_executed_action = torch.zeros_like(self._processed_actions)
        self._target_actions = torch.zeros_like(self._processed_actions)
        self._rate_limits = torch.tensor(
            [cfg.max_v_rate, cfg.max_steer_rate],
            device=self.device,
            dtype=torch.float32,
        )

    @property
    def target_actions(self) -> torch.Tensor:
        """Physical target actions after bounds/scale/offset, before rate limiting."""
        return self._target_actions

    @property
    def previous_executed_action(self) -> torch.Tensor:
        return self._previous_executed_action

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        if self._bounding_strategy == "clip":
            target = torch.clip(actions, min=-1.0, max=1.0) * self._scale + self._offset
        elif self._bounding_strategy == "tanh":
            target = torch.tanh(actions) * self._scale + self._offset
        else:
            target = actions * self._scale + self._offset

        if self.cfg.no_reverse:
            target[:, 0] = torch.clamp(target[:, 0], min=0.0)

        self._target_actions[:] = target

        step_dt = float(self._env.step_dt)
        max_delta = self._rate_limits * step_dt
        action_delta = torch.clamp(
            target - self._previous_executed_action,
            min=-max_delta,
            max=max_delta,
        )
        self._processed_actions[:] = self._previous_executed_action + action_delta
        self._previous_executed_action[:] = self._processed_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        if self.cfg.reset_to_zero:
            self._previous_executed_action[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._target_actions[env_ids] = 0.0
        self._raw_actions[env_ids] = 0.0


@configclass
class RateLimitedCarVWActionCfg(AckermannActionCfg):
    """Configuration for rate-limited 2D CarVW/Ackermann actions."""

    class_type: type[ActionTerm] = RateLimitedCarVWAction

    max_v_rate: float = 2.0
    """Maximum forward command rate in m/s^2."""

    max_steer_rate: float = 2.0
    """Maximum steering command rate in rad/s."""

    reset_to_zero: bool = True
    """Whether to reset previous executed action to zero at episode reset."""

