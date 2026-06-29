from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.assets import Articulation
from WheeledLab.source.wheeledlab.wheeledlab.envs.mdp.actions import AckermannAction, AckermannActionCfg

class Car4WdAction(ActionTerm):
    cfg: Car4WDActionCfg

    _asset: Articulation

    _scale: torch.Tensor

    _offset: torch.Tensor

    _bounding_strategy: str | None

    _raw_actions: torch.Tensor

    _processed_actions: torch.Tensor

    def __init__(self, cfg:Car4WDActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        wheel_ids, wheel_names = self._asset.find_joints(cfg.wheel_joint_names)
        self. _wheel_ids = wheel_ids
        self._wheel_names = wheel_names

        steering_ids, steering_names = self._asset.find_joints(cfg.steering_joint_names)
        self._steering_ids = steering_ids
        self._steering_names = steering_names

        self._scale = torch.tensor(cfg.scale, device=self.device, dtype=torch.float32)
        self._offset = torch.tensor(cfg.offset, device=self.device, dtype=torch.float)
        self._bounding_strategy = cfg.bounding_strategy

        self.base_length = torch.tensor(cfg.base_length, device = self.device)
        self.base_width = torch.tensor(cfg.base_width, device = self.device)
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)

        self._prev_processed_actions = torch.zeros(env.num_envs, self.action_dim, device=self.device)
        self._max_action_change = torch.tensor(cfg.max_action_change, device=self.device)

    @property
    def action_dim(self) -> int:
        return 5

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        if self._bounding_strategy == "clip":
            processed = torch.clip(actions, min=-1.0, max=1.0)*self._scale+self._offset
        elif self._bounding_strategy == 'tanh':
            processed = torch.tanh(actions) * self._scale + self._offset
        else:
            processed = actions * self._scale +self._offset

        action_change = processed - self._prev_processed_actions
        action_change = torch.clamp(action_change, min=-self._max_action_change, max=self._max_action_change)
        self._processed_actions = self._prev_processed_actions + action_change
        self._prev_processed_actions[:] = self._processed_actions

    def apply_actions(self):
        #print(f"[DEBUG] processed_actions: {self.processed_actions[0]}")  # 只打印 env0
        wheel_speeds = self.processed_actions[:, :4]
        wheel_speeds_FL = self.processed_actions[:, 0]
        wheel_speeds_FR = self.processed_actions[:, 1]
        wheel_speeds_RL = self.processed_actions[:, 2]
        wheel_speeds_RR = self.processed_actions[:, 3]
        target_wheel_angles = self.processed_actions[:, 4]

        steering_angles_FR, steering_angles_FL = self._calculate_ackermann_angles(target_wheel_angles)
        front_wheel_angles = torch.stack([steering_angles_FR, steering_angles_FL], dim=1)

        #print(f"wheel_sppeds01: {wheel_speeds[:, 0]} wheel_sppeds02: {wheel_speeds[:, 1]} wheel_sppeds03: {wheel_speeds[:, 2]} wheel_sppeds04: {wheel_speeds[:, 3]}")
        #print(f"front_wheelangles_R: {front_wheel_angles[:, 0]} front_wheelangles_L: {front_wheel_angles[:, 1]}")
        wheel_speeds_cmd = self.processed_actions[:, :4]
        self._asset.set_joint_velocity_target(wheel_speeds, joint_ids = self._wheel_ids)
        if not hasattr(self, "_dbg_count"):
            self._dbg_count = 0
        self._dbg_count += 1
        if self._dbg_count % 200 == 0:
            w_now = self._asset.data.joint_vel[0, self._wheel_ids].detach().cpu()
            print(f"[DEBUG] cmd[0]={wheel_speeds_cmd[0].detach().cpu()}  vel_now[0]={w_now}")

        # self._asset.set_joint_velocity_target(wheel_speeds_FL, joint_ids = self._wheel_ids)
        # self._asset.set_joint_velocity_target(wheel_speeds_FR, joint_ids)
        # self._asset.set_joint_velocity_target(wheel_speeds_RL, joint_ids)
        # self._asset.set_joint_velocity_target(wheel_speeds_RR, joint_ids)
        self._asset.set_joint_position_target(front_wheel_angles, joint_ids = self._steering_ids)

    def _calculate_ackermann_angles(self, target_wheel_angles):
        L = self.base_length
        W = self.base_width

        # 1) clamp steering input to avoid tan() blow-up near +-pi/2
        # choose a realistic max steering, e.g. 0.5 rad (~28.6 deg)
        max_steer = 0.5
        a = torch.clamp(target_wheel_angles.float(), -max_steer, max_steer)

        # 2) compute tan safely
        tan_steering = torch.tan(a)

        # avoid division by very small tan -> huge radius
        eps = 1e-6
        tan_steering = torch.where(
            torch.abs(tan_steering) < eps,
            torch.sign(tan_steering) * eps,
            tan_steering,
        )

        R = L / tan_steering  # turning radius (can be +/-)

        # 3) protect denominators (R +/- W/2) from crossing 0
        denom_left = R - W / 2
        denom_right = R + W / 2

        denom_left = torch.where(torch.abs(denom_left) < eps, torch.sign(denom_left) * eps, denom_left)
        denom_right = torch.where(torch.abs(denom_right) < eps, torch.sign(denom_right) * eps, denom_right)

        delta_left = torch.atan(L / denom_left)
        delta_right = torch.atan(L / denom_right)

        # final sanity clamp (optional)
        delta_left = torch.clamp(delta_left, -max_steer, max_steer)
        delta_right = torch.clamp(delta_right, -max_steer, max_steer)

        return delta_left, delta_right
    
@configclass
class Car4WDActionCfg(ActionTermCfg):

    class_type: type[ActionTerm] = Car4WdAction

    wheel_joint_names: list[str] = MISSING

    steering_joint_names: list[str] = MISSING

    scale: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    
    offset: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0)

    bounding_strategy: str | None = "tanh"

    base_length: float = 1.0

    base_width: float = 1.0
    
    max_action_change: tuple[float, float, float, float, float] = (1.0, 1.0, 1.0, 1.0, 1.0)


class CarVWAction(AckermannAction):
    def __init__(self, cfg: "CarVWActionCfg", env):
        super().__init__(cfg, env)
        self._env = env
        env._thruster_cmd = torch.zeros(env.num_envs, device=self.device, dtype=torch.float32)
        # Rate limits in physical units per second.
        self._use_rate_limit = bool(cfg.use_rate_limit)
        self._max_speed_rate = float(cfg.max_speed_rate)
        self._max_steer_rate = float(cfg.max_steer_rate)
        self._control_dt = float(getattr(env, "step_dt", 1.0))
        self._prev_processed_actions = torch.zeros(
            env.num_envs, self.action_dim, device=self.device, dtype=torch.float32
        )
        self._has_prev_processed_actions = False

    @property
    def action_dim(self) -> int:
        return 3 if bool(getattr(self.cfg, "use_thruster_action", False)) else 2

    def process_actions(self, actions):
        self._raw_actions[:] = actions
        if self._bounding_strategy == "clip":
            processed = torch.clip(actions, min=-1.0, max=1.0) * self._scale + self._offset
        elif self._bounding_strategy == "tanh":
            processed = torch.tanh(actions) * self._scale + self._offset
        else:
            processed = actions * self._scale + self._offset

        if self.cfg.no_reverse:
            processed[:, 0] = torch.clamp(processed[:, 0], min=0.0)
        if self.action_dim >= 3:
            processed[:, 2] = torch.clamp(processed[:, 2], min=-float(self.cfg.thruster_cmd_limit), max=float(self.cfg.thruster_cmd_limit))
            self._env._thruster_cmd[:] = processed[:, 2]
        else:
            self._env._thruster_cmd.zero_()

        if self._use_rate_limit:
            if not self._has_prev_processed_actions:
                self._prev_processed_actions[:] = processed
                self._has_prev_processed_actions = True
            max_dv = self._max_speed_rate * self._control_dt
            max_ds = self._max_steer_rate * self._control_dt
            dv = torch.clamp(
                processed[:, 0] - self._prev_processed_actions[:, 0], min=-max_dv, max=max_dv
            )
            ds = torch.clamp(
                processed[:, 1] - self._prev_processed_actions[:, 1], min=-max_ds, max=max_ds
            )
            processed[:, 0] = self._prev_processed_actions[:, 0] + dv
            processed[:, 1] = self._prev_processed_actions[:, 1] + ds
            self._prev_processed_actions[:] = processed

        self._processed_actions = processed

    def _calculate_ackermann_angles_and_velocities(self, target_velocity, target_steering_angle):

        L = self.base_length
        W = self.base_width
        wheel_radius = self.wheel_rad

        # Keep steering in a realistic range to avoid tan() singularity near +-pi/2.
        max_steer = 0.6  # rad, about 34 deg
        steer = torch.clamp(target_steering_angle.float(), -max_steer, max_steer)
        tan_steering = torch.tan(steer)
        eps = 1e-6
        tan_steering = torch.where(
            torch.abs(tan_steering) < eps,
            torch.sign(tan_steering) * eps,
            tan_steering,
        )
        R = L / tan_steering

        # Standard Ackermann geometry for front left/right wheel angles.
        denom_left = R - W / 2
        denom_right = R + W / 2
        denom_left = torch.where(torch.abs(denom_left) < eps, torch.sign(denom_left) * eps, denom_left)
        denom_right = torch.where(torch.abs(denom_right) < eps, torch.sign(denom_right) * eps, denom_right)
        delta_left = torch.atan(L / denom_left)
        delta_right = torch.atan(L / denom_right)
        delta_left = torch.clamp(delta_left, -max_steer, max_steer)
        delta_right = torch.clamp(delta_right, -max_steer, max_steer)

        # Assuming the rear wheels follow the path's radius adjusted for their position
        R_rear_left = torch.sqrt((R - W/2)**2 + L**2)
        R_rear_right = torch.sqrt((R + W/2)**2 + L**2)

        # Velocity adjustment based on wheel's distance from the IC
        v_front_left = target_velocity * torch.abs(R_rear_left / (R * wheel_radius))
        v_front_right = target_velocity * torch.abs(R_rear_right / (R * wheel_radius))

        v_back_left = target_velocity * torch.abs((R - W / 2) / (R * wheel_radius))
        v_back_right = target_velocity * torch.abs((R + W / 2) / (R * wheel_radius))

        # Calculate target rotation for each wheel based on its velocity
        wheel_speeds = torch.stack([v_back_left, v_back_right, v_front_left, v_front_right], dim=1)

        return delta_left, delta_right, wheel_speeds


@configclass
class CarVWActionCfg(AckermannActionCfg):
    """2D action term config for (v, w)."""

    class_type: type[ActionTerm] = CarVWAction
    use_rate_limit: bool = True
    max_speed_rate: float = 1.0
    max_steer_rate: float = 1.2
    use_thruster_action: bool = False
    thruster_cmd_limit: float = 1.0
