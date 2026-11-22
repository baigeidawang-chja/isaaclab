"""Sparse, locally linear nominal MPC solved with OSQP."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import osqp
import scipy.sparse as sparse

from .interfaces import NominalMPCReference, NominalMPCResult, NominalMPCState, wrap_to_pi
from .nominal_model import (
    FirstOrderDelayModel,
    GainScheduledFirstOrderModel,
    NominalCar4WDModel,
    load_nominal_vehicle_model,
)


DEFAULT_OSQP_MPC_CONFIG_PATH = Path(__file__).parent / "config" / "nominal_car4wd_bidirectional_osqp_mpc.json"


@dataclass(frozen=True)
class OSQPMPCConfig:
    horizon_steps: int
    state_weights: tuple[float, float, float]
    terminal_state_weights: tuple[float, float, float]
    control_weights: tuple[float, float]
    control_rate_weights: tuple[float, float]
    state_lower: tuple[float, float, float]
    state_upper: tuple[float, float, float]
    control_lower: tuple[float, float]
    control_upper: tuple[float, float]
    action_term_lower: tuple[float, float]
    action_term_upper: tuple[float, float]
    control_rate_per_s: tuple[float, float]
    eps_abs: float
    eps_rel: float
    max_iter: int
    polish: bool
    adaptive_rho: bool
    verbose: bool
    control_period_s: float
    reference_speed_bounds: tuple[float, float]
    identified_control_lower: tuple[float, float]
    identified_control_upper: tuple[float, float]
    model_config_path: Path

    def __post_init__(self) -> None:
        arrays = (
            self.state_weights,
            self.terminal_state_weights,
            self.control_weights,
            self.control_rate_weights,
            self.state_lower,
            self.state_upper,
            self.control_lower,
            self.control_upper,
            self.action_term_lower,
            self.action_term_upper,
            self.control_rate_per_s,
            self.reference_speed_bounds,
            self.identified_control_lower,
            self.identified_control_upper,
        )
        if self.horizon_steps < 2:
            raise ValueError("MPC horizon must contain at least two steps.")
        if not all(np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("MPC weights and constraints must be finite.")
        if np.any(np.asarray(self.state_weights) < 0.0) or np.any(
            np.asarray(self.terminal_state_weights) < 0.0
        ):
            raise ValueError("MPC state weights cannot be negative.")
        if np.any(np.asarray(self.control_weights) < 0.0) or np.any(
            np.asarray(self.control_rate_weights) < 0.0
        ):
            raise ValueError("MPC control weights cannot be negative.")
        if np.any(np.asarray(self.state_lower) >= np.asarray(self.state_upper)):
            raise ValueError("MPC state bounds are invalid.")
        if np.any(np.asarray(self.control_lower) >= np.asarray(self.control_upper)):
            raise ValueError("MPC control bounds are invalid.")
        if np.any(np.asarray(self.action_term_lower) >= np.asarray(self.action_term_upper)):
            raise ValueError("ActionTerm physical bounds are invalid.")
        if np.any(np.asarray(self.control_rate_per_s) <= 0.0):
            raise ValueError("MPC control-rate limits must be positive.")
        if self.control_period_s <= 0.0 or not math.isfinite(self.control_period_s):
            raise ValueError("MPC control period must be finite and positive.")
        if self.eps_abs <= 0.0 or self.eps_rel <= 0.0 or self.max_iter < 1:
            raise ValueError("OSQP tolerances and iteration limit must be positive.")
        if self.reference_speed_bounds[0] >= self.reference_speed_bounds[1]:
            raise ValueError("Planner speed-reference bounds are invalid.")
        for lower, upper in (
            (self.action_term_lower, self.action_term_upper),
            (self.identified_control_lower, self.identified_control_upper),
        ):
            if np.any(np.asarray(self.control_lower) < lower) or np.any(
                np.asarray(self.control_upper) > upper
            ):
                raise ValueError("MPC control bounds exceed ActionTerm or identified input support.")

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "OSQPMPCConfig":
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported nominal OSQP MPC schema version.")
        weights = data["weights"]
        constraints = data["constraints"]
        action_bounds = constraints["action_term_physical_bounds"]
        effective_bounds = constraints["effective_mpc_control_bounds"]
        identified_bounds = constraints["identified_model_bounds"]
        state_bounds = constraints["predicted_state_bounds"]
        rates = constraints["physical_rate_limits_per_second"]
        solver = data["solver"]
        return cls(
            horizon_steps=int(data["horizon_steps"]),
            model_config_path=Path(__file__).parent / "config" / data["model_parameter_file"],
            state_weights=tuple(float(value) for value in weights["state"]),
            terminal_state_weights=tuple(float(value) for value in weights["terminal_state"]),
            control_weights=tuple(float(value) for value in weights["control"]),
            control_rate_weights=tuple(float(value) for value in weights["control_rate"]),
            state_lower=(
                float(state_bounds["heading_error_rad"][0]),
                float(state_bounds["body_speed_m_s"][0]),
                float(state_bounds["center_steering_rad"][0]),
            ),
            state_upper=(
                float(state_bounds["heading_error_rad"][1]),
                float(state_bounds["body_speed_m_s"][1]),
                float(state_bounds["center_steering_rad"][1]),
            ),
            control_lower=(
                float(effective_bounds["speed_target_m_s"][0]),
                float(effective_bounds["center_steering_target_rad"][0]),
            ),
            control_upper=(
                float(effective_bounds["speed_target_m_s"][1]),
                float(effective_bounds["center_steering_target_rad"][1]),
            ),
            action_term_lower=(
                float(action_bounds["speed_target_m_s"][0]),
                float(action_bounds["center_steering_target_rad"][0]),
            ),
            action_term_upper=(
                float(action_bounds["speed_target_m_s"][1]),
                float(action_bounds["center_steering_target_rad"][1]),
            ),
            control_rate_per_s=(
                float(rates["speed_target_m_s2"]),
                float(rates["center_steering_target_rad_s"]),
            ),
            eps_abs=float(solver["eps_abs"]),
            eps_rel=float(solver["eps_rel"]),
            max_iter=int(solver["max_iter"]),
            polish=bool(solver["polish"]),
            adaptive_rho=bool(solver["adaptive_rho"]),
            verbose=bool(solver["verbose"]),
            control_period_s=float(data["provenance"]["control_period_s"]),
            reference_speed_bounds=tuple(
                float(value) for value in constraints.get(
                    "planner_reference_bounds", {"speed_m_s": effective_bounds["speed_target_m_s"]}
                )["speed_m_s"]
            ),
            identified_control_lower=(
                float(identified_bounds["speed_target_m_s"][0]),
                float(identified_bounds["center_steering_target_rad"][0]),
            ),
            identified_control_upper=(
                float(identified_bounds["speed_target_m_s"][1]),
                float(identified_bounds["center_steering_target_rad"][1]),
            ),
        )


def load_osqp_mpc_config(path: str | Path = DEFAULT_OSQP_MPC_CONFIG_PATH) -> OSQPMPCConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as config_file:
        data = json.load(config_file)
    return replace(OSQPMPCConfig.from_mapping(data),
                   model_config_path=config_path.parent / data["model_parameter_file"])


class OSQPNominalMPC:
    """Region-frozen linear MPC using identified dynamics and physical controls."""

    state_dim = 3
    control_dim = 2

    def __init__(
        self,
        model: NominalCar4WDModel | None = None,
        config: OSQPMPCConfig | None = None,
    ) -> None:
        self.config = config or load_osqp_mpc_config()
        self.model = model or load_nominal_vehicle_model(self.config.model_config_path)
        if isinstance(self.model.speed, GainScheduledFirstOrderModel):
            if (self.config.control_lower[0] < self.model.speed.regions[0].input_min - 1.0e-6
                    or self.config.control_upper[0] > self.model.speed.regions[-1].input_max + 1.0e-6):
                raise ValueError("MPC speed bounds exceed the loaded identified model, not just its config label.")
        if not math.isclose(
            self.model.sample_time_s, self.config.control_period_s, abs_tol=1.0e-12
        ):
            raise ValueError("Identified model dt does not match the MPC/ActionTerm control period.")
        if self.model.speed.delay_steps != 0 or self.model.steering.delay_steps != 0:
            raise NotImplementedError(
                "This three-state QP requires the independently selected zero-delay model; "
                "nonzero identified delays require explicit delay-state augmentation."
            )
        self._last_solution: np.ndarray | None = None
        self._last_control = np.zeros(self.control_dim, dtype=np.float64)

    @property
    def control_period_s(self) -> float:
        return self.config.control_period_s

    def reset(self, previous_control: Sequence[float] | None = None) -> None:
        self._last_solution = None
        values = (0.0, 0.0) if previous_control is None else previous_control
        self._last_control = self._validated_control(values)

    def _validated_control(self, control: Sequence[float]) -> np.ndarray:
        values = np.asarray(control, dtype=np.float64)
        if values.shape != (self.control_dim,) or not np.all(np.isfinite(values)):
            raise ValueError("Previous physical control must be a finite two-vector.")
        return values

    def _speed_model(self, scheduling_control_m_s: float) -> tuple[int, FirstOrderDelayModel]:
        speed = self.model.speed
        if isinstance(speed, GainScheduledFirstOrderModel):
            bounded = float(
                np.clip(
                    scheduling_control_m_s,
                    speed.regions[0].input_min,
                    speed.regions[-1].input_max,
                )
            )
            for index, region in enumerate(speed.regions):
                is_last = index == len(speed.regions) - 1
                if region.input_min <= bounded < region.input_max or (
                    is_last and math.isclose(bounded, region.input_max, abs_tol=1.0e-9)
                ):
                    return index, region.model
            raise RuntimeError("Unable to select an identified speed region.")
        return 0, speed

    def linearized_dynamics(
        self,
        speed_model: FirstOrderDelayModel,
        linearization_speed_m_s: float,
        linearization_steering_rad: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return x[k+1] = A*x[k] + B*u[k] + c for the frozen local model."""
        dt = self.model.sample_time_s
        wheelbase = self.model.wheelbase_m
        gain = self.model.yaw_gain
        delta = float(linearization_steering_rad)
        speed = float(linearization_speed_m_s)
        sec_squared = 1.0 / math.cos(delta) ** 2
        dr_dv = gain * math.tan(delta) / wheelbase
        dr_ddelta = gain * speed * sec_squared / wheelbase
        yaw_rate = gain * speed * math.tan(delta) / wheelbase
        affine_yaw = yaw_rate - dr_dv * speed - dr_ddelta * delta

        a = np.asarray(
            [
                [1.0, -dt * dr_dv, -dt * dr_ddelta],
                [0.0, speed_model.a, 0.0],
                [0.0, 0.0, self.model.steering.a],
            ],
            dtype=np.float64,
        )
        b = np.asarray(
            [
                [0.0, 0.0],
                [speed_model.b, 0.0],
                [0.0, self.model.steering.b],
            ],
            dtype=np.float64,
        )
        c = np.asarray([-dt * affine_yaw, 0.0, 0.0], dtype=np.float64)
        return a, b, c

    def _x_slice(self, step: int) -> slice:
        start = step * self.state_dim
        return slice(start, start + self.state_dim)

    def _u_slice(self, step: int) -> slice:
        state_variables = (self.config.horizon_steps + 1) * self.state_dim
        start = state_variables + step * self.control_dim
        return slice(start, start + self.control_dim)

    def _build_qp(self, state: np.ndarray, reference: NominalMPCReference, previous_control: np.ndarray):
        horizon = self.config.horizon_steps
        nx = self.state_dim
        nu = self.control_dim
        variables = (horizon + 1) * nx + horizon * nu
        speed_region, speed_model = self._speed_model(previous_control[0])
        # Preserve measured travel direction while braking through zero. At rest,
        # use the requested departure direction; a zero-speed hold cannot turn.
        linearization_speed = float(state[1]) if abs(state[1]) > 0.02 else reference.speed_m_s
        linearization_steering = float(
            np.clip(state[2], self.config.state_lower[2], self.config.state_upper[2])
        )
        dynamics_a, dynamics_b, dynamics_c = self.linearized_dynamics(
            speed_model, linearization_speed, linearization_steering
        )

        p = sparse.lil_matrix((variables, variables), dtype=np.float64)
        q = np.zeros(variables, dtype=np.float64)
        state_reference = reference.as_array()
        input_reference = np.asarray([reference.speed_m_s, 0.0], dtype=np.float64)
        q_state = np.asarray(self.config.state_weights)
        q_terminal = np.asarray(self.config.terminal_state_weights)
        r_control = np.asarray(self.config.control_weights)
        r_rate = np.asarray(self.config.control_rate_weights)

        for step in range(horizon + 1):
            weights = q_terminal if step == horizon else q_state
            indices = np.arange(self._x_slice(step).start, self._x_slice(step).stop)
            p[indices, indices] += 2.0 * weights
            q[indices] += -2.0 * weights * state_reference
        for step in range(horizon):
            indices = np.arange(self._u_slice(step).start, self._u_slice(step).stop)
            p[indices, indices] += 2.0 * r_control
            q[indices] += -2.0 * r_control * input_reference
            if step == 0:
                p[indices, indices] += 2.0 * r_rate
                q[indices] += -2.0 * r_rate * previous_control
            else:
                previous = np.arange(self._u_slice(step - 1).start, self._u_slice(step - 1).stop)
                p[indices, indices] += 2.0 * r_rate
                p[previous, previous] += 2.0 * r_rate
                for axis in range(nu):
                    p[indices[axis], previous[axis]] += -2.0 * r_rate[axis]
                    p[previous[axis], indices[axis]] += -2.0 * r_rate[axis]

        row_count = nx + horizon * nx + horizon * nx + horizon * nu + horizon * nu
        matrix = sparse.lil_matrix((row_count, variables), dtype=np.float64)
        lower = np.empty(row_count, dtype=np.float64)
        upper = np.empty(row_count, dtype=np.float64)
        row = 0
        matrix[row : row + nx, self._x_slice(0)] = sparse.eye(nx)
        lower[row : row + nx] = state
        upper[row : row + nx] = state
        row += nx

        for step in range(horizon):
            matrix[row : row + nx, self._x_slice(step + 1)] = sparse.eye(nx)
            matrix[row : row + nx, self._x_slice(step)] = -dynamics_a
            matrix[row : row + nx, self._u_slice(step)] = -dynamics_b
            lower[row : row + nx] = dynamics_c
            upper[row : row + nx] = dynamics_c
            row += nx

        state_lower = np.asarray(self.config.state_lower)
        state_upper = np.asarray(self.config.state_upper)
        for step in range(1, horizon + 1):
            matrix[row : row + nx, self._x_slice(step)] = sparse.eye(nx)
            lower[row : row + nx] = state_lower
            upper[row : row + nx] = state_upper
            row += nx

        control_lower = np.asarray(self.config.control_lower)
        control_upper = np.asarray(self.config.control_upper)
        for step in range(horizon):
            matrix[row : row + nu, self._u_slice(step)] = sparse.eye(nu)
            lower[row : row + nu] = control_lower
            upper[row : row + nu] = control_upper
            row += nu

        max_change = np.asarray(self.config.control_rate_per_s) * self.control_period_s
        for step in range(horizon):
            matrix[row : row + nu, self._u_slice(step)] = sparse.eye(nu)
            if step == 0:
                lower[row : row + nu] = previous_control - max_change
                upper[row : row + nu] = previous_control + max_change
            else:
                matrix[row : row + nu, self._u_slice(step - 1)] = -sparse.eye(nu)
                lower[row : row + nu] = -max_change
                upper[row : row + nu] = max_change
            row += nu
        if row != row_count:
            raise RuntimeError("Internal MPC constraint row count mismatch.")

        return (
            sparse.triu(p.tocsc()).tocsc(),
            q,
            matrix.tocsc(),
            lower,
            upper,
            dynamics_a,
            dynamics_b,
            dynamics_c,
            speed_region,
            linearization_speed,
            linearization_steering,
        )

    def _shifted_warm_start(self) -> np.ndarray | None:
        if self._last_solution is None:
            return None
        horizon = self.config.horizon_steps
        states = np.vstack([self._last_solution[self._x_slice(step)] for step in range(horizon + 1)])
        controls = np.vstack([self._last_solution[self._u_slice(step)] for step in range(horizon)])
        return np.concatenate(
            [np.vstack([states[1:], states[-1]]).ravel(), np.vstack([controls[1:], controls[-1]]).ravel()]
        )

    def _run_osqp(self, p, q, a, lower, upper, warm_start):
        solver = osqp.OSQP()
        solver.setup(
            P=p,
            q=q,
            A=a,
            l=lower,
            u=upper,
            eps_abs=self.config.eps_abs,
            eps_rel=self.config.eps_rel,
            max_iter=self.config.max_iter,
            polish=self.config.polish,
            adaptive_rho=self.config.adaptive_rho,
            verbose=self.config.verbose,
            warm_start=True,
        )
        if warm_start is not None:
            solver.warm_start(x=warm_start)
        return solver.solve()

    def _fallback(
        self,
        reason: str,
        state: np.ndarray | None,
        previous_control: np.ndarray | None,
        elapsed_s: float,
        *,
        status: str = "fallback",
        status_value: int | None = None,
        info: Any | None = None,
    ) -> NominalMPCResult:
        previous = self._last_control if previous_control is None else previous_control
        previous = np.asarray(previous, dtype=np.float64)
        if previous.shape != (2,) or not np.all(np.isfinite(previous)):
            previous = np.zeros(2, dtype=np.float64)
        max_change = np.asarray(self.config.control_rate_per_s) * self.control_period_s
        control = previous + np.clip(-previous, -max_change, max_change)
        control = np.clip(control, self.config.action_term_lower, self.config.action_term_upper)
        safe_state = (
            np.zeros(self.state_dim, dtype=np.float64)
            if state is None or state.shape != (self.state_dim,) or not np.all(np.isfinite(state))
            else state
        )
        predicted_states = np.tile(safe_state, (self.config.horizon_steps + 1, 1))
        predicted_controls = np.tile(control, (self.config.horizon_steps, 1))
        self._last_solution = None
        self._last_control = control.copy()
        return NominalMPCResult(
            control=control,
            predicted_states=predicted_states,
            predicted_controls=predicted_controls,
            success=False,
            status=status,
            status_value=status_value,
            solve_time_s=elapsed_s,
            solver_run_time_s=getattr(info, "run_time", None),
            iterations=getattr(info, "iter", None),
            objective=getattr(info, "obj_val", None),
            primal_residual=getattr(info, "pri_res", None),
            dual_residual=getattr(info, "dua_res", None),
            warm_start_used=False,
            fallback_used=True,
            fallback_reason=reason,
            speed_region_index=None,
            linearization_speed_m_s=None,
            linearization_steering_rad=None,
        )

    def solve(
        self,
        state: NominalMPCState | Sequence[float],
        reference: NominalMPCReference | Sequence[float],
        previous_control: Sequence[float] | None = None,
    ) -> NominalMPCResult:
        """Solve one QP and return the first physical speed/steering target."""
        started = time.perf_counter()
        try:
            previous = (
                self._last_control.copy()
                if previous_control is None
                else self._validated_control(previous_control)
            )
            state_values = (
                state.as_array()
                if isinstance(state, NominalMPCState)
                else np.asarray(state, dtype=np.float64)
            )
            if state_values.shape != (self.state_dim,) or not np.all(np.isfinite(state_values)):
                raise ValueError("Nominal MPC state must be a finite three-vector.")
            state_values = state_values.copy()
            state_values[0] = wrap_to_pi(float(state_values[0]))
            if isinstance(reference, NominalMPCReference):
                reference_value = reference
            else:
                reference_array = np.asarray(reference, dtype=np.float64)
                if reference_array.shape != (2,) or not np.all(np.isfinite(reference_array)):
                    raise ValueError("Nominal MPC reference must be [speed, heading-error target].")
                reference_value = NominalMPCReference(
                    speed_m_s=float(reference_array[0]),
                    heading_error_rad=wrap_to_pi(float(reference_array[1])),
                )
            if not self.config.reference_speed_bounds[0] <= reference_value.speed_m_s <= self.config.reference_speed_bounds[1]:
                raise ValueError("Speed reference is outside the planner reference range.")
            warm_start = self._shifted_warm_start()
            qp = self._build_qp(state_values, reference_value, previous)
            p, q, matrix, lower, upper = qp[:5]
            speed_region, linearization_speed, linearization_steering = qp[8:]
            try:
                solution = self._run_osqp(p, q, matrix, lower, upper, warm_start)
            except Exception as error:
                return self._fallback(
                    f"osqp_exception: {type(error).__name__}: {error}",
                    state_values,
                    previous,
                    time.perf_counter() - started,
                    status="solver_exception",
                )
        except (ValueError, FloatingPointError) as error:
            return self._fallback(
                f"invalid_input: {error}",
                locals().get("state_values"),
                locals().get("previous"),
                time.perf_counter() - started,
                status="invalid_input",
            )

        info = solution.info
        status = str(getattr(info, "status", "unknown"))
        status_value = getattr(info, "status_val", None)
        success = status.lower() in {"solved", "solved inaccurate"} and solution.x is not None
        elapsed = time.perf_counter() - started
        if not success:
            return self._fallback(
                f"osqp_{status.lower().replace(' ', '_')}",
                state_values,
                previous,
                elapsed,
                status=status,
                status_value=status_value,
                info=info,
            )

        decision = np.asarray(solution.x, dtype=np.float64)
        if decision.shape != (p.shape[0],) or not np.all(np.isfinite(decision)):
            return self._fallback(
                "osqp_nonfinite_solution", state_values, previous, elapsed,
                status=status, status_value=status_value, info=info,
            )
        horizon = self.config.horizon_steps
        predicted_states = np.vstack([decision[self._x_slice(step)] for step in range(horizon + 1)])
        predicted_controls = np.vstack([decision[self._u_slice(step)] for step in range(horizon)])
        control = predicted_controls[0].copy()
        self._last_solution = decision.copy()
        self._last_control = control.copy()
        return NominalMPCResult(
            control=control,
            predicted_states=predicted_states,
            predicted_controls=predicted_controls,
            success=True,
            status=status,
            status_value=status_value,
            solve_time_s=elapsed,
            solver_run_time_s=getattr(info, "run_time", None),
            iterations=getattr(info, "iter", None),
            objective=getattr(info, "obj_val", None),
            primal_residual=getattr(info, "pri_res", None),
            dual_residual=getattr(info, "dua_res", None),
            warm_start_used=warm_start is not None,
            fallback_used=False,
            fallback_reason=None,
            speed_region_index=speed_region,
            linearization_speed_m_s=linearization_speed,
            linearization_steering_rad=linearization_steering,
        )


class NominalMPC(OSQPNominalMPC):
    """Public nominal MPC interface backed by the standard sparse OSQP QP."""

