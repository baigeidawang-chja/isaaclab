"""Discrete nominal Car4WD model identified from policy-independent data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Sequence


DEFAULT_NOMINAL_MODEL_CONFIG_PATH = Path(__file__).parent / "config" / "nominal_car4wd_model.json"


@dataclass(frozen=True)
class FirstOrderDelayModel:
    """Scalar first-order model: y[k] = a*y[k-1] + b*u[k-delay]."""

    sample_time_s: float
    delay_steps: int
    a: float
    b: float

    def __post_init__(self) -> None:
        values = (self.sample_time_s, self.a, self.b)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("First-order model parameters must be finite.")
        if self.sample_time_s <= 0.0:
            raise ValueError("Sample time must be positive.")
        if self.delay_steps < 0:
            raise ValueError("Delay must be non-negative.")
        if not 0.0 <= self.a < 1.0:
            raise ValueError("Stable first-order pole must be in [0, 1).")
        if self.b <= 0.0:
            raise ValueError("First-order input coefficient must be positive.")

    @property
    def dc_gain(self) -> float:
        return self.b / (1.0 - self.a)

    @property
    def time_constant_s(self) -> float:
        if self.a == 0.0:
            return 0.0
        return -self.sample_time_s / math.log(self.a)

    @property
    def delay_s(self) -> float:
        return self.delay_steps * self.sample_time_s

    def next_output(self, previous_output: float, delayed_input: float) -> float:
        if not math.isfinite(previous_output) or not math.isfinite(delayed_input):
            raise ValueError("Model inputs must be finite.")
        return self.a * previous_output + self.b * delayed_input

    def simulate(
        self,
        inputs: Sequence[float],
        initial_output: float,
        input_history: Sequence[float] | None = None,
    ) -> list[float]:
        """Simulate samples aligned with inputs, preserving the measured first output."""
        if not inputs:
            return []
        if not math.isfinite(initial_output):
            raise ValueError("Initial output must be finite.")
        history = list(input_history or [])
        if len(history) > self.delay_steps:
            history = history[-self.delay_steps :]
        first_input = float(inputs[0])
        if not math.isfinite(first_input):
            raise ValueError("Model inputs must be finite.")
        if len(history) < self.delay_steps:
            history = [first_input] * (self.delay_steps - len(history)) + history

        delayed_inputs = history + [float(value) for value in inputs]
        outputs = [float(initial_output)]
        for sample_index in range(1, len(inputs)):
            delayed_input = delayed_inputs[sample_index]
            outputs.append(self.next_output(outputs[-1], delayed_input))
        return outputs

    @classmethod
    def from_mapping(cls, data: dict[str, Any], sample_time_s: float) -> "FirstOrderDelayModel":
        return cls(
            sample_time_s=sample_time_s,
            delay_steps=int(data["delay_steps"]),
            a=float(data["a"]),
            b=float(data["b"]),
        )


@dataclass(frozen=True)
class FirstOrderRegion:
    """Input interval and its locally linear first-order model."""

    input_min: float
    input_max: float
    model: FirstOrderDelayModel

    def __post_init__(self) -> None:
        if not math.isfinite(self.input_min) or not math.isfinite(self.input_max):
            raise ValueError("Scheduled-model bounds must be finite.")
        if self.input_min >= self.input_max:
            raise ValueError("Scheduled-model lower bound must be smaller than its upper bound.")


@dataclass(frozen=True)
class GainScheduledFirstOrderModel:
    """Piecewise first-order model scheduled by the delayed physical input."""

    regions: tuple[FirstOrderRegion, ...]

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("Gain-scheduled model requires at least one region.")
        sample_time = self.regions[0].model.sample_time_s
        delay_steps = self.regions[0].model.delay_steps
        previous_max = self.regions[0].input_min
        for region in self.regions:
            if not math.isclose(region.input_min, previous_max, abs_tol=1.0e-12):
                raise ValueError("Gain-scheduled regions must be contiguous and ordered.")
            if not math.isclose(region.model.sample_time_s, sample_time, abs_tol=1.0e-12):
                raise ValueError("All scheduled regions must use the same sample time.")
            if region.model.delay_steps != delay_steps:
                raise ValueError("All scheduled regions must use the same delay.")
            previous_max = region.input_max

    @property
    def sample_time_s(self) -> float:
        return self.regions[0].model.sample_time_s

    @property
    def delay_steps(self) -> int:
        return self.regions[0].model.delay_steps

    def model_for(self, delayed_input: float) -> FirstOrderDelayModel:
        if not math.isfinite(delayed_input):
            raise ValueError("Scheduled-model input must be finite.")
        for index, region in enumerate(self.regions):
            is_last = index == len(self.regions) - 1
            if region.input_min <= delayed_input < region.input_max or (
                is_last and math.isclose(delayed_input, region.input_max, abs_tol=1.0e-6)
            ):
                return region.model
        raise ValueError(
            f"Input {delayed_input} is outside the identified interval "
            f"[{self.regions[0].input_min}, {self.regions[-1].input_max}]."
        )

    def next_output(self, previous_output: float, delayed_input: float) -> float:
        return self.model_for(delayed_input).next_output(previous_output, delayed_input)

    def simulate(self, inputs: Sequence[float], initial_output: float) -> list[float]:
        if not inputs:
            return []
        values = [float(value) for value in inputs]
        outputs = [float(initial_output)]
        for sample_index in range(1, len(values)):
            delayed_index = max(0, sample_index - self.delay_steps)
            outputs.append(self.next_output(outputs[-1], values[delayed_index]))
        return outputs


@dataclass(frozen=True)
class NominalCar4WDModel:
    """Nominal longitudinal, steering-actuator, and Ackermann yaw model."""

    sample_time_s: float
    wheelbase_m: float
    speed: FirstOrderDelayModel | GainScheduledFirstOrderModel
    steering: FirstOrderDelayModel
    yaw_gain: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.sample_time_s) or self.sample_time_s <= 0.0:
            raise ValueError("Sample time must be finite and positive.")
        if not math.isfinite(self.wheelbase_m) or self.wheelbase_m <= 0.0:
            raise ValueError("Wheelbase must be finite and positive.")
        if not math.isfinite(self.yaw_gain) or self.yaw_gain <= 0.0:
            raise ValueError("Yaw gain must be finite and positive.")
        if not math.isclose(self.speed.sample_time_s, self.sample_time_s, abs_tol=1.0e-12):
            raise ValueError("Speed model sample time does not match the vehicle model.")
        if not math.isclose(self.steering.sample_time_s, self.sample_time_s, abs_tol=1.0e-12):
            raise ValueError("Steering model sample time does not match the vehicle model.")

    def yaw_rate_rad_s(self, speed_m_s: float, steering_rad: float) -> float:
        if not math.isfinite(speed_m_s) or not math.isfinite(steering_rad):
            raise ValueError("Yaw-model inputs must be finite.")
        return self.yaw_gain * speed_m_s * math.tan(steering_rad) / self.wheelbase_m

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "NominalCar4WDModel":
        if data.get("schema_version") != 1:
            raise ValueError("Unsupported nominal vehicle model schema version.")
        sample_time_s = float(data["sample_time_s"])
        dynamics = data["dynamics"]
        speed_data = dynamics["longitudinal_speed"]
        if speed_data.get("type", "first_order") == "gain_scheduled_first_order":
            speed_model: FirstOrderDelayModel | GainScheduledFirstOrderModel = (
                GainScheduledFirstOrderModel(
                    tuple(
                        FirstOrderRegion(
                            input_min=float(region["input_range_m_s"][0]),
                            input_max=float(region["input_range_m_s"][1]),
                            model=FirstOrderDelayModel.from_mapping(region, sample_time_s),
                        )
                        for region in speed_data["regions"]
                    )
                )
            )
        else:
            speed_model = FirstOrderDelayModel.from_mapping(speed_data, sample_time_s)
        return cls(
            sample_time_s=sample_time_s,
            wheelbase_m=float(data["geometry"]["wheelbase_m"]),
            speed=speed_model,
            steering=FirstOrderDelayModel.from_mapping(dynamics["equivalent_center_steering"], sample_time_s),
            yaw_gain=float(dynamics["yaw_rate"]["empirical_gain"]),
        )


def load_nominal_vehicle_model(
    path: str | Path = DEFAULT_NOMINAL_MODEL_CONFIG_PATH,
) -> NominalCar4WDModel:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as config_file:
        data = json.load(config_file)
    return NominalCar4WDModel.from_mapping(data)
