"""Fit and independently validate the nominal Car4WD model from recorded CSV data."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from isaaclab_tasks.user.traction_protective_control.mpc.nominal_model import (
    FirstOrderDelayModel,
    FirstOrderRegion,
    GainScheduledFirstOrderModel,
    NominalCar4WDModel,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(
    "/media/chja/CE54D158C95990271/IsaacLabTrainingData/"
    "traction_protective_control_nominal_identification/20261028_152550"
)
DEFAULT_STEERING_CONFIG = (
    REPO_ROOT
    / "source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/mpc/config"
    / "nominal_car4wd_steering.json"
)
DEFAULT_MODEL_CONFIG = DEFAULT_STEERING_CONFIG.with_name("nominal_car4wd_model.json")

REQUIRED_COLUMNS = {
    "split",
    "profile_type",
    "profile_name",
    "run_index",
    "segment_index",
    "segment_step",
    "run_time_s",
    "step_dt_s",
    "requested_target_speed_m_s",
    "requested_target_steering_rad",
    "executed_speed_m_s",
    "executed_steering_rad",
    "actual_body_vx_m_s",
    "actual_equivalent_center_steering_rad",
    "actual_yaw_rate_rad_s",
}


@dataclass(frozen=True)
class Candidate:
    structure: str
    delay_steps: int
    a: float
    b: float
    dc_gain: float
    time_constant_s: float
    cv_simulation_rmse: float
    identification_one_step_rmse: float
    identification_simulation_rmse: float
    parameter_count: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    fit = subparsers.add_parser("fit", help="Fit using identification.csv only")
    fit.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    fit.add_argument("--output-dir", type=Path, required=True)
    fit.add_argument("--parameter-output", type=Path, default=DEFAULT_MODEL_CONFIG)
    fit.add_argument("--max-delay-steps", type=int, default=6)

    validate = subparsers.add_parser(
        "validate", help="Evaluate frozen parameters using validation.csv only"
    )
    validate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--parameters", type=Path, default=DEFAULT_MODEL_CONFIG)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _load_csv(path: Path, expected_split: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset is not accessible: {path}")
    data = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if set(data["split"].unique()) != {expected_split}:
        raise ValueError(f"{path.name} does not contain only the {expected_split} split.")
    numeric = data[list(REQUIRED_COLUMNS - {"split", "profile_type", "profile_name"})]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name} contains non-finite required values.")
    step_times = data["step_dt_s"].unique()
    if len(step_times) != 1 or not math.isclose(float(step_times[0]), 0.05, abs_tol=1.0e-12):
        raise ValueError(f"Expected a fixed 0.05 s sample time, got {step_times}.")
    return data.sort_values(["run_index", "run_time_s"]).reset_index(drop=True)


def _sample_arrays(
    data: pd.DataFrame,
    input_column: str,
    output_column: str,
    delay_steps: int,
    excluded_rows: set[int] | None = None,
    selected_rows: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    previous: list[float] = []
    delayed_input: list[float] = []
    target: list[float] = []
    excluded_rows = excluded_rows or set()
    for _, run in data.groupby("run_index", sort=True):
        run = run.sort_values("run_time_s")
        row_ids = run.index.to_numpy()
        inputs = run[input_column].to_numpy(dtype=float)
        outputs = run[output_column].to_numpy(dtype=float)
        for position in range(max(1, delay_steps), len(run)):
            row_id = int(row_ids[position])
            if row_id in excluded_rows:
                continue
            if selected_rows is not None and row_id not in selected_rows:
                continue
            previous.append(float(outputs[position - 1]))
            delayed_input.append(float(inputs[position - delay_steps]))
            target.append(float(outputs[position]))
    if not target:
        raise ValueError("No regression samples remain after applying the selection.")
    return np.asarray(previous), np.asarray(delayed_input), np.asarray(target)


def _fit_model(
    data: pd.DataFrame,
    input_column: str,
    output_column: str,
    delay_steps: int,
    structure: str,
    excluded_rows: set[int] | None = None,
    selected_rows: set[int] | None = None,
) -> FirstOrderDelayModel:
    previous, delayed_input, target = _sample_arrays(
        data,
        input_column,
        output_column,
        delay_steps,
        excluded_rows,
        selected_rows,
    )
    sample_time = float(data["step_dt_s"].iloc[0])
    if structure == "unity_gain":
        denominator = float(np.dot(previous - delayed_input, previous - delayed_input))
        if denominator <= 1.0e-16:
            raise ValueError("Unity-gain fit is not excited.")
        a = float(np.dot(previous - delayed_input, target - delayed_input) / denominator)
        a = float(np.clip(a, 0.0, 0.9999))
        gain = 1.0
    elif structure == "free_gain":
        def residual(parameters: np.ndarray) -> np.ndarray:
            pole, gain_value = parameters
            return pole * previous + (1.0 - pole) * gain_value * delayed_input - target

        result = least_squares(
            residual,
            x0=np.asarray([0.8, 1.0]),
            bounds=(np.asarray([0.0, 0.2]), np.asarray([0.9999, 3.0])),
        )
        if not result.success:
            raise RuntimeError(f"First-order fit failed: {result.message}")
        a, gain = (float(value) for value in result.x)
    else:
        raise ValueError(f"Unknown model structure: {structure}")
    return FirstOrderDelayModel(sample_time, delay_steps, a, (1.0 - a) * gain)


def _one_step_predictions(
    data: pd.DataFrame,
    model: FirstOrderDelayModel,
    input_column: str,
    output_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    previous, delayed_input, target = _sample_arrays(
        data, input_column, output_column, model.delay_steps
    )
    predicted = model.a * previous + model.b * delayed_input
    return target, predicted


def _simulate_runs(
    data: pd.DataFrame,
    model: FirstOrderDelayModel | GainScheduledFirstOrderModel,
    input_column: str,
    output_column: str,
) -> pd.Series:
    prediction = pd.Series(np.nan, index=data.index, dtype=float)
    for _, run in data.groupby("run_index", sort=True):
        run = run.sort_values("run_time_s")
        inputs = run[input_column].to_numpy(dtype=float)
        measured = run[output_column].to_numpy(dtype=float)
        prediction.loc[run.index] = model.simulate(inputs.tolist(), float(measured[0]))
    return prediction


def _simulate_interval(
    run: pd.DataFrame,
    model: FirstOrderDelayModel,
    input_column: str,
    output_column: str,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    inputs = run[input_column].to_numpy(dtype=float)
    measured = run[output_column].to_numpy(dtype=float)
    first = max(1, start)
    if first >= stop:
        return np.asarray([]), np.asarray([])
    prediction = float(measured[first - 1])
    predicted: list[float] = []
    target: list[float] = []
    for position in range(first, stop):
        delayed_position = max(0, position - model.delay_steps)
        prediction = model.next_output(prediction, float(inputs[delayed_position]))
        predicted.append(prediction)
        target.append(float(measured[position]))
    return np.asarray(target), np.asarray(predicted)


def _cross_validation_rmse(
    data: pd.DataFrame,
    input_column: str,
    output_column: str,
    delay_steps: int,
    structure: str,
    fold_kind: str,
) -> float:
    residuals: list[np.ndarray] = []
    if fold_kind == "segment":
        run = data.sort_values("run_time_s")
        for segment_index, segment in run.groupby("segment_index", sort=True):
            excluded = set(segment.index.astype(int))
            model = _fit_model(
                data, input_column, output_column, delay_steps, structure, excluded_rows=excluded
            )
            positions = np.flatnonzero(run["segment_index"].to_numpy() == segment_index)
            measured, predicted = _simulate_interval(
                run,
                model,
                input_column,
                output_column,
                int(positions[0]),
                int(positions[-1]) + 1,
            )
            residuals.append(measured - predicted)
    elif fold_kind == "run":
        for run_index, run in data.groupby("run_index", sort=True):
            excluded = set(run.index.astype(int))
            model = _fit_model(
                data, input_column, output_column, delay_steps, structure, excluded_rows=excluded
            )
            measured, predicted = _simulate_interval(
                run.sort_values("run_time_s"),
                model,
                input_column,
                output_column,
                0,
                len(run),
            )
            residuals.append(measured - predicted)
    else:
        raise ValueError(f"Unknown fold kind: {fold_kind}")
    residual = np.concatenate([values for values in residuals if len(values)])
    return float(np.sqrt(np.mean(np.square(residual))))


def _metric(measured: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int]:
    measured_array = np.asarray(list(measured), dtype=float)
    predicted_array = np.asarray(list(predicted), dtype=float)
    valid = np.isfinite(measured_array) & np.isfinite(predicted_array)
    measured_array = measured_array[valid]
    predicted_array = predicted_array[valid]
    residual = measured_array - predicted_array
    denominator = float(np.sum(np.square(measured_array - np.mean(measured_array))))
    return {
        "samples": int(len(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias_measured_minus_predicted": float(np.mean(residual)),
        "max_abs_error": float(np.max(np.abs(residual))),
        "r2": float(1.0 - np.sum(np.square(residual)) / denominator)
        if denominator > 1.0e-16
        else float("nan"),
    }


def _candidate_table(
    data: pd.DataFrame,
    input_column: str,
    output_column: str,
    max_delay_steps: int,
    fold_kind: str,
) -> tuple[FirstOrderDelayModel, list[Candidate]]:
    candidates: list[Candidate] = []
    for structure in ("unity_gain", "free_gain"):
        for delay_steps in range(max_delay_steps + 1):
            model = _fit_model(data, input_column, output_column, delay_steps, structure)
            measured_one, predicted_one = _one_step_predictions(
                data, model, input_column, output_column
            )
            simulated = _simulate_runs(data, model, input_column, output_column)
            candidates.append(
                Candidate(
                    structure=structure,
                    delay_steps=delay_steps,
                    a=model.a,
                    b=model.b,
                    dc_gain=model.dc_gain,
                    time_constant_s=model.time_constant_s,
                    cv_simulation_rmse=_cross_validation_rmse(
                        data,
                        input_column,
                        output_column,
                        delay_steps,
                        structure,
                        fold_kind,
                    ),
                    identification_one_step_rmse=float(
                        np.sqrt(np.mean(np.square(measured_one - predicted_one)))
                    ),
                    identification_simulation_rmse=_metric(
                        data[output_column], simulated
                    )["rmse"],
                    parameter_count=1 if structure == "unity_gain" else 2,
                )
            )
    minimum = min(candidate.cv_simulation_rmse for candidate in candidates)
    simple_candidates = [
        candidate
        for candidate in candidates
        if candidate.cv_simulation_rmse <= minimum * 1.01
    ]
    selected = min(
        simple_candidates,
        key=lambda candidate: (
            candidate.parameter_count,
            candidate.delay_steps,
            candidate.cv_simulation_rmse,
        ),
    )
    model = FirstOrderDelayModel(
        float(data["step_dt_s"].iloc[0]),
        selected.delay_steps,
        selected.a,
        selected.b,
    )
    return model, candidates


def _region_rows(
    data: pd.DataFrame,
    input_column: str,
    delay_steps: int,
    lower: float,
    upper: float,
    is_last: bool,
) -> set[int]:
    rows: set[int] = set()
    for _, run in data.groupby("run_index", sort=True):
        run = run.sort_values("run_time_s")
        row_ids = run.index.to_numpy()
        inputs = run[input_column].to_numpy(dtype=float)
        for position in range(max(1, delay_steps), len(run)):
            value = float(inputs[position - delay_steps])
            in_region = lower <= value < upper
            if is_last and math.isclose(value, upper, abs_tol=1.0e-6):
                in_region = True
            if in_region:
                rows.add(int(row_ids[position]))
    return rows


def _fit_speed_schedule(
    data: pd.DataFrame,
    boundaries: list[float],
    delay_steps: int,
    excluded_rows: set[int] | None = None,
) -> GainScheduledFirstOrderModel:
    regions: list[FirstOrderRegion] = []
    excluded_rows = excluded_rows or set()
    for index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        selected_rows = _region_rows(
            data,
            "executed_speed_m_s",
            delay_steps,
            lower,
            upper,
            index == len(boundaries) - 2,
        )
        model = _fit_model(
            data,
            "executed_speed_m_s",
            "actual_body_vx_m_s",
            delay_steps,
            "free_gain",
            excluded_rows=excluded_rows,
            selected_rows=selected_rows,
        )
        regions.append(FirstOrderRegion(lower, upper, model))
    return GainScheduledFirstOrderModel(tuple(regions))


def _scheduled_one_step(
    data: pd.DataFrame,
    model: GainScheduledFirstOrderModel,
) -> tuple[np.ndarray, np.ndarray]:
    measured: list[float] = []
    predicted: list[float] = []
    for _, run in data.groupby("run_index", sort=True):
        run = run.sort_values("run_time_s")
        inputs = run["executed_speed_m_s"].to_numpy(dtype=float)
        outputs = run["actual_body_vx_m_s"].to_numpy(dtype=float)
        for position in range(max(1, model.delay_steps), len(run)):
            delayed = float(inputs[position - model.delay_steps])
            measured.append(float(outputs[position]))
            predicted.append(model.next_output(float(outputs[position - 1]), delayed))
    return np.asarray(measured), np.asarray(predicted)


def _scheduled_cv_rmse(
    data: pd.DataFrame,
    boundaries: list[float],
    delay_steps: int,
) -> float:
    run = data.sort_values("run_time_s")
    residuals: list[np.ndarray] = []
    for segment_index, segment in run.groupby("segment_index", sort=True):
        model = _fit_speed_schedule(
            data,
            boundaries,
            delay_steps,
            excluded_rows=set(segment.index.astype(int)),
        )
        positions = np.flatnonzero(run["segment_index"].to_numpy() == segment_index)
        measured, predicted = _simulate_interval(
            run,
            model,
            "executed_speed_m_s",
            "actual_body_vx_m_s",
            int(positions[0]),
            int(positions[-1]) + 1,
        )
        residuals.append(measured - predicted)
    residual = np.concatenate([values for values in residuals if len(values)])
    return float(np.sqrt(np.mean(np.square(residual))))


def _speed_schedule_table(
    data: pd.DataFrame,
    max_delay_steps: int,
) -> tuple[GainScheduledFirstOrderModel, list[dict[str, Any]], dict[str, Any]]:
    schedule_definitions = (
        ("one_region", [0.0, 1.2]),
        ("two_regions", [0.0, 0.75, 1.2]),
        ("three_regions", [0.0, 0.45, 1.05, 1.2]),
        ("four_regions", [0.0, 0.45, 0.75, 1.05, 1.2]),
    )
    candidates: list[dict[str, Any]] = []
    fitted_models: dict[tuple[str, int], GainScheduledFirstOrderModel] = {}
    for name, boundaries in schedule_definitions:
        for delay_steps in range(max_delay_steps + 1):
            try:
                model = _fit_speed_schedule(data, boundaries, delay_steps)
                measured_one, predicted_one = _scheduled_one_step(data, model)
                simulation = _simulate_runs(
                    data, model, "executed_speed_m_s", "actual_body_vx_m_s"
                )
                residual = measured_one - predicted_one
                row = {
                    "name": name,
                    "boundaries_m_s": "|".join(str(value) for value in boundaries),
                    "region_count": len(boundaries) - 1,
                    "delay_steps": delay_steps,
                    "parameter_count": 2 * (len(boundaries) - 1),
                    "cv_simulation_rmse": _scheduled_cv_rmse(
                        data, boundaries, delay_steps
                    ),
                    "identification_one_step_rmse": float(
                        np.sqrt(np.mean(np.square(residual)))
                    ),
                    "identification_simulation_rmse": _metric(
                        data["actual_body_vx_m_s"], simulation
                    )["rmse"],
                    "bic": _bic(residual, 2 * (len(boundaries) - 1)),
                }
            except ValueError:
                continue
            candidates.append(row)
            fitted_models[(name, delay_steps)] = model
    minimum_bic = min(float(row["bic"]) for row in candidates)
    eligible = [
        row for row in candidates if float(row["bic"]) <= minimum_bic + 2.0
    ]
    selected = min(
        eligible,
        key=lambda row: (
            int(row["parameter_count"]),
            int(row["delay_steps"]),
            float(row["bic"]),
        ),
    )
    model = fitted_models[(str(selected["name"]), int(selected["delay_steps"]))]
    return model, candidates, selected


def _bic(residual: np.ndarray, parameter_count: int) -> float:
    count = len(residual)
    rss = max(float(np.sum(np.square(residual))), 1.0e-30)
    return count * math.log(rss / count) + parameter_count * math.log(count)


def _group_metrics(
    data: pd.DataFrame,
    measured_column: str,
    predicted_column: str,
    groups: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for keys, group in data.groupby(groups, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {name: value.item() if hasattr(value, "item") else value for name, value in zip(groups, keys)}
        row.update(_metric(group[measured_column], group[predicted_column]))
        rows.append(row)
    return rows


def _transition_labels(data: pd.DataFrame) -> pd.Series:
    labels = pd.Series("hold", index=data.index, dtype=object)
    for _, run in data.groupby("run_index", sort=True):
        segment_targets = (
            run.groupby("segment_index", sort=True)["requested_target_speed_m_s"].first()
        )
        previous = None
        for segment_index, target in segment_targets.items():
            if previous is not None:
                if target > previous + 1.0e-9:
                    labels.loc[run.index[run["segment_index"] == segment_index]] = "acceleration"
                elif target < previous - 1.0e-9:
                    labels.loc[run.index[run["segment_index"] == segment_index]] = "deceleration"
            previous = target
    return labels


def _piecewise_evidence(
    data: pd.DataFrame,
    model: FirstOrderDelayModel,
    input_column: str,
    output_column: str,
    groups: pd.Series,
    selected_groups: tuple[str, ...],
) -> dict[str, Any]:
    selected_rows = set(data.index[groups.isin(selected_groups)].astype(int))
    previous, delayed, target = _sample_arrays(
        data,
        input_column,
        output_column,
        model.delay_steps,
        selected_rows=selected_rows,
    )
    unified_residual = target - (model.a * previous + model.b * delayed)
    separate_residuals: list[np.ndarray] = []
    separate_parameters: dict[str, Any] = {}
    for group_name in selected_groups:
        row_ids = set(data.index[groups == group_name].astype(int))
        branch = _fit_model(
            data,
            input_column,
            output_column,
            model.delay_steps,
            "free_gain",
            selected_rows=row_ids,
        )
        branch_previous, branch_input, branch_target = _sample_arrays(
            data,
            input_column,
            output_column,
            model.delay_steps,
            selected_rows=row_ids,
        )
        separate_residuals.append(
            branch_target - (branch.a * branch_previous + branch.b * branch_input)
        )
        separate_parameters[group_name] = {
            "a": branch.a,
            "b": branch.b,
            "dc_gain": branch.dc_gain,
            "time_constant_s": branch.time_constant_s,
        }
    separate_residual = np.concatenate(separate_residuals)
    unified_bic = _bic(unified_residual, 2)
    separate_bic = _bic(separate_residual, 2 * len(selected_groups))
    return {
        "groups": separate_parameters,
        "unified_bic": unified_bic,
        "piecewise_bic": separate_bic,
        "delta_bic_unified_minus_piecewise": unified_bic - separate_bic,
        "piecewise_supported_by_bic": bool(unified_bic - separate_bic > 10.0),
        "note": "Diagnostic only; the frozen nominal model remains unified.",
    }


def _fit_yaw_gain(data: pd.DataFrame, wheelbase_m: float) -> tuple[float, dict[str, Any]]:
    base = (
        data["actual_body_vx_m_s"].to_numpy(dtype=float)
        * np.tan(data["actual_equivalent_center_steering_rad"].to_numpy(dtype=float))
        / wheelbase_m
    )
    measured = data["actual_yaw_rate_rad_s"].to_numpy(dtype=float)
    fitted_gain = float(np.dot(base, measured) / np.dot(base, base))
    unit_residual = measured - base
    gain_residual = measured - fitted_gain * base

    cv_unit: list[np.ndarray] = []
    cv_gain: list[np.ndarray] = []
    run_values = data["run_index"].to_numpy()
    for run_index in sorted(data["run_index"].unique()):
        train = run_values != run_index
        test = ~train
        fold_gain = float(np.dot(base[train], measured[train]) / np.dot(base[train], base[train]))
        cv_unit.append(measured[test] - base[test])
        cv_gain.append(measured[test] - fold_gain * base[test])
    cv_unit_rmse = float(np.sqrt(np.mean(np.square(np.concatenate(cv_unit)))))
    cv_gain_rmse = float(np.sqrt(np.mean(np.square(np.concatenate(cv_gain)))))
    delta_bic = _bic(unit_residual, 0) - _bic(gain_residual, 1)
    improvement = (cv_unit_rmse - cv_gain_rmse) / cv_unit_rmse
    selected = bool(delta_bic > 10.0 and improvement >= 0.02)
    gain = fitted_gain if selected else 1.0
    return gain, {
        "unit_gain_rmse": float(np.sqrt(np.mean(np.square(unit_residual)))),
        "fitted_gain": fitted_gain,
        "fitted_gain_rmse": float(np.sqrt(np.mean(np.square(gain_residual)))),
        "leave_one_speed_run_out_unit_rmse": cv_unit_rmse,
        "leave_one_speed_run_out_gain_rmse": cv_gain_rmse,
        "cross_validation_relative_improvement": improvement,
        "delta_bic_unit_minus_gain": delta_bic,
        "empirical_gain_selected": selected,
        "selection_rule": "delta BIC > 10 and blocked-CV RMSE improvement >= 2%",
    }


def _add_predictions(
    data: pd.DataFrame,
    model: NominalCar4WDModel,
) -> pd.DataFrame:
    predictions = data.copy()
    predictions["predicted_body_vx_m_s"] = np.nan
    speed_rows = predictions["profile_type"] == "speed_step"
    predictions.loc[speed_rows, "predicted_body_vx_m_s"] = _simulate_runs(
        predictions.loc[speed_rows],
        model.speed,
        "executed_speed_m_s",
        "actual_body_vx_m_s",
    )
    predictions["predicted_equivalent_center_steering_rad"] = np.nan
    steering_rows = predictions["profile_type"] == "steering_step"
    predictions.loc[steering_rows, "predicted_equivalent_center_steering_rad"] = _simulate_runs(
        predictions.loc[steering_rows],
        model.steering,
        "executed_steering_rad",
        "actual_equivalent_center_steering_rad",
    )
    base_yaw = (
        predictions["actual_body_vx_m_s"]
        * np.tan(predictions["actual_equivalent_center_steering_rad"])
        / model.wheelbase_m
    )
    predictions["predicted_yaw_rate_unit_gain_rad_s"] = base_yaw
    predictions["predicted_yaw_rate_rad_s"] = model.yaw_gain * base_yaw
    return predictions


def _plot_predictions(data: pd.DataFrame, path: Path, title: str) -> None:
    steering_runs = list(data[data["profile_type"] == "steering_step"].groupby("run_index"))
    figure, axes = plt.subplots(2 + len(steering_runs), 1, figsize=(12, 3.1 * (2 + len(steering_runs))))
    speed = data[data["profile_type"] == "speed_step"]
    axes[0].plot(speed["run_time_s"], speed["executed_speed_m_s"], label="executed input", linewidth=1)
    axes[0].plot(speed["run_time_s"], speed["actual_body_vx_m_s"], label="measured", linewidth=1)
    axes[0].plot(speed["run_time_s"], speed["predicted_body_vx_m_s"], label="predicted", linewidth=1)
    axes[0].set_ylabel("speed [m/s]")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.25)
    for axis, (_, run) in zip(axes[1:-1], steering_runs):
        axis.plot(run["run_time_s"], run["executed_steering_rad"], label="executed input", linewidth=1)
        axis.plot(run["run_time_s"], run["actual_equivalent_center_steering_rad"], label="measured", linewidth=1)
        axis.plot(
            run["run_time_s"],
            run["predicted_equivalent_center_steering_rad"],
            label="predicted",
            linewidth=1,
        )
        axis.set_ylabel("steer [rad]")
        axis.set_title(str(run["profile_name"].iloc[0]))
        axis.grid(alpha=0.25)
    steering = data[data["profile_type"] == "steering_step"]
    axes[-1].scatter(
        steering["predicted_yaw_rate_unit_gain_rad_s"],
        steering["actual_yaw_rate_rad_s"],
        s=5,
        alpha=0.35,
        label="unit gain",
    )
    axes[-1].scatter(
        steering["predicted_yaw_rate_rad_s"],
        steering["actual_yaw_rate_rad_s"],
        s=5,
        alpha=0.35,
        label="selected model",
    )
    limit = float(np.max(np.abs(steering[["actual_yaw_rate_rad_s", "predicted_yaw_rate_rad_s"]].to_numpy())))
    axes[-1].plot([-limit, limit], [-limit, limit], color="black", linewidth=1)
    axes[-1].set_xlabel("predicted yaw rate [rad/s]")
    axes[-1].set_ylabel("measured yaw rate [rad/s]")
    axes[-1].legend()
    axes[-1].grid(alpha=0.25)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_residuals(data: pd.DataFrame, path: Path, title: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    plots = (
        ("actual_body_vx_m_s", "predicted_body_vx_m_s", "speed [m/s]"),
        (
            "actual_equivalent_center_steering_rad",
            "predicted_equivalent_center_steering_rad",
            "steering [rad]",
        ),
        ("actual_yaw_rate_rad_s", "predicted_yaw_rate_rad_s", "yaw rate [rad/s]"),
    )
    for axis, (measured, predicted, label) in zip(axes, plots):
        residual = data[measured] - data[predicted]
        residual = residual[np.isfinite(residual)]
        axis.hist(residual, bins=50)
        axis.axvline(float(residual.mean()), color="black", linewidth=1)
        axis.set_xlabel(f"{label} measured - predicted")
        axis.set_ylabel("samples")
        axis.grid(alpha=0.2)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _evaluate_predictions(data: pd.DataFrame) -> dict[str, Any]:
    speed = data[data["profile_type"] == "speed_step"].copy()
    speed["transition"] = _transition_labels(speed)
    steering = data[data["profile_type"] == "steering_step"].copy()
    steering["steering_sign"] = np.where(
        steering["requested_target_steering_rad"] > 1.0e-9,
        "left",
        np.where(steering["requested_target_steering_rad"] < -1.0e-9, "right", "zero"),
    )
    return {
        "speed": _metric(speed["actual_body_vx_m_s"], speed["predicted_body_vx_m_s"]),
        "speed_by_transition": _group_metrics(
            speed, "actual_body_vx_m_s", "predicted_body_vx_m_s", ["transition"]
        ),
        "steering": _metric(
            steering["actual_equivalent_center_steering_rad"],
            steering["predicted_equivalent_center_steering_rad"],
        ),
        "steering_by_sign_and_speed": _group_metrics(
            steering,
            "actual_equivalent_center_steering_rad",
            "predicted_equivalent_center_steering_rad",
            ["requested_target_speed_m_s", "steering_sign"],
        ),
        "yaw_unit_gain": _metric(
            steering["actual_yaw_rate_rad_s"],
            steering["predicted_yaw_rate_unit_gain_rad_s"],
        ),
        "yaw_selected_gain": _metric(
            steering["actual_yaw_rate_rad_s"], steering["predicted_yaw_rate_rad_s"]
        ),
        "yaw_by_sign_and_speed": _group_metrics(
            steering,
            "actual_yaw_rate_rad_s",
            "predicted_yaw_rate_rad_s",
            ["requested_target_speed_m_s", "steering_sign"],
        ),
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fit(args: argparse.Namespace) -> None:
    identification_path = args.data_dir.expanduser().resolve() / "identification.csv"
    steering_config_path = DEFAULT_STEERING_CONFIG.resolve()
    steering_config = _load_json(steering_config_path)
    action_scale = float(
        steering_config["normalized_action_interface"]["steering_scale_rad"]
    )
    if not math.isclose(action_scale, 0.24, abs_tol=1.0e-12):
        raise RuntimeError(
            f"Expected the frozen 0.24 rad steering scale, got {action_scale}."
        )
    identification = _load_csv(identification_path, "identification")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    speed_data = identification[identification["profile_type"] == "speed_step"].copy()
    steering_data = identification[identification["profile_type"] == "steering_step"].copy()
    _, speed_candidates = _candidate_table(
        speed_data,
        "executed_speed_m_s",
        "actual_body_vx_m_s",
        args.max_delay_steps,
        "segment",
    )
    speed_model, speed_schedule_candidates, selected_speed_schedule = (
        _speed_schedule_table(speed_data, args.max_delay_steps)
    )
    steering_model, steering_candidates = _candidate_table(
        steering_data,
        "executed_steering_rad",
        "actual_equivalent_center_steering_rad",
        args.max_delay_steps,
        "run",
    )
    wheelbase = float(steering_config["geometry"]["wheelbase_m"])
    yaw_gain, yaw_selection = _fit_yaw_gain(steering_data, wheelbase)
    model = NominalCar4WDModel(
        sample_time_s=0.05,
        wheelbase_m=wheelbase,
        speed=speed_model,
        steering=steering_model,
        yaw_gain=yaw_gain,
    )
    predictions = _add_predictions(identification, model)
    metrics = _evaluate_predictions(predictions)

    transition_metrics = {
        row["transition"]: row for row in metrics["speed_by_transition"]
    }
    acceleration_rmse = float(transition_metrics["acceleration"]["rmse"])
    deceleration_rmse = float(transition_metrics["deceleration"]["rmse"])
    speed_direction_diagnostic = {
        "acceleration_rmse_m_s": acceleration_rmse,
        "deceleration_rmse_m_s": deceleration_rmse,
        "rmse_ratio": max(acceleration_rmse, deceleration_rmse)
        / min(acceleration_rmse, deceleration_rmse),
        "direction_split_recommended": bool(
            max(acceleration_rmse, deceleration_rmse) > 0.05
            or max(acceleration_rmse, deceleration_rmse)
            / min(acceleration_rmse, deceleration_rmse)
            > 1.5
        ),
        "decision_rule": "recommend only if either RMSE exceeds 0.05 m/s or their ratio exceeds 1.5",
    }
    steering_sign = pd.Series(
        np.where(
            steering_data["requested_target_steering_rad"] > 1.0e-9,
            "left",
            np.where(
                steering_data["requested_target_steering_rad"] < -1.0e-9,
                "right",
                "zero",
            ),
        ),
        index=steering_data.index,
    )
    steering_piecewise = _piecewise_evidence(
        steering_data,
        steering_model,
        "executed_steering_rad",
        "actual_equivalent_center_steering_rad",
        steering_sign,
        ("left", "right"),
    )
    left_gain = float(steering_piecewise["groups"]["left"]["dc_gain"])
    right_gain = float(steering_piecewise["groups"]["right"]["dc_gain"])
    steering_piecewise["relative_dc_gain_difference"] = abs(left_gain - right_gain) / (
        0.5 * (left_gain + right_gain)
    )
    steering_piecewise["direction_split_recommended"] = bool(
        steering_piecewise["relative_dc_gain_difference"] > 0.05
    )
    steering_piecewise["decision_rule"] = (
        "BIC alone is insufficient; require more than 5% left/right DC-gain difference"
    )
    selected_steering = min(
        (
            candidate
            for candidate in steering_candidates
            if candidate.a == steering_model.a and candidate.b == steering_model.b
        ),
        key=lambda candidate: candidate.cv_simulation_rmse,
    )
    parameters = {
        "schema_version": 1,
        "model_name": "nominal_car4wd_first_order_ackermann",
        "sample_time_s": 0.05,
        "geometry": {
            "wheelbase_m": wheelbase,
            "track_width_m": float(steering_config["geometry"]["track_width_m"]),
        },
        "action_interface": {
            "speed_unit": "m/s",
            "steering_unit": "rad",
            "steering_scale_rad": action_scale,
            "steering_scale_status": "current frozen interface; 0.243 rad documents are historical",
        },
        "dynamics": {
            "longitudinal_speed": {
                "type": "gain_scheduled_first_order",
                "equation": "v[k] = a*v[k-1] + b*v_cmd[k-delay]",
                "input": "executed_speed_m_s",
                "output": "actual_body_vx_m_s",
                "schedule_variable": "delayed executed_speed_m_s",
                "selected_candidate": selected_speed_schedule,
                "regions": [
                    {
                        "input_range_m_s": [region.input_min, region.input_max],
                        "delay_steps": region.model.delay_steps,
                        "delay_s": region.model.delay_s,
                        "a": region.model.a,
                        "b": region.model.b,
                        "dc_gain": region.model.dc_gain,
                        "time_constant_s": region.model.time_constant_s,
                    }
                    for region in speed_model.regions
                ],
            },
            "equivalent_center_steering": {
                "type": "first_order",
                "equation": "delta[k] = a*delta[k-1] + b*delta_cmd[k-delay]",
                "input": "executed_steering_rad",
                "output": "actual_equivalent_center_steering_rad",
                "structure": selected_steering.structure,
                "delay_steps": steering_model.delay_steps,
                "delay_s": steering_model.delay_s,
                "a": steering_model.a,
                "b": steering_model.b,
                "dc_gain": steering_model.dc_gain,
                "time_constant_s": steering_model.time_constant_s,
            },
            "yaw_rate": {
                "equation": "r = empirical_gain * v_x * tan(delta) / wheelbase",
                "speed_input": "actual_body_vx_m_s",
                "steering_input": "actual_equivalent_center_steering_rad",
                "empirical_gain": yaw_gain,
                "gain_selection": yaw_selection,
            },
        },
        "selection": {
            "candidate_delay_steps": list(range(args.max_delay_steps + 1)),
            "structures": ["unity_gain", "free_gain"],
            "cross_validation": {
                "speed": "leave-one-segment-out simulation RMSE",
                "steering": "leave-one-speed-run-out simulation RMSE",
            },
            "speed_selection_rule": (
                "minimum identification BIC; within delta BIC 2 prefer fewer parameters then shorter delay"
            ),
            "steering_selection_rule": (
                "within 1% of minimum CV RMSE, prefer fewer parameters then shorter delay"
            ),
            "piecewise_diagnostics": {
                "speed_acceleration_deceleration": speed_direction_diagnostic,
                "speed_operating_regions": selected_speed_schedule,
                "steering_left_right": steering_piecewise,
            },
        },
        "identification": {
            "source_csv": str(identification_path),
            "source_sha256": _sha256(identification_path),
            "rows": int(len(identification)),
            "metrics": metrics,
        },
        "applicability": {
            "data_source": "Isaac Sim flat high-traction open-loop data; not real hardware",
            "speed_command_range_m_s": [
                float(speed_data["executed_speed_m_s"].min()),
                float(speed_data["executed_speed_m_s"].max()),
            ],
            "steering_command_range_rad": [
                float(steering_data["executed_steering_rad"].min()),
                float(steering_data["executed_steering_rad"].max()),
            ],
            "steering_identification_speeds_m_s": sorted(
                float(value)
                for value in steering_data["requested_target_speed_m_s"].unique()
            ),
            "surface": "flat fixed high traction",
            "extrapolation_allowed": False,
        },
        "provenance": {
            "base_commit": "9e581b7ceb4be3abc4aba35ee2a2e2ec652560e0",
            "steering_config": str(steering_config_path),
            "steering_config_sha256": _sha256(steering_config_path),
            "validation_used_for_parameter_selection": False,
        },
    }

    parameter_output = args.parameter_output.expanduser().resolve()
    _write_json(parameter_output, parameters)
    _write_json(output_dir / "frozen_parameters.json", parameters)
    _write_json(
        output_dir / "identification_metrics.json",
        {
            "metrics": metrics,
            "yaw_gain_selection": yaw_selection,
            "piecewise_diagnostics": {
                "speed_acceleration_deceleration": speed_direction_diagnostic,
                "speed_operating_regions": selected_speed_schedule,
                "steering_left_right": steering_piecewise,
            },
        },
    )
    pd.DataFrame([asdict(candidate) for candidate in speed_candidates]).to_csv(
        output_dir / "speed_candidates.csv", index=False
    )
    pd.DataFrame(speed_schedule_candidates).to_csv(
        output_dir / "speed_schedule_candidates.csv", index=False
    )
    pd.DataFrame([asdict(candidate) for candidate in steering_candidates]).to_csv(
        output_dir / "steering_candidates.csv", index=False
    )
    predictions.to_csv(output_dir / "identification_predictions.csv", index=False)
    _plot_predictions(
        predictions,
        output_dir / "identification_prediction_curves.png",
        "Nominal Car4WD identification",
    )
    _plot_residuals(
        predictions,
        output_dir / "identification_residuals.png",
        "Nominal Car4WD identification residuals",
    )
    print(json.dumps({"parameters": str(parameter_output), "metrics": metrics}, indent=2))


def validate(args: argparse.Namespace) -> None:
    validation_path = args.data_dir.expanduser().resolve() / "validation.csv"
    parameters_path = args.parameters.expanduser().resolve()
    parameters = _load_json(parameters_path)
    if parameters["provenance"].get("validation_used_for_parameter_selection") is not False:
        raise RuntimeError("Parameter provenance does not certify an identification-only fit.")
    model = NominalCar4WDModel.from_mapping(parameters)
    validation = _load_csv(validation_path, "validation")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    predictions = _add_predictions(validation, model)
    metrics = _evaluate_predictions(predictions)
    result = {
        "parameters": str(parameters_path),
        "parameters_sha256": _sha256(parameters_path),
        "validation_csv": str(validation_path),
        "validation_sha256": _sha256(validation_path),
        "validation_rows": int(len(validation)),
        "parameters_refit_or_selected_on_validation": False,
        "metrics": metrics,
    }
    _write_json(output_dir / "validation_metrics.json", result)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    _plot_predictions(
        predictions,
        output_dir / "validation_prediction_curves.png",
        "Nominal Car4WD independent validation",
    )
    _plot_residuals(
        predictions,
        output_dir / "validation_residuals.png",
        "Nominal Car4WD independent validation residuals",
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    args = _parse_args()
    if args.command == "fit":
        fit(args)
    else:
        validate(args)


if __name__ == "__main__":
    main()
