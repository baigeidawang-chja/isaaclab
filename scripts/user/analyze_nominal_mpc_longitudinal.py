"""Reproducible longitudinal tables/figures from real eval_nominal_mpc CSVs only."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_nominal_mpc import _read_run, _stage_metrics

WHEELS = ("front_left", "front_right", "rear_left", "rear_right")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def last_window(frame, seconds=1.0):
    end = float(frame.sample_time_s.iloc[-1])
    # Round decimal timestamps before the strict open-left comparison.
    return frame[frame.sample_time_s.round(9) > round(end - seconds, 9)]


def confirmed_crossing(frame, column, sign, time_column):
    crossed = frame[column].to_numpy() * sign > 0.02
    for i in range(len(crossed) - 2):
        if crossed[i:i + 3].all():
            return float(frame[time_column].iloc[i] - frame.command_time_s.iloc[0])
    return None


def stage_metrics(frame, previous_speed, expected_samples, nominal_min, nominal_max):
    frame = frame.copy()
    v = float(frame.desired_speed_m_s.iloc[0])
    origin = float(frame.command_time_s.iloc[0])
    end = float(frame.sample_time_s.iloc[-1])
    frame["speed_error"] = v - frame.actual_body_vx_m_s
    previous = None if previous_speed is None else (previous_speed, 0.0)
    result = _stage_metrics(frame.to_dict("records"), previous, 0.05, 0.05, 1.0,
                            1.5, nominal_max, -1.5, nominal_min)
    tail = last_window(frame)
    e = frame.speed_error.to_numpy()
    result.update(
        stage_start_s=origin, stage_end_s=end, previous_speed_m_s=previous_speed,
        complete_stage=len(frame) == expected_samples, expected_samples=expected_samples,
        steady_samples=len(tail), steady_actual_body_speed_m_s=float(tail.actual_body_vx_m_s.mean()),
        steady_speed_bias_m_s=float(tail.speed_error.mean()),
        heading_max_abs_rad=float(frame.actual_heading_error_rad.abs().max()),
        pre_switch_body_speed_m_s=float(frame.pre_body_speed_m_s.iloc[0]),
        pre_switch_executed_speed_m_s=float(frame.previous_executed_speed_m_s.iloc[0]),
        pre_switch_heading_rad=float(frame.pre_heading_rad.iloc[0]),
        speed_rate_limit_fraction=float((np.abs(frame.executed_speed_m_s - frame.previous_executed_speed_m_s) >= 0.1 - 1e-6).mean()),
        speed_rate_limit_max_delta_m_s=float(np.abs(frame.executed_speed_m_s - frame.previous_executed_speed_m_s).max()),
    )
    tail_count = 0
    for error in e[::-1]:
        if abs(error) > 0.05:
            break
        tail_count += 1
    result["trailing_in_band_samples"] = tail_count
    result["trailing_in_band_span_s"] = max(0.0, (tail_count - 1) * float(frame.step_dt_s.iloc[0]))
    result["settled_within_observation"] = tail_count > 0
    for label, subset in (("whole", frame), ("steady", tail)):
        upper = subset.mpc_speed_target_m_s >= 1.5 - 1e-4
        lower = subset.mpc_speed_target_m_s <= -1.5 + 1e-4
        result[f"{label}_upper_saturation_fraction"] = float(upper.mean())
        result[f"{label}_lower_saturation_fraction"] = float(lower.mean())
        result[f"{label}_saturation_fraction"] = float((upper | lower).mean())
    projection = {"actual_world_vx_m_s", "actual_world_vy_m_s"} <= set(frame.columns)
    result["world_projection_available"] = projection
    result["projected_speed_mae_m_s"] = None
    result["projected_speed_rmse_m_s"] = None
    if projection:
        projected = frame.actual_world_vx_m_s * np.cos(frame.desired_heading_rad) + frame.actual_world_vy_m_s * np.sin(frame.desired_heading_rad)
        result["projected_speed_mae_m_s"] = float(np.abs(v - projected).mean())
        result["projected_speed_rmse_m_s"] = float(np.sqrt(np.mean((v - projected) ** 2)))
        result["projection_vs_world_vx_max_m_s"] = float(np.abs(projected - frame.actual_world_vx_m_s).max())
    result["executed_crossing_time_s"] = None
    result["actual_crossing_time_s"] = None
    result["vehicle_after_input_crossing_delay_s"] = None
    if previous_speed is not None and previous_speed * v < 0:
        sign = math.copysign(1, v)
        target_cross = confirmed_crossing(frame, "executed_speed_m_s", sign, "command_time_s")
        actual_cross = confirmed_crossing(frame, "actual_body_vx_m_s", sign, "sample_time_s")
        result.update(executed_crossing_time_s=target_cross, actual_crossing_time_s=actual_cross)
        if target_cross is not None and actual_cross is not None:
            result["vehicle_after_input_crossing_delay_s"] = actual_cross - target_cross
    result["stop_tail_signed_world_x_drift_m"] = None
    result["stop_tail_path_length_m"] = None
    if v == 0 and "actual_world_x_m" in frame:
        x = np.r_[tail.pre_world_x_m.iloc[0], tail.actual_world_x_m]
        y = np.r_[tail.pre_world_y_m.iloc[0], tail.actual_world_y_m]
        result["stop_tail_signed_world_x_drift_m"] = float(x[-1] - x[0])
        result["stop_tail_path_length_m"] = float(np.hypot(np.diff(x), np.diff(y)).sum())
    for wheel in WHEELS:
        target = tail[f"target_wheel_speed_{wheel}_rad_s"]
        actual = tail[f"actual_wheel_speed_{wheel}_rad_s"]
        computed = tail[f"computed_wheel_torque_{wheel}_nm"]
        applied = tail[f"applied_wheel_torque_{wheel}_nm"]
        result[f"steady_{wheel}_target_rad_s"] = float(target.mean())
        result[f"steady_{wheel}_actual_rad_s"] = float(actual.mean())
        result[f"steady_{wheel}_error_mae_rad_s"] = float((target - actual).abs().mean())
        result[f"steady_{wheel}_computed_abs_nm"] = float(computed.abs().mean())
        result[f"steady_{wheel}_applied_abs_nm"] = float(applied.abs().mean())
        result[f"steady_{wheel}_torque_clip_fraction"] = float((computed.abs() > applied.abs() + 1e-4).mean())
        result[f"steady_{wheel}_sign_correct_fraction"] = float((actual * np.sign(v) > 0).mean()) if v else None
    for side in ("front_left", "front_right"):
        result[f"steady_actual_steer_{side}_rad"] = float(tail[f"actual_steering_{side}_rad"].mean())
    last_two = last_window(frame, 2.0)
    slope = float(np.polyfit(last_two.sample_time_s, last_two.actual_body_vx_m_s, 1)[0])
    before_tail = last_two[last_two.sample_time_s.round(9) <= round(end - 1, 9)]
    difference = float(tail.actual_body_vx_m_s.mean() - before_tail.actual_body_vx_m_s.mean()) if len(before_tail) else 0.0
    result["last_2s_speed_slope_m_s2"] = slope
    result["last_two_1s_mean_difference_m_s"] = difference
    result["extension_trigger"] = abs(v - 0.8) < 1e-9 and (abs(slope) > 0.01 or abs(difference) > 0.02)
    return result


def timing(frame):
    milliseconds = frame.solve_time_s.to_numpy() * 1000
    return {"solve_median_ms": float(np.median(milliseconds)), "solve_p95_ms": float(np.percentile(milliseconds, 95)),
            "solve_max_ms": float(milliseconds.max()), "solve_over_50ms": int((milliseconds > 50).sum()),
            "statuses": dict(Counter(frame.solver_status)), "fallbacks": int(frame.fallback_used.sum())}


def read_metrics(path):
    meta, model, config, rows = _read_run(path, verify_live_asset=False)
    frame = pd.DataFrame(rows)
    for column in frame:
        if column not in ("data_source", "task", "profile", "solver_status", "fallback_reason", "termination_reason"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
    valid = (frame.terminated == 0) & (frame.truncated == 0)
    if "response_sample_valid" in frame:
        valid &= frame.response_sample_valid == 1
    # Stop at the first reset, even if a future collector accidentally continues.
    invalid = np.flatnonzero(~valid.to_numpy())
    response = frame.iloc[:invalid[0]] if len(invalid) else frame
    regions = model["dynamics"]["longitudinal_speed"]["regions"]
    lo, hi = config["constraints"]["effective_mpc_control_bounds"]["speed_target_m_s"]
    nominal_min = min(r["dc_gain"] * max(lo, r["input_range_m_s"][0]) for r in regions)
    nominal_max = max(r["dc_gain"] * min(hi, r["input_range_m_s"][1]) for r in regions)
    stages = []
    previous = None
    for index, part in response.groupby("stage_index", sort=True):
        expected_start = index * meta["segment_duration_s"]
        expected_end = meta["duration_s"] if index == len(meta["commands"]) - 1 else (index + 1) * meta["segment_duration_s"]
        stage = stage_metrics(part, previous, round((expected_end - expected_start) / meta["step_dt_s"]), nominal_min, nominal_max)
        stage["case"] = path.name
        stage["run_dir"] = str(path.parent.name + "/" + path.name)
        stages.append(stage)
        previous = stage["desired_speed_m_s"]
    summary = {"case": path.name, "samples": len(frame), "valid_response_samples": len(response),
               "excluded_reset_samples": len(frame) - len(response), **timing(frame),
               "termination_reason": meta["termination_reason"],
               "max_speed_mapping_error_m_s": float((frame.mpc_speed_target_m_s - frame.physical_target_speed_m_s).abs().max()),
               "max_speed_limiter_adjustment_m_s": float((frame.physical_target_speed_m_s - frame.executed_speed_m_s).abs().max()),
               "max_post_pre_speed_alignment_error_m_s": float(np.abs(response.pre_body_speed_m_s.to_numpy()[1:] - response.actual_body_vx_m_s.to_numpy()[:-1]).max()) if len(response) > 1 else 0.0,
               "max_previous_executed_alignment_error_m_s": float(np.abs(response.previous_executed_speed_m_s.to_numpy()[1:] - response.executed_speed_m_s.to_numpy()[:-1]).max()) if len(response) > 1 else 0.0,
               "all_numeric_finite": bool(np.isfinite(response.select_dtypes(include="number")).all().all()),
               "stages": stages}
    # Reachability was read from this run\'s frozen model snapshot.
    save_json(path / "longitudinal_analysis.json", summary)
    pd.DataFrame(stages).to_csv(path / "longitudinal_stages.csv", index=False)
    return meta, frame, summary


def smoke_comparison(root, frames, metas):
    reference = frames["B1"]
    columns = ["desired_speed_m_s", "desired_heading_rad", "actual_body_vx_m_s", "actual_heading_rad",
               "executed_speed_m_s", "executed_steering_rad", "mpc_speed_target_m_s"]
    comparisons = []
    for name in ("B2", "B3"):
        differences = (frames[name][columns] - reference[columns]).abs().max().to_dict()
        initial = {key: float(np.max(np.abs(np.array(metas[name]["initial_state"][key]) - np.array(metas["B1"]["initial_state"][key]))))
                   for key in ("root_state_w", "joint_pos_rad", "joint_vel_rad_s", "executed_action")}
        comparisons.append({"comparison": f"B1/{name}", "seed": metas[name]["seed"], "initial_max_differences": initial,
                            "trajectory_max_differences": differences, "technical_repeat": max(differences.values()) == 0.0 and max(initial.values()) == 0.0,
                            "timing_B1": timing(reference), "timing_repeat": timing(frames[name])})
    save_json(root / "analysis/smoke_reproducibility.json", comparisons)


def draw(root, frames, stages):
    out = root / "analysis/figures"
    out.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.22})

    def finish(fig, filename):
        fig.tight_layout()
        fig.savefig(out / (filename + ".png"), dpi=180)
        fig.savefig(out / (filename + ".pdf"))
        plt.close(fig)

    fixed = stages[stages.case.str.startswith("F")]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([-0.6, 0.9], [-0.6, 0.9], "--", color="0.5", label="Identity")
    ax.scatter(fixed.desired_speed_m_s, fixed.steady_actual_body_speed_m_s, c="#007d77", zorder=3)
    for row in fixed.itertuples():
        ax.annotate(row.case, (row.desired_speed_m_s, row.steady_actual_body_speed_m_s), xytext=(5, -12), textcoords="offset points")
    ax.set(xlabel="Reference speed (m/s)", ylabel="Measured body speed, final 1 s mean (m/s)", title="Fixed-speed runs: one run per operating point", xlim=(-0.6, 0.9), ylim=(-0.6, 0.9))
    ax.legend()
    finish(fig, "01_fixed_speed")

    fig, ax = plt.subplots(figsize=(11, 4.2))
    f = frames["S1"]
    for column, label, color, style, time_column in [
        ("desired_speed_m_s", "Reference", "black", "--", "command_time_s"),
        ("mpc_speed_target_m_s", "MPC physical target", "#ba4260", "-", "command_time_s"),
        ("executed_speed_m_s", "Rate-limited target", "#c08a00", ":", "command_time_s"),
        ("actual_body_vx_m_s", "Measured body speed", "#007d77", "-", "sample_time_s")]:
        ax.plot(f[time_column], f[column], style, color=color, label=label, linewidth=1.5)
    ax.set(xlabel="Time (s)", ylabel="Speed (m/s)", title="S1: increasing and decreasing references")
    ax.legend(ncol=2)
    finish(fig, "02_S1_tracking")

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    for name, color in (("S2a", "#007d77"), ("S2b", "#ba4260")):
        f = frames[name].query("stage_index == 2")
        t = f.sample_time_s - 16
        axes[0].plot(t, f.actual_body_vx_m_s, color=color, label=name + " measured")
        axes[0].plot(f.command_time_s - 16, f.mpc_speed_target_m_s, ":", color=color, label=name + " MPC target")
        axes[1].step(f.command_time_s - 16, (f.mpc_speed_target_m_s >= 1.5 - 1e-4).astype(float) + (0 if name == "S2a" else 1.5), where="post", color=color, label=name)
    axes[0].axhline(0.8, color="black", linestyle="--", label="Reference = 0.8 m/s")
    axes[0].axhspan(0.75, 0.85, color="0.5", alpha=0.1)
    axes[0].set(ylabel="Speed (m/s)", title="S2: different entry conditions (not an isolated history effect)")
    axes[0].legend(ncol=2)
    axes[1].set(xlabel="Time since reference switch (s)", ylabel="Upper-bound hit", yticks=[0, 1, 1.5, 2.5], yticklabels=["a: 0", "a: 1", "b: 0", "b: 1"])
    finish(fig, "03_S2_entry_conditions")

    fig, axes = plt.subplots(3, 2, figsize=(13, 8), sharex="col")
    colors = ["#007d77", "#ba4260", "#c08a00", "#4e66a0"]
    for col, name in enumerate(("S3", "S4")):
        f = frames[name]
        for field, label, style in (("desired_speed_m_s", "Reference", "--"), ("executed_speed_m_s", "Executed target", ":"), ("actual_body_vx_m_s", "Body speed", "-")):
            axes[0, col].plot(f.command_time_s if field != "actual_body_vx_m_s" else f.sample_time_s, f[field], style, label=label)
        axes[0, col].set(title=name, ylabel="Speed (m/s)")
        for wheel, color in zip(WHEELS, colors):
            axes[1, col].plot(f.sample_time_s, f[f"actual_wheel_speed_{wheel}_rad_s"], color=color, label=wheel.replace("_", " "))
            axes[1, col].plot(f.command_time_s, f[f"target_wheel_speed_{wheel}_rad_s"], ":", color=color, alpha=.6)
            axes[2, col].plot(f.sample_time_s, f[f"applied_wheel_torque_{wheel}_nm"], color=color)
            axes[2, col].plot(f.sample_time_s, f[f"computed_wheel_torque_{wheel}_nm"], ":", color=color, alpha=.5)
        axes[1, col].set(ylabel="Wheel speed (rad/s)")
        axes[2, col].set(ylabel="Wheel torque (N m)", xlabel="Time (s)")
        axes[0, col].legend(fontsize=8, ncol=3)
        axes[1, col].legend(fontsize=8, ncol=2)
    fig.suptitle("Cross-zero runs: solid = measured/applied; dotted = target/computed", fontsize=12)
    finish(fig, "04_reversals_wheels_torques")


def spot_checks(root, stages):
    checks = []
    for name, index in (("F7", 0), ("S4", 1), ("S3", 3)):
        with (root / "formal" / name / "steps.csv").open() as stream:
            rows = [r for r in csv.DictReader(stream) if int(r["stage_index"]) == index]
        errors = [float(r["desired_speed_m_s"]) - float(r["actual_body_vx_m_s"]) for r in rows]
        end = float(rows[-1]["sample_time_s"])
        tail = [r for r in rows if round(float(r["sample_time_s"]), 9) > round(end - 1, 9)]
        values = {"speed_mae_m_s": sum(map(abs, errors)) / len(errors),
                  "speed_rmse_m_s": math.sqrt(sum(e * e for e in errors) / len(errors)),
                  "steady_speed_bias_m_s": sum(float(r["desired_speed_m_s"]) - float(r["actual_body_vx_m_s"]) for r in tail) / len(tail)}
        if name == "F7":
            values["steady_saturation_fraction"] = sum(abs(float(r["mpc_speed_target_m_s"])) >= 1.5 - 1e-4 for r in tail) / len(tail)
        if name == "S4":
            for column, clock, key in (
                ("executed_speed_m_s", "command_time_s", "executed_crossing_time_s"),
                ("actual_body_vx_m_s", "sample_time_s", "actual_crossing_time_s"),
            ):
                first = next(i for i in range(len(rows) - 2) if all(float(r[column]) < -0.02 for r in rows[i:i+3]))
                values[key] = float(rows[first][clock]) - float(rows[0]["command_time_s"])
            values["vehicle_after_input_crossing_delay_s"] = values["actual_crossing_time_s"] - values["executed_crossing_time_s"]
        if name == "S3":
            xy = [(float(tail[0]["pre_world_x_m"]), float(tail[0]["pre_world_y_m"]))]
            xy += [(float(r["actual_world_x_m"]), float(r["actual_world_y_m"])) for r in tail]
            values["stop_tail_signed_world_x_drift_m"] = xy[-1][0] - xy[0][0]
            values["stop_tail_path_length_m"] = sum(math.hypot(b[0]-a[0], b[1]-a[1]) for a,b in zip(xy, xy[1:]))
        table = stages[(stages.case == name) & (stages.stage_index == index)].iloc[0]
        discrepancy = {k: abs(value - float(table[k])) for k, value in values.items()}
        assert max(discrepancy.values()) < 1e-12
        selected = {0, len(rows) - 21, len(rows) - 20, len(rows) - 1}
        for time_key, clock in (("executed_crossing_time_s", "command_time_s"), ("actual_crossing_time_s", "sample_time_s")):
            if not pd.isna(table[time_key]):
                first = min(range(len(rows)), key=lambda i: abs(float(rows[i][clock]) - float(rows[0]["command_time_s"]) - table[time_key]))
                selected.update(range(max(0, first - 1), min(len(rows), first + 3)))
        checks.append({"case": name, "stage": index, "samples": len(rows), "tail_samples": len(tail),
                       "tail_first_sample_s": float(tail[0]["sample_time_s"]), "tail_last_sample_s": end,
                       "independent_stdlib_metrics": values, "table_absolute_differences": discrepancy,
                       "raw_rows": [rows[i] for i in sorted(selected)]})
    save_json(root / "analysis/raw_spot_checks.json", checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    frames, metas, stages, runs = {}, {}, [], []
    for phase in ("smoke", "formal", "supplement"):
        for path in sorted((root / phase).glob("*/metadata.json")):
            meta, frame, summary = read_metrics(path.parent)
            metas[path.parent.name], frames[path.parent.name] = meta, frame
            summary["phase"] = phase
            stages.extend({**s, "phase": phase} for s in summary.pop("stages"))
            runs.append(summary)
    all_stages = pd.DataFrame(stages)
    all_stages.to_csv(out / "all_stage_metrics.csv", index=False)
    pd.DataFrame(runs).to_csv(out / "run_summary.csv", index=False)
    formal = all_stages[all_stages.phase == "formal"]
    formal[formal.case.str.startswith("F")].to_csv(out / "fixed_results.csv", index=False)
    formal[formal.case.str.startswith("S")].to_csv(out / "directed_step_results.csv", index=False)
    save_json(out / "run_summary.json", runs)
    smoke_comparison(root, frames, metas)
    if len(formal) != 29 or sum(r["samples"] for r in runs if r["phase"] == "formal") != 4640:
        raise RuntimeError("Formal matrix is incomplete: retained partial analysis, no figures or full-matrix conclusion")
    formal_frames = [frames[r["case"]] for r in runs if r["phase"] == "formal"]
    save_json(out / "formal_timing.json", timing(pd.concat(formal_frames)))
    supplements = []
    for row in formal[formal.extension_trigger].itertuples():
        profile = metas[row.case]["commands"][:int(row.stage_index) + 1]
        supplements.append([f"L_{row.case}_{int(row.stage_index)}", [p[0] for p in profile], (int(row.stage_index) + 1) * 8 + 8, 8, 42])
    save_json(out / "suggested_supplements.json", supplements)
    draw(root, frames, formal)
    spot_checks(root, formal)
    index = [json.loads(p.read_text()) for phase in ("smoke", "formal", "supplement") for p in sorted((root / phase).glob("*.result.json"))]
    save_json(out / "run_index.json", index)
    pd.DataFrame(index).to_csv(out / "run_index.csv", index=False)
    manifest = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                for phase in ("smoke", "formal", "supplement") for p in (root / phase).rglob("*") if p.is_file()}
    save_json(out / "artifact_sha256.json", manifest)
    print(formal[["case", "stage_index", "desired_speed_m_s", "speed_mae_m_s", "steady_speed_bias_m_s", "speed_settling_time_s", "extension_trigger"]].to_string(index=False))
    print("Supplement candidates:", supplements)


if __name__ == "__main__":
    main()
