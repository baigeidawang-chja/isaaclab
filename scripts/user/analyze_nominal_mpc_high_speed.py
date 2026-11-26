"""Audit the paired 1.2 m/s high-speed diagnostic from recorded Isaac Sim CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
WHEELS = ("front_left", "front_right", "rear_left", "rear_right")
NAMES = {"A": "A_nominal_mpc", "B": "B_max_physical_input"}
TARGET = 1.2
LIMIT = 1.5
DT = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def window(frame: pd.DataFrame, end_s: float, length_s: float) -> pd.DataFrame:
    time = frame.sample_time_s.round(9)
    result = frame[(time > end_s - length_s) & (time <= end_s)]
    expected = round(length_s / DT)
    if len(result) != expected:
        raise ValueError(f"Window ({end_s - length_s}, {end_s}] has {len(result)} rather than {expected} samples")
    return result


def tail_span(frame: pd.DataFrame) -> tuple[int, float]:
    within = (frame.actual_body_vx_m_s.sub(TARGET).abs() <= 0.05).to_numpy()
    count = 0
    for value in within[::-1]:
        if not value:
            break
        count += 1
    return count, max(0.0, (count - 1) * DT)


def read_run(root: Path, group: str) -> tuple[dict, dict, pd.DataFrame]:
    directory = root / NAMES[group]
    meta = json.loads((directory / "metadata.json").read_text())
    runtime = json.loads((directory / "runtime_environment.json").read_text())
    frame = pd.read_csv(directory / "steps.csv")
    if len(frame) != meta["steps"] or len(frame) != 320:
        raise ValueError(f"{group}: expected exactly 320 recorded control steps")
    if any(frame.terminated) or any(frame.truncated) or not all(frame.response_sample_valid):
        raise ValueError(f"{group}: terminated, truncated or reset samples cannot form a continuous response")
    if meta["termination_reason"] != "evaluation_horizon" or meta["fallback_count"] != 0:
        raise ValueError(f"{group}: unexpected termination or steering-controller fallback")
    if meta["solver_status_counts"] != {"solved": 320}:
        raise ValueError(f"{group}: unsolved MPC steps")
    if meta["seed"] != 42 or meta["step_dt_s"] != DT or meta["commands"] != [[TARGET, 0.0]]:
        raise ValueError(f"{group}: mismatched reference, seed, or control period")
    if not np.allclose(frame.desired_speed_m_s, TARGET) or not np.allclose(frame.desired_heading_rad, 0):
        raise ValueError(f"{group}: per-step planner reference changed")
    if not np.allclose(frame.command_time_s, np.arange(320) * DT, atol=1e-9):
        raise ValueError(f"{group}: command timestamps shifted")
    if not np.allclose(frame.sample_time_s, np.arange(1, 321) * DT, atol=1e-9):
        raise ValueError(f"{group}: post-step timestamps shifted")
    if not np.allclose(frame.pre_body_speed_m_s.iloc[1:], frame.actual_body_vx_m_s.iloc[:-1], atol=1e-9):
        raise ValueError(f"{group}: pre/post speed alignment changed")
    if not np.allclose(frame.previous_executed_speed_m_s.iloc[1:], frame.executed_speed_m_s.iloc[:-1], atol=1e-9):
        raise ValueError(f"{group}: previous executed input does not match recorded prior input")
    if not np.allclose(frame.actual_projected_speed_m_s, frame.actual_world_vx_m_s, atol=1e-9):
        raise ValueError(f"{group}: zero-heading world projection mismatch")
    if max(abs(frame.physical_target_speed_m_s - frame.selected_speed_target_m_s)) > 2e-6:
        raise ValueError(f"{group}: physical action adapter mapping mismatch")
    if max(abs(frame.executed_speed_m_s.diff().iloc[1:])) > 0.100002:
        raise ValueError(f"{group}: ActionTerm speed-rate limit exceeded")
    for label, source in (("model", "model_config"), ("mpc", "mpc_config"), ("steering", "steering_config")):
        if sha256(directory / f"{label}_config.json") != meta[f"{label}_config_sha256"]:
            raise ValueError(f"{group}: {source} snapshot hash mismatch")
    if sha256(Path(meta["robot_usd_path"])) != meta["robot_usd_sha256"]:
        raise ValueError(f"{group}: loaded vehicle USD hash mismatch")
    if sha256(Path(runtime["apply_actions_file"])) != runtime["apply_actions_sha256"]:
        raise ValueError(f"{group}: WheeledLab action implementation hash mismatch")
    for wheel, joint_id in zip(WHEELS, (9, 8, 4, 5)):
        if meta["wheel_joint_mapping"][wheel]["id"] != joint_id:
            raise ValueError(f"{group}: wheel joint mapping mismatch for {wheel}")
    if meta["action_wheel_ids"] != [4, 5, 9, 8] or meta["action_steering_ids"] != [7, 6]:
        raise ValueError(f"{group}: action output joint order mismatch")
    return meta, runtime, frame


def metrics(group: str, frame: pd.DataFrame, end_s: int) -> dict:
    observed = frame[frame.sample_time_s.round(9) <= end_s]
    final_second = window(frame, end_s, 1)
    final_four = window(frame, end_s, 4)
    y = final_second.actual_body_vx_m_s
    count, span = tail_span(observed)
    upper = final_second.mpc_speed_target_m_s >= LIMIT - 1e-4
    first_half = final_four.iloc[:40].actual_body_vx_m_s.mean()
    second_half = final_four.iloc[40:].actual_body_vx_m_s.mean()
    return {
        "group": group, "horizon_s": end_s, "samples": len(observed),
        "last_1s_speed_mean_m_s": float(y.mean()),
        "last_1s_signed_error_m_s": float(TARGET - y.mean()),
        "last_1s_speed_min_m_s": float(y.min()),
        "last_1s_speed_max_m_s": float(y.max()),
        "last_1s_speed_range_m_s": float(y.max() - y.min()),
        "last_1s_speed_std_m_s": float(y.std(ddof=1)),
        "last_1s_mean_in_band": bool(1.15 <= y.mean() <= 1.25),
        "ever_reached_1p2": bool((observed.actual_body_vx_m_s >= TARGET).any()),
        "last_1s_in_band_fraction": float((y.sub(TARGET).abs() <= 0.05).mean()),
        "trailing_in_band_samples": count,
        "trailing_in_band_span_s": span,
        "last_4s_speed_mean_m_s": float(final_four.actual_body_vx_m_s.mean()),
        "last_4s_speed_min_m_s": float(final_four.actual_body_vx_m_s.min()),
        "last_4s_speed_max_m_s": float(final_four.actual_body_vx_m_s.max()),
        "last_4s_speed_slope_m_s2": float(np.polyfit(final_four.sample_time_s, final_four.actual_body_vx_m_s, 1)[0]),
        "last_4s_second_2s_minus_first_2s_m_s": float(second_half - first_half),
        "whole_mpc_upper_fraction": float((observed.mpc_speed_target_m_s >= LIMIT - 1e-4).mean()) if group == "A" else None,
        "last_1s_mpc_upper_fraction": float(upper.mean()) if group == "A" else None,
        "last_1s_physical_speed_target_mean_m_s": float(final_second.physical_target_speed_m_s.mean()),
        "last_1s_executed_speed_target_mean_m_s": float(final_second.executed_speed_m_s.mean()),
        "last_1s_heading_mae_rad": float(final_second.actual_heading_error_rad.abs().mean()),
        "last_1s_heading_max_abs_rad": float(final_second.actual_heading_error_rad.abs().max()),
        "whole_heading_max_abs_rad": float(observed.actual_heading_error_rad.abs().max()),
        "last_1s_left_steer_mean_rad": float(final_second.actual_steering_front_left_rad.mean()),
        "last_1s_right_steer_mean_rad": float(final_second.actual_steering_front_right_rad.mean()),
        "last_1s_steer_max_abs_rad": float(final_second[["actual_steering_front_left_rad", "actual_steering_front_right_rad"]].abs().to_numpy().max()),
    }


def wheel_metrics(group: str, frame: pd.DataFrame, end_s: int) -> list[dict]:
    final_second = window(frame, end_s, 1)
    result = []
    for wheel in WHEELS:
        target = final_second[f"target_wheel_speed_{wheel}_rad_s"]
        actual = final_second[f"actual_wheel_speed_{wheel}_rad_s"]
        computed = final_second[f"computed_wheel_torque_{wheel}_nm"]
        applied = final_second[f"applied_wheel_torque_{wheel}_nm"]
        result.append({
            "group": group, "horizon_s": end_s, "wheel": wheel,
            "target_wheel_speed_mean_rad_s": float(target.mean()),
            "actual_wheel_speed_mean_rad_s": float(actual.mean()),
            "wheel_speed_error_mae_rad_s": float((target - actual).abs().mean()),
            "computed_torque_abs_mean_nm": float(computed.abs().mean()),
            "applied_torque_abs_mean_nm": float(applied.abs().mean()),
            "torque_clip_fraction": float((computed.abs() > applied.abs() + 1e-4).mean()),
        })
    return result


def plot(root: Path, frames: dict[str, pd.DataFrame]) -> None:
    a, b = frames["A"], frames["B"]
    fig, (upper, lower, straight) = plt.subplots(
        3, 1, sharex=True, figsize=(10.5, 8), gridspec_kw={"height_ratios": [2, 2, 1.4]}
    )
    upper.plot(a.command_time_s, a.desired_speed_m_s, "k--", label="Planner reference 1.2 m/s")
    upper.plot(a.command_time_s, a.physical_target_speed_m_s, color="#b64f64", label="A physical target")
    upper.plot(a.command_time_s, a.executed_speed_m_s, color="#b64f64", ls=":", label="A executed target")
    upper.plot(b.command_time_s, b.physical_target_speed_m_s, color="#226caa", label="B physical target")
    upper.plot(b.command_time_s, b.executed_speed_m_s, color="#226caa", ls=":", label="B executed target")
    upper.set(ylabel="Physical speed input (m/s)", title="Paired high-speed diagnostic: same 1.2 m/s planner reference")
    upper.legend(ncol=3, fontsize=8)
    lower.axhspan(1.15, 1.25, color="#999999", alpha=.15, label="1.2 ± 0.05 m/s band")
    lower.plot(a.sample_time_s, a.actual_body_vx_m_s, color="#b64f64", label="A measured body speed")
    lower.plot(b.sample_time_s, b.actual_body_vx_m_s, color="#226caa", label="B measured body speed")
    lower.axhline(TARGET, ls="--", color="black", lw=1)
    lower.set(ylabel="Post-step body speed (m/s)")
    lower.legend(ncol=3, fontsize=8)
    straight.plot(a.sample_time_s, a.actual_heading_rad, color="#b64f64", label="A heading")
    straight.plot(b.sample_time_s, b.actual_heading_rad, color="#226caa", label="B heading")
    straight.plot(a.sample_time_s, a.actual_equivalent_center_steering_rad, color="#b64f64", ls=":", label="A center steer")
    straight.plot(b.sample_time_s, b.actual_equivalent_center_steering_rad, color="#226caa", ls=":", label="B center steer")
    straight.set(xlabel="Time since reset (s)", ylabel="Heading / steering (rad)", xlim=(0, 16))
    straight.legend(ncol=4, fontsize=8)
    for axis in (upper, lower, straight):
        axis.grid(alpha=.25)
        axis.axvline(8, color="0.5", ls=":", lw=1)
    fig.tight_layout()
    fig.savefig(root / "paired_response.png", dpi=200)
    fig.savefig(root / "paired_response.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--historical-f7", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    loaded = {group: read_run(root, group) for group in NAMES}
    metas = {group: loaded[group][0] for group in NAMES}
    runtimes = {group: loaded[group][1] for group in NAMES}
    frames = {group: loaded[group][2] for group in NAMES}
    a, b = frames["A"], frames["B"]
    for key in ("initial_state", "robot_usd_sha256", "model_config_sha256", "mpc_config_sha256",
                "steering_config_sha256", "seed", "step_dt_s", "action_scale", "action_offset"):
        if metas["A"][key] != metas["B"][key]:
            raise ValueError(f"Pair configuration mismatch: {key}")
    if runtimes["A"]["environment"] != runtimes["B"]["environment"]:
        raise ValueError("Pair effective environment configurations differ")
    if runtimes["A"]["apply_actions_sha256"] != runtimes["B"]["apply_actions_sha256"]:
        raise ValueError("Pair action implementation mismatch")
    if metas["A"]["longitudinal_output_mode"] != "nominal_mpc" or metas["B"]["longitudinal_output_mode"] != "fixed_physical_speed_target":
        raise ValueError("Pair longitudinal output modes are not A=MPC, B=fixed target")
    if metas["A"]["physical_speed_override_m_s"] is not None or metas["B"]["physical_speed_override_m_s"] != LIMIT:
        raise ValueError("Pair override values differ from protocol")
    if max(abs(a.selected_speed_target_m_s - a.mpc_speed_target_m_s)) > 1e-9:
        raise ValueError("A physical target deviates from nominal MPC output")
    if not np.allclose(b.selected_speed_target_m_s, LIMIT, atol=1e-9):
        raise ValueError("B physical target is not constant 1.5 m/s")
    reached = np.flatnonzero((b.executed_speed_m_s >= LIMIT - 1e-4).to_numpy())
    if len(reached) == 0 or not np.all(b.executed_speed_m_s.iloc[reached[0]:] >= LIMIT - 1e-4):
        raise ValueError("B executed target did not reach and hold 1.5 m/s")
    if not np.allclose(b.executed_speed_m_s.iloc[:reached[0] + 1], np.minimum((np.arange(reached[0] + 1) + 1) * .1, LIMIT), atol=2e-6):
        raise ValueError("B executed target did not ramp at 0.1 m/s per control period")
    old = pd.read_csv(args.historical_f7.expanduser() / "steps.csv")
    columns = ("desired_speed_m_s", "desired_heading_rad", "mpc_speed_target_m_s", "mpc_steering_target_rad",
               "physical_target_speed_m_s", "executed_speed_m_s", "executed_steering_rad",
               "actual_body_vx_m_s", "actual_heading_rad", "actual_steering_front_left_rad",
               "actual_steering_front_right_rad")
    prefix_diff = {column: float(np.max(np.abs(a[column].iloc[:160].to_numpy() - old[column].to_numpy())))
                   for column in columns}
    if len(old) != 160 or max(prefix_diff.values()) > 1e-6:
        raise ValueError("A first 8 seconds differ materially from historical F7")
    records = [metrics(group, frames[group], horizon) for group in NAMES for horizon in (8, 16)]
    wheel_records = [record for group in NAMES for horizon in (8, 16)
                     for record in wheel_metrics(group, frames[group], horizon)]
    pd.DataFrame(records).to_csv(root / "core_metrics.csv", index=False)
    pd.DataFrame(wheel_records).to_csv(root / "wheel_comparison.csv", index=False)
    plot(root, frames)
    input_differences = {}
    for label, start, stop in (("whole_16s", 0, 16), ("first_8s", 0, 8),
                               ("second_8s", 8, 16), ("last_4s", 12, 16),
                               ("last_1s", 15, 16)):
        selected = (a.sample_time_s.round(9) > start) & (a.sample_time_s.round(9) <= stop)
        input_delta = b.loc[selected, "executed_speed_m_s"].to_numpy() - a.loc[selected, "executed_speed_m_s"].to_numpy()
        speed_delta = b.loc[selected, "actual_body_vx_m_s"].to_numpy() - a.loc[selected, "actual_body_vx_m_s"].to_numpy()
        input_differences[label] = {
            "executed_target_B_minus_A_mean_m_s": float(input_delta.mean()),
            "executed_target_max_abs_difference_m_s": float(np.abs(input_delta).max()),
            "measured_speed_B_minus_A_mean_m_s": float(speed_delta.mean()),
            "measured_speed_max_abs_difference_m_s": float(np.abs(speed_delta).max()),
        }
    diagnostic = {
        "data_source": "Isaac Lab paired single-environment simulation",
        "comparison": "A nominal MPC speed+steering; B fixed physical speed target 1.5 m/s with nominal MPC steering",
        "planner_reference_speed_m_s": TARGET,
        "planner_reference_heading_rad": 0.0,
        "seed": 42, "control_period_s": DT,
        "A_historical_F7_first_8s_max_differences": prefix_diff,
        "B_first_sustained_upper_input_command_time_s": float(b.command_time_s.iloc[reached[0]]),
        "B_first_sustained_upper_input_post_sample_time_s": float(b.sample_time_s.iloc[reached[0]]),
        "B_upper_input_held_samples": len(b) - int(reached[0]),
        "B_previous_input_matches_actual_executed": True,
        "paired_input_differences": input_differences,
        "effective_environment_config_identical": True,
        "A_mpc_upper_fraction_16s": float((a.mpc_speed_target_m_s >= LIMIT - 1e-4).mean()),
        "A_mpc_upper_fraction_8s": float((a.mpc_speed_target_m_s.iloc[:160] >= LIMIT - 1e-4).mean()),
        "B_predicted_states_role": metas["B"]["prediction_role"],
        "metadata": {group: {key: metas[group][key] for key in ("argv", "steps", "termination_reason", "solver_status_counts",
                                                                "fallback_count", "robot_usd_path", "robot_usd_sha256",
                                                                "model_config_sha256", "mpc_config_sha256", "steering_config_sha256",
                                                                "initial_state", "longitudinal_output_mode", "physical_speed_override_m_s")}
                     for group in NAMES},
        "runtime": {group: {"versions": runtimes[group]["versions"], "isaac_sim_version": runtimes[group]["isaac_sim_version"],
                            "apply_actions_file": runtimes[group]["apply_actions_file"],
                            "apply_actions_sha256": runtimes[group]["apply_actions_sha256"]}
                    for group in NAMES},
        "complete_solve_status": {group: metas[group]["solver_status_counts"] for group in NAMES},
    }
    save_json(root / "diagnostic_checks.json", diagnostic)
    plot_sources = [
        REPO / "scripts/user/eval_nominal_mpc.py",
        Path(__file__),
        REPO / "source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control/mdp/actions.py",
        Path(runtimes["A"]["apply_actions_file"]),
    ]
    source_set_hash = hashlib.sha256(
        "".join(sha256(path) for path in plot_sources).encode("ascii")
    ).hexdigest()[:12]
    snapshots = root / f"source_snapshots_{source_set_hash}"
    snapshots.mkdir(exist_ok=True)
    source_hashes = {}
    for path in plot_sources:
        destination = snapshots / path.name
        if destination.exists():
            if destination.read_bytes() != path.read_bytes():
                raise ValueError(f"Source snapshot differs from the current file: {destination}")
        else:
            destination.write_bytes(path.read_bytes())
        source_hashes[str(path)] = sha256(path)
    save_json(root / "provenance.json", {
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip(),
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "worktree_status": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=REPO, text=True),
        "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], text=True),
        "source_sha256": source_hashes,
        "source_snapshot_dir": str(snapshots),
        "historical_f7_steps_sha256": sha256(args.historical_f7.expanduser() / "steps.csv"),
        "independent_runs_per_group": 1,
    })
    print(pd.DataFrame(records).to_string(index=False))
    print("B first held 1.5 m/s input:", diagnostic["B_first_sustained_upper_input_command_time_s"], "s")
    print("A/F7 first 8s max difference:", max(prefix_diff.values()))


if __name__ == "__main__":
    main()
