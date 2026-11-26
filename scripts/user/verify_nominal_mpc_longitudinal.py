"""Audit raw-row alignment, ideal Ackermann mapping, run provenance and extended prefixes."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from analyze_nominal_mpc_longitudinal import WHEELS, save_json

REPO = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for phase in ("smoke", "formal", "supplement"):
        for path in sorted((root / phase).glob("*/metadata.json")):
            meta = json.loads(path.read_text())
            frame = pd.read_csv(path.parent / "steps.csv")
            runtime = json.loads((path.parent / "runtime_environment.json").read_text())
            action = runtime["environment"]["actions"]["throttle_steer"]
            length, width, radius = action["base_length"], action["base_width"], action["wheel_radius"]
            tangent = np.tan(frame.executed_steering_rad.to_numpy())
            turn_radius = np.divide(length, tangent, out=np.full_like(tangent, 1e6), where=tangent != 0)
            speed = frame.executed_speed_m_s.to_numpy()
            expected = {
                "rear_left": speed * np.abs((turn_radius - width / 2) / (turn_radius * radius)),
                "rear_right": speed * np.abs((turn_radius + width / 2) / (turn_radius * radius)),
                "front_left": speed * np.abs(np.sqrt((turn_radius - width / 2) ** 2 + length ** 2) / (turn_radius * radius)),
                "front_right": speed * np.abs(np.sqrt((turn_radius + width / 2) ** 2 + length ** 2) / (turn_radius * radius)),
            }
            steering = {"front_left": np.arctan(length / (turn_radius - width / 2)),
                        "front_right": np.arctan(length / (turn_radius + width / 2))}
            wheel_error = {w: float(np.max(np.abs(expected[w] - frame[f"target_wheel_speed_{w}_rad_s"]))) for w in WHEELS}
            steer_error = {w: float(np.max(np.abs(v - frame[f"target_steering_{w}_rad"]))) for w, v in steering.items()}
            correct_sign = {w: bool((np.sign(frame[f"target_wheel_speed_{w}_rad_s"]) == np.sign(speed)).all()) for w in WHEELS}
            assert max(wheel_error.values()) < 1e-5
            assert max(steer_error.values()) < 1e-6
            assert all(correct_sign.values())
            assert [meta["wheel_joint_mapping"][w]["id"] for w in WHEELS] == [9, 8, 4, 5]
            assert meta["action_wheel_ids"] == [4, 5, 9, 8]
            assert meta["action_steering_ids"] == [7, 6]
            assert np.max(np.abs(np.array(meta["initial_state"]["root_state_w"])[7:])) == 0
            assert meta["initial_state"]["executed_action"] == [0, 0]
            assert abs(float(frame.pre_heading_rad.iloc[0])) < 1e-9
            assert runtime["environment"]["observations"]["policy"]["enable_corruption"] is False
            assert frame.environment_desired_speed_m_s.sub(frame.desired_speed_m_s).abs().max() < 1e-6
            assert np.max(np.abs(frame.normalized_action_speed * 1.5 - frame.mpc_speed_target_m_s)) < 1e-6
            assert np.max(np.abs(frame.normalized_action_steering * meta["action_scale"][1] - frame.mpc_steering_target_rad)) < 1e-6
            log = path.parent.parent / (path.parent.name + ".console.log")
            warning_lines = [line for line in log.read_text().splitlines() if "[Warning]" in line or "[Error]" in line or "UserWarning:" in line]
            records.append({"case": path.parent.name, "phase": phase, "steps": len(frame),
                            "wheel_mapping_max_error_rad_s": wheel_error, "steering_mapping_max_error_rad": steer_error,
                            "target_wheel_sign_correct": correct_sign, "warning_lines": warning_lines,
                            "wheel_mapping": meta["wheel_joint_mapping"], "steering_mapping": meta["steering_joint_mapping"],
                            "runtime_versions": runtime["versions"], "osqp_native_version": runtime["osqp_native_version"],
                            "action_apply_path": runtime["apply_actions_file"], "action_apply_sha256": runtime["apply_actions_sha256"],
                            "initial_heading_rad": float(frame.pre_heading_rad.iloc[0]),
                            "initial_root_state_w": meta["initial_state"]["root_state_w"],
                            "termination": meta["termination_reason"]})
    prefixes = []
    for name, source in (("L_S2a_2", "S2a"), ("L_S4_2", "S4")):
        new = root / "supplement" / name / "steps.csv"
        if not new.exists():
            continue
        old = pd.read_csv(root / "formal" / source / "steps.csv")
        extended = pd.read_csv(new).iloc[:len(old)]
        columns = ["desired_speed_m_s", "actual_body_vx_m_s", "actual_heading_rad", "executed_speed_m_s", "executed_steering_rad"]
        differences = (old[columns] - extended[columns]).abs().max().to_dict()
        prefixes.append({"supplement": name, "formal": source, "prefix_samples": len(old), "maximum_differences": differences})
        assert max(differences.values()) == 0
    inventory = json.loads((root / "provenance/inventory.json").read_text())
    changes = {name: sha(REPO / name) for name, digest in inventory["files_sha256"].items() if sha(REPO / name) != digest}
    assert not changes, changes
    data_dir = root.parents[1] / "traction_protective_control_nominal_identification/20261120_173554"
    data_hashes = {name: sha(data_dir / name) for name in ("identification.csv", "validation.csv")}
    assert data_hashes == {
        "identification.csv": "323cac91fd9c98384cab59253db715c4d11c5acb46c8bdf67406f36fd36f9870",
        "validation.csv": "3258542fde8b5a403a54ef9bf3998bfde813f2055470c0958418aa44fbf35529",
    }
    save_json(root / "analysis/collection_verification.json", {"runs": records, "extended_prefixes": prefixes,
              "frozen_source_changes": changes, "identification_source_sha256": data_hashes,
              "historical_identification_usd_hash": None, "cross_machine": "not_tested"})
    final = root / "provenance/final_sources"
    final.mkdir(exist_ok=True)
    files = [REPO / "scripts/user" / name for name in (
        "eval_nominal_mpc.py", "analyze_nominal_mpc.py", "run_nominal_mpc_longitudinal.py",
        "analyze_nominal_mpc_longitudinal.py", "verify_nominal_mpc_longitudinal.py", "test_nominal_mpc_longitudinal.py")]
    for path in files:
        (final / path.name).write_bytes(path.read_bytes())
    save_json(final / "sha256.json", {path.name: sha(path) for path in files})
    (root / "provenance/final_worktree.patch").write_text(subprocess.check_output(["git", "diff", "HEAD"], cwd=REPO, text=True))
    (root / "provenance/final_status.txt").write_text(subprocess.check_output(["git", "status", "--short", "--branch"], cwd=REPO, text=True))
    print(json.dumps({"verified_runs": len(records), "steps_by_phase": dict(Counter({p: sum(r["steps"] for r in records if r["phase"] == p) for p in ("smoke", "formal", "supplement")})), "extended_prefixes": prefixes, "frozen_changes": changes}, indent=2))


if __name__ == "__main__":
    main()
