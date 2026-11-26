"""Run the frozen longitudinal matrix in separate Isaac processes; never overwrite runs."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REPO = Path(__file__).resolve().parents[2]
TASK = "Isaac-TractionProtect-CommandFollow-Car4WD-v0"
BASE = REPO / "source/isaaclab_tasks/isaaclab_tasks/user/traction_protective_control"
FROZEN = {
    "nominal_car4wd_bidirectional_model.json": "78fe257a39f108a103c0a0edb3dcc1afb53aff3b861bcac9926cfca7a28da3f3",
    "nominal_car4wd_bidirectional_osqp_mpc.json": "96371980dd186468da813fe339f02838956ef49f9fa19ba8be043c6ef6f1aded",
    "nominal_car4wd_steering.json": "16104c2c3febad5921d21d9d1989191cd2ee574c5d8c78325d319af778090ac0",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(command):
    return subprocess.check_output(command, cwd=REPO, text=True)


def cases(phase):
    if phase == "smoke":
        return [("B1", [0.6], 4, 4, 42), ("B2", [0.6], 4, 4, 42), ("B3", [0.6], 4, 4, 43)]
    return [(f"F{i}", [v], 8, 8, 42) for i, v in enumerate([-0.5, -0.25, 0, 0.3, 0.6, 0.9, 1.2], 1)] + [
        ("S1", [0.3, 0.6, 0.9, 1.2, 0.9, 0.6, 0.3], 56, 8, 42),
        ("S2a", [0.6, 0.3, 1.2], 24, 8, 42),
        ("S2b", [0.6, 0.9, 1.2], 24, 8, 42),
        ("S3", [0.5, -0.5, 0.5, 0, -0.5, 0], 48, 8, 42),
        ("S4", [1.2, -0.5, 1.2], 24, 8, 42),
    ]


def provenance(root):
    target = root / "provenance"
    target.mkdir()
    paths = list(BASE.rglob("*.py")) + list((BASE / "mpc/config").glob("*.json"))
    paths += [REPO / "scripts/user/eval_nominal_mpc.py", REPO / "scripts/user/analyze_nominal_mpc.py",
              Path(__file__), REPO / "source/isaaclab_assets/isaaclab_assets/robots/car_edit.py",
              REPO / "WheeledLab/source/wheeledlab/wheeledlab/envs/mdp/actions/ackermann_actions.py",
              REPO / "AGENTS.md"]
    records = {}
    for path in paths:
        relative = path.relative_to(REPO)
        destination = target / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
        records[str(relative)] = digest(path)
    usd = Path("/home/chja/myproject/car_no_damper_04.usd")
    records[str(usd)] = digest(usd)
    for filename, expected in FROZEN.items():
        assert digest(BASE / "mpc/config" / filename) == expected, filename
    assert records[str(usd)] == "28c56b286c78df43a6f9d06a9f9393be205f9b3279211b5c05b14b95ee7a5940"
    for label, location in [("main", REPO), ("WheeledLab", REPO / "WheeledLab"), ("tdmpc2", REPO / "tdmpc2/tdmpc2")]:
        (target / f"{label}.patch").write_text(capture(["git", "-C", str(location), "diff", "HEAD"]))
    write_json(target / "inventory.json", {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "commit": capture(["git", "rev-parse", "HEAD"]).strip(),
        "branch": capture(["git", "branch", "--show-current"]).strip(),
        "status": capture(["git", "status", "--short", "--branch"]),
        "diff_from_experiment": capture(["git", "diff", "--stat", "db77845751351fe789582db36f7beafec3aaf9a7"]),
        "gpu": capture(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]),
        "files_sha256": records,
        "historical_identification_usd_hash": None,
        "trend_rule": "At 1.2 m/s, extend if absolute last-2s OLS slope >0.01 m/s^2 or last-1s vs previous-1s mean difference >0.02 m/s.",
        "cross_machine_status": "local_only",
    })


def run_case(root, phase, case):
    name, speeds, duration, segment, seed = case
    parent = root / phase
    parent.mkdir(exist_ok=True)
    output = parent / name
    log = parent / f"{name}.console.log"
    if output.exists() or log.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {log}")
    command = [str(REPO / "isaaclab.sh"), "-p", "scripts/user/eval_nominal_mpc.py",
               "--task", TASK, "--headless", "--profile", "fixed" if len(speeds) == 1 else "speed_step",
               "--speed", str(speeds[0]), "--speed_sequence", ",".join(map(str, speeds)),
               "--heading", "0", "--duration", str(duration), "--segment_duration", str(segment),
               "--seed", str(seed), "--output_dir", str(output)]
    write_json(parent / f"{name}.launch.json", {"argv": command, "shell": shlex.join(command), "case": case})
    print(f"START {phase}/{name}: {duration}s, {speeds}", flush=True)
    start = time.perf_counter()
    with log.open("x") as stream:
        result = subprocess.run(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
    record = {"id": name, "phase": phase, "run_dir": str(output.relative_to(root)),
              "exit_code": result.returncode, "wall_time_s": time.perf_counter() - start,
              "expected_steps": round(duration / 0.05), "commands": speeds, "seed": seed}
    if result.returncode == 0 and (output / "metadata.json").exists():
        meta = json.loads((output / "metadata.json").read_text())
        with (output / "steps.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        prediction_count = sum(1 for _ in (output / "predictions.jsonl").open())
        record.update(steps=len(rows), stages=len({r["stage_index"] for r in rows}),
                      termination=meta["termination_reason"], statuses=meta["solver_status_counts"],
                      fallbacks=meta["fallback_count"], csv_sha256=digest(output / "steps.csv"))
        checks = {
            "steps": len(rows) == meta["steps"] == record["expected_steps"] == prediction_count,
            "commands": meta["commands"] == [[v, 0] for v in speeds],
            "final_stage": int(rows[-1]["stage_index"]) == len(speeds) - 1,
            "continuous": all(r["response_sample_valid"] == "1" for r in rows),
            "projection": all(float(r["actual_projected_speed_m_s"]) == float(r["actual_world_vx_m_s"]) for r in rows),
            "timing": all(abs(float(r["sample_time_s"]) - (i + 1) * 0.05) < 1e-9 for i, r in enumerate(rows)),
            "config": all(meta[f"{key}_config_sha256"] == expected for key, expected in zip(("model", "mpc", "steering"), FROZEN.values())),
        }
        record["checks"] = checks
        record["complete"] = all(checks.values())
    else:
        record["complete"] = False
    write_json(parent / f"{name}.result.json", record)
    print(json.dumps(record), flush=True)
    if not record["complete"]:
        raise RuntimeError(f"Run incomplete: inspect {log}; no subsequent cases launched")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("smoke", "formal", "supplement"), required=True)
    parser.add_argument("--case-json", type=Path, help="Supplement cases: [id, speeds, duration, segment, seed]")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not root.exists():
        root.mkdir(parents=True)
        provenance(root)
    elif not (root / "provenance/inventory.json").is_file():
        raise RuntimeError("Existing root has no experiment provenance")
    inventory = json.loads((root / "provenance/inventory.json").read_text())
    for name, expected in inventory["files_sha256"].items():
        if digest(REPO / name) != expected:
            raise RuntimeError(f"Frozen source changed since experiment start: {name}")
    selected = json.loads(args.case_json.read_text()) if args.phase == "supplement" else cases(args.phase)
    for case in selected:
        run_case(root, args.phase, case)


if __name__ == "__main__":
    main()
