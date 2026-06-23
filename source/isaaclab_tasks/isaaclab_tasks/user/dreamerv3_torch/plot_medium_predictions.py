"""Plot Amphibious medium-state labels and offline predictor outputs."""

from __future__ import annotations

import argparse
import os
import pathlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABEL_NAMES = ["lambda", "eta_wheel", "eta_thruster", "drag_scale"]


def _save_label_curves(labels: np.ndarray, figures_dir: pathlib.Path, env_id: int, max_points: int):
    steps = np.arange(min(labels.shape[0], max_points))
    env_id = min(env_id, labels.shape[1] - 1)
    values = labels[: len(steps), env_id]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(steps, values[:, idx], linewidth=1.5)
        ax.set_ylabel(LABEL_NAMES[idx])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"Medium-state labels/env {env_id}")
    fig.tight_layout()
    path = figures_dir / "medium_label_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved={path}")


def _save_prediction_curve(preds, figures_dir: pathlib.Path, key: str, env_id: int):
    target = preds["target"]
    mlp_pred = preds["mlp_pred"]
    gru_pred = preds["gru_pred"]
    times = preds["time"]
    env_ids = preds["env_id"]
    idx = LABEL_NAMES.index(key)
    mask = env_ids == env_id
    if not np.any(mask):
        env_id = int(env_ids[0])
        mask = env_ids == env_id

    order = np.argsort(times[mask])
    plot_times = times[mask][order]
    true = target[mask, idx][order]
    mlp = mlp_pred[mask, idx][order]
    gru = gru_pred[mask, idx][order]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(plot_times, true, label=f"true {key}", linewidth=2.0)
    ax.plot(plot_times, mlp, label="MLP", linewidth=1.4, alpha=0.85)
    ax.plot(plot_times, gru, label="GRU", linewidth=1.4, alpha=0.85)
    ax.set_xlabel("step")
    ax.set_ylabel(key)
    ax.set_title(f"{key}: true vs MLP vs GRU/env {env_id}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / f"{key}_prediction_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved={path}")


def main():
    parser = argparse.ArgumentParser(description="Plot medium-state labels and predictions.")
    parser.add_argument("--dataset", type=str, default="runs/medium_dataset/amphibious_medium_dataset.npz")
    parser.add_argument("--predictions", type=str, default="runs/medium_dataset/medium_predictions.npz")
    parser.add_argument("--figures_dir", type=str, default="runs/medium_dataset/figures")
    parser.add_argument("--env_id", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=2000)
    args = parser.parse_args()

    figures_dir = pathlib.Path(args.figures_dir).expanduser()
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(pathlib.Path(args.dataset).expanduser())
    labels = dataset["medium_state_label"].astype(np.float32)
    _save_label_curves(labels, figures_dir, args.env_id, args.max_points)

    pred_path = pathlib.Path(args.predictions).expanduser()
    if not pred_path.exists():
        print(f"[PLOT] predictions not found, skipped comparison plots: {pred_path}")
        return
    preds = np.load(pred_path)
    _save_prediction_curve(preds, figures_dir, "eta_wheel", args.env_id)
    _save_prediction_curve(preds, figures_dir, "eta_thruster", args.env_id)


if __name__ == "__main__":
    main()
