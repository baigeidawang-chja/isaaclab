"""Plot episode-aware AmphibiousTerrain medium-state labels and predictions."""

from __future__ import annotations

import argparse
import os
import pathlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LABEL_NAMES = [
    "lambda_medium",
    "eta_wheel",
    "eta_thruster",
    "drag_scale",
    "slope_sin",
    "terrain_height",
    "terrain_phase",
]


def _label_names(data, dim: int):
    if "label_names" in data:
        return [str(x) for x in data["label_names"].tolist()]
    return DEFAULT_LABEL_NAMES[:dim]


def _segments_from_done(done_1d: np.ndarray, min_length: int):
    starts = [0]
    done_idx = np.flatnonzero(done_1d)
    starts.extend((idx + 1 for idx in done_idx if idx + 1 < len(done_1d)))
    ends = list(done_idx + 1)
    if not ends or ends[-1] < len(done_1d):
        ends.append(len(done_1d))
    segments = []
    for start in starts:
        end_candidates = [end for end in ends if end > start]
        if not end_candidates:
            continue
        end = end_candidates[0]
        if end - start >= min_length:
            segments.append((start, end))
    return segments or [(0, len(done_1d))]


def _select_dataset_episode(labels: np.ndarray, done: np.ndarray, min_length: int):
    best = None
    for env_id in range(labels.shape[1]):
        for start, end in _segments_from_done(done[:, env_id], min_length):
            lambda_values = labels[start:end, env_id, 0]
            coverage = float(lambda_values.max() - lambda_values.min())
            score = (coverage, end - start)
            if best is None or score > best[0]:
                best = (score, env_id, start, end)
    _, env_id, start, end = best
    return env_id, start, end


def _select_prediction_episode(preds, min_length: int):
    target = preds["target"]
    env_ids = preds["env_id"]
    done = preds["done"] if "done" in preds else np.zeros(len(target), dtype=np.bool_)
    best = None
    for env_id in np.unique(env_ids):
        idx = np.flatnonzero(env_ids == env_id)
        idx = idx[np.argsort(preds["time"][idx])]
        local_done = done[idx]
        for start, end in _segments_from_done(local_done, min_length):
            segment = idx[start:end]
            lambda_values = target[segment, 0]
            coverage = float(lambda_values.max() - lambda_values.min())
            score = (coverage, len(segment))
            if best is None or score > best[0]:
                best = (score, segment)
    return best[1]


def _save_label_curves(labels, done, label_names, figures_dir, max_points, min_episode_length):
    env_id, start, end = _select_dataset_episode(labels, done, min_episode_length)
    end = min(end, start + max_points)
    values = labels[start:end, env_id]
    steps = np.arange(start, end)

    rows = min(values.shape[-1], len(label_names))
    fig, axes = plt.subplots(rows, 1, figsize=(11, 1.8 * rows), sharex=True)
    axes = np.atleast_1d(axes)
    for idx, ax in enumerate(axes):
        ax.plot(steps, values[:, idx], linewidth=1.6)
        ax.set_ylabel(label_names[idx])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle(f"Medium-state labels/env {env_id}, episode [{start}, {end})")
    fig.tight_layout()
    path = figures_dir / "medium_label_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved={path}")


def _save_prediction_curve(preds, label_names, figures_dir, key, max_points, min_episode_length):
    if key not in label_names:
        print(f"[PLOT] skipped {key}: not present in label_names={label_names}")
        return
    idx = label_names.index(key)
    segment = _select_prediction_episode(preds, min_episode_length)[:max_points]
    order = np.argsort(preds["time"][segment])
    segment = segment[order]
    times = preds["time"][segment]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(times, preds["target"][segment, idx], label=f"true {key}", linewidth=2.0)
    ax.plot(times, preds["mlp_pred"][segment, idx], label="MLP", linewidth=1.4, alpha=0.85)
    ax.plot(times, preds["gru_pred"][segment, idx], label="GRU", linewidth=1.4, alpha=0.85)
    ax.set_xlabel("step")
    ax.set_ylabel(key)
    ax.set_title(f"{key}: true vs MLP vs GRU")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = figures_dir / f"{key}_prediction_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved={path}")


def _save_histogram(labels, label_names, figures_dir):
    lambda_values = labels[..., 0].reshape(-1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lambda_values, bins=30, alpha=0.85)
    ax.set_xlabel(label_names[0])
    ax.set_ylabel("count")
    ax.set_title("lambda distribution")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = figures_dir / "label_histogram_lambda.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"[PLOT] saved={path}")


def main():
    parser = argparse.ArgumentParser(description="Plot medium-state labels and predictions.")
    parser.add_argument("--dataset", type=str, default="runs/medium_dataset/amphibious_terrain_medium_dataset.npz")
    parser.add_argument("--predictions", type=str, default="runs/medium_dataset/amphibious_terrain_predictions.npz")
    parser.add_argument("--figures_dir", type=str, default="runs/medium_dataset/figures")
    parser.add_argument("--max_points", type=int, default=2000)
    parser.add_argument("--min_episode_length", type=int, default=50)
    args = parser.parse_args()

    figures_dir = pathlib.Path(args.figures_dir).expanduser()
    figures_dir.mkdir(parents=True, exist_ok=True)

    dataset = np.load(pathlib.Path(args.dataset).expanduser())
    labels = dataset["medium_state_label"].astype(np.float32)
    done = dataset["done"].astype(np.bool_) if "done" in dataset else np.zeros(labels.shape[:2], dtype=np.bool_)
    names = _label_names(dataset, labels.shape[-1])
    _save_label_curves(labels, done, names, figures_dir, args.max_points, args.min_episode_length)
    _save_histogram(labels, names, figures_dir)

    pred_path = pathlib.Path(args.predictions).expanduser()
    if not pred_path.exists():
        print(f"[PLOT] predictions not found, skipped comparison plots: {pred_path}")
        return
    preds = np.load(pred_path)
    pred_names = _label_names(preds, preds["target"].shape[-1])
    _save_prediction_curve(preds, pred_names, figures_dir, "eta_wheel", args.max_points, args.min_episode_length)
    _save_prediction_curve(preds, pred_names, figures_dir, "eta_thruster", args.max_points, args.min_episode_length)
    _save_prediction_curve(preds, pred_names, figures_dir, "drag_scale", args.max_points, args.min_episode_length)


if __name__ == "__main__":
    main()
