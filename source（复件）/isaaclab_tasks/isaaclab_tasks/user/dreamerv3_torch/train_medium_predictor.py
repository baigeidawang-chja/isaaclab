"""Train offline MLP and GRU predictors for Amphibious medium-state labels."""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_LABEL_NAMES = [
    "eta_wheel",
    "eta_thruster",
]


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, obs):
        return self.net(obs)


class GRUPredictor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim))

    def forward(self, seq):
        out, _ = self.gru(seq)
        return self.head(out[:, -1])


def _stats(x: np.ndarray, eps: float = 1e-6):
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    return mean, np.maximum(std, eps)


def _make_sequences(obs, action, label, window: int, start_t: int, end_t: int, done=None):
    seqs, targets, target_obs, times, env_ids, target_done = [], [], [], [], [], []
    for env_id in range(obs.shape[1]):
        for t in range(max(window - 1, start_t), end_t):
            obs_hist = obs[t - window + 1 : t + 1, env_id]
            act_hist = action[t - window + 1 : t + 1, env_id]
            seqs.append(np.concatenate([obs_hist, act_hist], axis=-1))
            targets.append(label[t, env_id])
            target_obs.append(obs[t, env_id])
            times.append(t)
            env_ids.append(env_id)
            target_done.append(False if done is None else bool(done[t, env_id]))
    return (
        np.asarray(seqs, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(target_obs, dtype=np.float32),
        np.asarray(times, dtype=np.int64),
        np.asarray(env_ids, dtype=np.int64),
        np.asarray(target_done, dtype=np.bool_),
    )


def _train_model(model, loader, device, epochs: int, lr: float):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        losses = []
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        print(f"[TRAIN] {model.__class__.__name__} epoch={epoch + 1}/{epochs} loss={np.mean(losses):.6f}")


@torch.no_grad()
def _predict(model, x: np.ndarray, device):
    model.eval()
    preds = []
    for start in range(0, len(x), 4096):
        xb = torch.as_tensor(x[start : start + 4096], device=device, dtype=torch.float32)
        preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _print_metrics(name: str, pred: np.ndarray, target: np.ndarray, label_names):
    report_keys = ["eta_wheel", "eta_thruster"]
    transition = np.zeros(target.shape[0], dtype=bool)
    if "eta_wheel" in label_names:
        eta_wheel = target[:, label_names.index("eta_wheel")]
        transition |= (eta_wheel > 0.1) & (eta_wheel < 0.9)
    if "eta_thruster" in label_names:
        eta_thruster = target[:, label_names.index("eta_thruster")]
        transition |= (eta_thruster > 0.1) & (eta_thruster < 0.9)
    transition = transition if np.any(transition) else np.ones_like(transition, dtype=bool)
    for key in report_keys:
        if key not in label_names:
            continue
        idx = label_names.index(key)
        err = pred[:, idx] - target[:, idx]
        mae = np.mean(np.abs(err))
        mse = np.mean(err**2)
        transition_mae = np.mean(np.abs(err[transition]))
        print(f"[METRIC] {name} {key} MAE={mae:.6f} MSE={mse:.6f} transition_MAE={transition_mae:.6f}")


def main():
    parser = argparse.ArgumentParser(description="Train medium-state MLP and GRU predictors.")
    parser.add_argument("--dataset", type=str, default="runs/medium_dataset/amphibious_terrain_medium_dataset.npz")
    parser.add_argument("--output", type=str, default="runs/medium_dataset/amphibious_terrain_predictions.npz")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = np.load(pathlib.Path(args.dataset).expanduser())
    obs = data["obs"].astype(np.float32)
    action = data["action"].astype(np.float32)
    label = data["medium_state_label"].astype(np.float32)
    done = data["done"].astype(np.bool_) if "done" in data else None
    if "label_names" in data:
        label_names = [str(x) for x in data["label_names"].tolist()]
    else:
        label_names = DEFAULT_LABEL_NAMES[: label.shape[-1]]
    split_t = max(int(obs.shape[0] * 0.8), int(args.window))

    obs_mean, obs_std = _stats(obs[:split_t].reshape(-1, obs.shape[-1]))
    act_mean, act_std = _stats(action[:split_t].reshape(-1, action.shape[-1]))
    label_mean, label_std = _stats(label[:split_t].reshape(-1, label.shape[-1]))

    mlp_x_train = ((obs[:split_t].reshape(-1, obs.shape[-1]) - obs_mean) / obs_std).astype(np.float32)
    mlp_y_train = ((label[:split_t].reshape(-1, label.shape[-1]) - label_mean) / label_std).astype(np.float32)

    seq_train, y_train, _, _, _, _ = _make_sequences(
        obs, action, label, args.window, args.window - 1, split_t, done=done
    )
    seq_test, y_test, target_obs_test, times_test, env_ids_test, done_test = _make_sequences(
        obs, action, label, args.window, split_t, obs.shape[0], done=done
    )
    seq_mean = np.concatenate([obs_mean.reshape(-1), act_mean.reshape(-1)]).astype(np.float32)
    seq_std = np.concatenate([obs_std.reshape(-1), act_std.reshape(-1)]).astype(np.float32)
    seq_train_n = ((seq_train - seq_mean) / seq_std).astype(np.float32)
    seq_test_n = ((seq_test - seq_mean) / seq_std).astype(np.float32)
    y_train_n = ((y_train - label_mean) / label_std).astype(np.float32)
    target_obs_test_n = ((target_obs_test - obs_mean) / obs_std).astype(np.float32)

    device = torch.device(args.device)
    mlp = MLP(obs.shape[-1], label.shape[-1], args.hidden).to(device)
    gru = GRUPredictor(obs.shape[-1] + action.shape[-1], label.shape[-1], args.hidden).to(device)

    mlp_loader = DataLoader(
        TensorDataset(torch.from_numpy(mlp_x_train), torch.from_numpy(mlp_y_train)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    gru_loader = DataLoader(
        TensorDataset(torch.from_numpy(seq_train_n), torch.from_numpy(y_train_n)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    _train_model(mlp, mlp_loader, device, args.epochs, args.lr)
    _train_model(gru, gru_loader, device, args.epochs, args.lr)

    mlp_pred = _predict(mlp, target_obs_test_n, device) * label_std + label_mean
    gru_pred = _predict(gru, seq_test_n, device) * label_std + label_mean
    _print_metrics("MLP", mlp_pred, y_test, label_names)
    _print_metrics("GRU", gru_pred, y_test, label_names)

    output = pathlib.Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        target=y_test,
        mlp_pred=mlp_pred.astype(np.float32),
        gru_pred=gru_pred.astype(np.float32),
        time=times_test,
        env_id=env_ids_test,
        done=done_test,
        label_names=np.asarray(label_names),
    )
    print(f"[TRAIN] saved predictions={output}")


if __name__ == "__main__":
    main()
