import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def demo_key(name):
    # demo_10 should come after demo_9
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def flatten_obs_array(arr, T):
    arr = np.asarray(arr)
    if arr.shape[0] != T:
        raise ValueError(f"Expected first dim {T}, got {arr.shape}")
    return arr.reshape(T, -1).astype(np.float32)


def make_features(demo):
    actions = np.asarray(demo["actions"], dtype=np.float32)
    T = actions.shape[0]
    obs = demo["obs"]

    parts = []

    # Low-dimensional privileged smoke-test features.
    for key in [
        "robot0_eef_pos",
        "robot0_gripper_qpos",
        "object_pos",
        "target_xy",
        "phase_id",
    ]:
        if key not in obs:
            raise KeyError(f"Missing obs/{key}")
        parts.append(flatten_obs_array(obs[key], T))

    # Helpful time signal so the BC can learn phase progression.
    t = np.linspace(0.0, 1.0, T, dtype=np.float32).reshape(T, 1)
    parts.append(t)

    X = np.concatenate(parts, axis=1).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, actions


class BCMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/ur5e_robomimic/processed/ur5e_milk_r140_success_only.hdf5")
    ap.add_argument("--out", default="runs/ur5e_bc/lowdim_milk_r140_bc.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X_by_demo = []
    Y_by_demo = []
    demo_names = []

    with h5py.File(dataset_path, "r") as f:
        names = sorted(f["data"].keys(), key=demo_key)

        for name in names:
            d = f["data"][name]
            X, Y = make_features(d)
            X_by_demo.append(X)
            Y_by_demo.append(Y)
            demo_names.append(name)

    n = len(demo_names)
    idx = np.arange(n)
    np.random.shuffle(idx)

    n_val = max(1, int(round(n * args.val_fraction)))
    val_idx = set(idx[:n_val].tolist())

    X_train, Y_train, X_val, Y_val = [], [], [], []

    for i, name in enumerate(demo_names):
        if i in val_idx:
            X_val.append(X_by_demo[i])
            Y_val.append(Y_by_demo[i])
        else:
            X_train.append(X_by_demo[i])
            Y_train.append(Y_by_demo[i])

    X_train = np.concatenate(X_train, axis=0)
    Y_train = np.concatenate(Y_train, axis=0)
    X_val = np.concatenate(X_val, axis=0)
    Y_val = np.concatenate(Y_val, axis=0)

    x_mean = X_train.mean(axis=0, keepdims=True)
    x_std = X_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-6] = 1.0

    y_mean = Y_train.mean(axis=0, keepdims=True)
    y_std = Y_train.std(axis=0, keepdims=True)
    y_std[y_std < 1e-6] = 1.0

    X_train_n = (X_train - x_mean) / x_std
    X_val_n = (X_val - x_mean) / x_std

    Y_train_n = (Y_train - y_mean) / y_std
    Y_val_n = (Y_val - y_mean) / y_std

    train_ds = TensorDataset(
        torch.from_numpy(X_train_n).float(),
        torch.from_numpy(Y_train_n).float(),
    )
    val_x = torch.from_numpy(X_val_n).float()
    val_y = torch.from_numpy(Y_val_n).float()

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BCMLP(X_train.shape[1], Y_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    print("dataset:", dataset_path)
    print("demos:", n, "train demos:", n - n_val, "val demos:", n_val)
    print("train samples:", X_train.shape[0], "val samples:", X_val.shape[0])
    print("obs_dim:", X_train.shape[1], "action_dim:", Y_train.shape[1])
    print("device:", device)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            train_losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            pred_val = model(val_x.to(device)).cpu()
            val_loss = float(loss_fn(pred_val, val_y).item())

            # action MAE in original action scale
            pred_val_orig = pred_val.numpy() * y_std + y_mean
            val_mae = np.mean(np.abs(pred_val_orig - Y_val), axis=0)

        train_loss = float(np.mean(train_losses))

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "obs_dim": X_train.shape[1],
                    "action_dim": Y_train.shape[1],
                    "x_mean": x_mean.astype(np.float32),
                    "x_std": x_std.astype(np.float32),
                    "y_mean": y_mean.astype(np.float32),
                    "y_std": y_std.astype(np.float32),
                    "dataset": str(dataset_path),
                    "demo_names": demo_names,
                    "val_demo_indices": sorted(list(val_idx)),
                },
                out_path,
            )

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} "
                f"train_mse={train_loss:.6f} "
                f"val_mse={val_loss:.6f} "
                f"val_action_mae={np.round(val_mae, 4)}"
            )

    print("best_val_mse:", best_val)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
