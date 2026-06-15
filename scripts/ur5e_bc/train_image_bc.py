import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def demo_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def lowdim_for_step(demo, step, horizon):
    obs = demo["obs"]

    eef = np.asarray(obs["robot0_eef_pos"][step]).reshape(-1).astype(np.float32)
    gq = np.asarray(obs["robot0_gripper_qpos"][step]).reshape(-1).astype(np.float32)
    phase = np.asarray(obs["phase_id"][step]).reshape(-1).astype(np.float32)

    t = np.array([step / max(1, horizon - 1)], dtype=np.float32)

    return np.concatenate([eef, gq, phase, t], axis=0).astype(np.float32)


def compute_stats(dataset_path, train_demo_names):
    lows = []
    acts = []

    with h5py.File(dataset_path, "r") as f:
        for name in train_demo_names:
            demo = f["data"][name]
            T = demo["actions"].shape[0]

            for step in range(T):
                lows.append(lowdim_for_step(demo, step, T))
                acts.append(np.asarray(demo["actions"][step], dtype=np.float32))

    lows = np.stack(lows, axis=0)
    acts = np.stack(acts, axis=0)

    low_mean = lows.mean(axis=0, keepdims=True).astype(np.float32)
    low_std = lows.std(axis=0, keepdims=True).astype(np.float32)
    low_std[low_std < 1e-6] = 1.0

    act_mean = acts.mean(axis=0, keepdims=True).astype(np.float32)
    act_std = acts.std(axis=0, keepdims=True).astype(np.float32)
    act_std[act_std < 1e-6] = 1.0

    return low_mean, low_std, act_mean, act_std


class ImageBCDataset(Dataset):
    def __init__(self, dataset_path, demo_names, low_mean, low_std, act_mean, act_std):
        self.dataset_path = str(dataset_path)
        self.demo_names = list(demo_names)
        self.low_mean = low_mean
        self.low_std = low_std
        self.act_mean = act_mean
        self.act_std = act_std

        self.samples = []
        with h5py.File(self.dataset_path, "r") as f:
            for name in self.demo_names:
                T = f["data"][name]["actions"].shape[0]
                for step in range(T):
                    self.samples.append((name, step, T))

        self._file = None

    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.dataset_path, "r")
        return self._file

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        f = self._get_file()
        name, step, T = self.samples[idx]
        demo = f["data"][name]

        img = np.asarray(demo["obs"]["agentview_image"][step])

        if img.ndim != 3:
            raise ValueError(f"Bad image shape: {img.shape}")

        if img.shape[-1] == 3:
            img = np.transpose(img, (2, 0, 1))
        elif img.shape[0] == 3:
            pass
        else:
            raise ValueError(f"Bad image channel shape: {img.shape}")

        img = np.ascontiguousarray(img).astype(np.float32)

        if img.max() > 2.0:
            img = img / 255.0

        low = lowdim_for_step(demo, step, T)
        low = (low.reshape(1, -1) - self.low_mean) / self.low_std
        low = low.reshape(-1).astype(np.float32)

        action = np.asarray(demo["actions"][step], dtype=np.float32)
        action_n = (action.reshape(1, -1) - self.act_mean) / self.act_std
        action_n = action_n.reshape(-1).astype(np.float32)

        return torch.from_numpy(img), torch.from_numpy(low), torch.from_numpy(action_n)


class ImageBCPolicy(nn.Module):
    def __init__(self, low_dim, action_dim):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.low_mlp = nn.Sequential(
            nn.Linear(low_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, image, low):
        img_feat = self.cnn(image)
        low_feat = self.low_mlp(low)
        x = torch.cat([img_feat, low_feat], dim=-1)
        return self.head(x)


def resize_images(images, image_size):
    if images.shape[-1] == image_size and images.shape[-2] == image_size:
        return images
    return F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="datasets/ur5e_robomimic/processed/ur5e_milk_r140_success_only.hdf5")
    ap.add_argument("--out", default="runs/ur5e_bc/image_milk_r140_bc.pt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--image_size", type=int, default=128)
    ap.add_argument("--val_fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        demo_names = sorted(f["data"].keys(), key=demo_key)

    n = len(demo_names)
    order = np.arange(n)
    np.random.shuffle(order)

    n_val = max(1, int(round(n * args.val_fraction)))
    val_ids = set(order[:n_val].tolist())

    train_demo_names = [name for i, name in enumerate(demo_names) if i not in val_ids]
    val_demo_names = [name for i, name in enumerate(demo_names) if i in val_ids]

    low_mean, low_std, act_mean, act_std = compute_stats(dataset_path, train_demo_names)

    train_ds = ImageBCDataset(dataset_path, train_demo_names, low_mean, low_std, act_mean, act_std)
    val_ds = ImageBCDataset(dataset_path, val_demo_names, low_mean, low_std, act_mean, act_std)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    low_dim = low_mean.shape[1]
    action_dim = act_mean.shape[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ImageBCPolicy(low_dim=low_dim, action_dim=action_dim).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    print("dataset:", dataset_path)
    print("train demos:", train_demo_names)
    print("val demos:", val_demo_names)
    print("train samples:", len(train_ds), "val samples:", len(val_ds))
    print("low_dim:", low_dim, "action_dim:", action_dim)
    print("image_size:", args.image_size)
    print("device:", device)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for image, low, action_n in train_loader:
            image = image.to(device, non_blocking=True)
            low = low.to(device, non_blocking=True)
            action_n = action_n.to(device, non_blocking=True)

            image = resize_images(image, args.image_size)

            pred = model(image, low)
            loss = loss_fn(pred, action_n)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        mae_sum = np.zeros((action_dim,), dtype=np.float64)
        mae_count = 0

        with torch.no_grad():
            for image, low, action_n in val_loader:
                image = image.to(device, non_blocking=True)
                low = low.to(device, non_blocking=True)
                action_n = action_n.to(device, non_blocking=True)

                image = resize_images(image, args.image_size)

                pred_n = model(image, low)
                val_loss = loss_fn(pred_n, action_n)
                val_losses.append(float(val_loss.item()))

                pred = pred_n.cpu().numpy() * act_std + act_mean
                true = action_n.cpu().numpy() * act_std + act_mean
                mae_sum += np.abs(pred - true).sum(axis=0)
                mae_count += pred.shape[0]

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_mae = mae_sum / max(1, mae_count)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "low_dim": low_dim,
                    "action_dim": action_dim,
                    "low_mean": low_mean,
                    "low_std": low_std,
                    "act_mean": act_mean,
                    "act_std": act_std,
                    "dataset": str(dataset_path),
                    "train_demo_names": train_demo_names,
                    "val_demo_names": val_demo_names,
                    "image_size": args.image_size,
                    "inputs": [
                        "agentview_image",
                        "robot0_eef_pos",
                        "robot0_gripper_qpos",
                        "phase_id",
                        "time",
                    ],
                    "excluded_privileged_inputs": [
                        "object_pos",
                        "target_xy",
                        "yolo_world",
                    ],
                },
                out_path,
            )

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
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
