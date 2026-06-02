import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="datasets/lift_success_demos_50.hdf5")
parser.add_argument("--camera", type=str, default="agentview_image")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=8e-4)
parser.add_argument("--num_bins", type=int, default=256)
args = parser.parse_args()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.lift_token_dataset_v2 import LiftTokenDatasetV2
from src.models.mini_rt2_policy_v2 import MiniRT2TokenPolicyV2


# Important dims:
# x, y, z, roll, pitch, yaw, gripper
DIM_WEIGHTS = torch.tensor([3.0, 3.0, 4.0, 0.15, 0.15, 0.15, 6.0])


def weighted_token_loss(logits, targets, criterion, dim_weights):
    """
    logits: [B, 7, 256]
    targets: [B, 7]
    """
    losses = []

    for d in range(targets.shape[1]):
        dim_loss = criterion(logits[:, d, :], targets[:, d])
        losses.append(dim_loss * dim_weights[d])

    return torch.stack(losses).mean()


def per_dim_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean(dim=0), correct.mean()


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_dim_acc = torch.zeros(7, device=device)
    n = 0

    dim_weights = DIM_WEIGHTS.to(device)

    for batch in loader:
        images = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        proprio = batch["proprio"].to(device)
        tokens = batch["tokens"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images, text_ids, proprio)

            loss = weighted_token_loss(
                logits=logits,
                targets=tokens,
                criterion=criterion,
                dim_weights=dim_weights,
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        dim_acc, acc = per_dim_accuracy(logits, tokens)

        total_loss += loss.item()
        total_acc += acc.item()
        total_dim_acc += dim_acc
        n += 1

    return (
        total_loss / max(n, 1),
        total_acc / max(n, 1),
        (total_dim_acc / max(n, 1)).detach().cpu(),
    )


def main():
    Path("checkpoints").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = LiftTokenDatasetV2(
        hdf5_path=args.data,
        camera_key=args.camera,
        num_bins=args.num_bins,
    )

    print("Dataset:", args.data)
    print("Samples:", len(dataset))
    print("Image:", dataset[0]["image"].shape)
    print("Proprio:", dataset[0]["proprio"].shape)
    print("Example tokens:", dataset[0]["tokens"].tolist())
    print("Dim weights [x,y,z,roll,pitch,yaw,gripper]:", DIM_WEIGHTS.tolist())

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    model = MiniRT2TokenPolicyV2(
        vocab_size=128,
        num_bins=args.num_bins,
        action_dim=7,
        proprio_dim=10,
        hidden_dim=384,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_dim_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )

        val_loss, val_acc, val_dim_acc = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        dt = time.time() - t0

        # Score focuses on hard dims only: x,y,z,gripper
        hard_score = float((val_dim_acc[[0, 1, 2, 6]]).mean())

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"hard_score={hard_score:.3f} | "
            f"time={dt:.1f}s"
        )

        print(
            "  val_dim_acc "
            f"x={val_dim_acc[0]:.3f} "
            f"y={val_dim_acc[1]:.3f} "
            f"z={val_dim_acc[2]:.3f} "
            f"r={val_dim_acc[3]:.3f} "
            f"p={val_dim_acc[4]:.3f} "
            f"yaw={val_dim_acc[5]:.3f} "
            f"grip={val_dim_acc[6]:.3f}"
        )

        ckpt = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "num_bins": args.num_bins,
            "action_dim": 7,
            "proprio_dim": 10,
            "camera": args.camera,
            "data": args.data,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "val_dim_acc": val_dim_acc.tolist(),
            "hard_score": hard_score,
            "hidden_dim": 384,
            "dim_weights": DIM_WEIGHTS.tolist(),
        }

        torch.save(ckpt, "checkpoints/mini_rt2_v3_latest.pt")

        if hard_score > best_score:
            best_score = hard_score
            torch.save(ckpt, "checkpoints/mini_rt2_v3_best.pt")
            print("  saved best checkpoint")

    print("Done.")
    print("Best checkpoint: checkpoints/mini_rt2_v3_best.pt")


if __name__ == "__main__":
    main()
