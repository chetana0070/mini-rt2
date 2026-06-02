import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="datasets/lift_success_demos.hdf5")
parser.add_argument("--camera", type=str, default="agentview_image")
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_bins", type=int, default=256)
args = parser.parse_args()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.lift_token_dataset_v2 import LiftTokenDatasetV2
from src.models.mini_rt2_policy_v2 import MiniRT2TokenPolicyV2


def token_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item()


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for batch in loader:
        images = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        proprio = batch["proprio"].to(device)
        tokens = batch["tokens"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images, text_ids, proprio)

            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                tokens.reshape(-1),
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        acc = token_accuracy(logits, tokens)

        total_loss += loss.item()
        total_acc += acc
        n += 1

    return total_loss / max(n, 1), total_acc / max(n, 1)


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

    print("Samples:", len(dataset))
    print("Image:", dataset[0]["image"].shape)
    print("Proprio:", dataset[0]["proprio"].shape)
    print("Tokens:", dataset[0]["tokens"].tolist())

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MiniRT2TokenPolicyV2(
        vocab_size=128,
        num_bins=args.num_bins,
        action_dim=7,
        proprio_dim=10,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True
        )

        val_loss, val_acc = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False
        )

        dt = time.time() - t0

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} | "
            f"time={dt:.1f}s"
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
        }

        torch.save(ckpt, "checkpoints/mini_rt2_v2_latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, "checkpoints/mini_rt2_v2_best.pt")
            print("  saved best checkpoint")

    print("Done.")
    print("Best checkpoint: checkpoints/mini_rt2_v2_best.pt")


if __name__ == "__main__":
    main()
