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
parser.add_argument("--epochs", type=int, default=80)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=8e-4)
parser.add_argument("--num_bins", type=int, default=256)
args = parser.parse_args()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.lift_token_dataset_v4 import LiftTokenDatasetV4
from src.models.mini_rt2_policy_v4 import MiniRT2StagePolicyV4


DIM_WEIGHTS = torch.tensor([3.0, 3.0, 4.0, 0.10, 0.10, 0.10, 6.0])


def weighted_token_loss(logits, targets, criterion, dim_weights):
    losses = []

    for d in range(targets.shape[1]):
        dim_loss = criterion(logits[:, d, :], targets[:, d])
        losses.append(dim_loss * dim_weights[d])

    return torch.stack(losses).mean()


def per_dim_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean(dim=0), correct.mean()


def run_epoch(model, loader, optimizer, token_criterion, stage_criterion, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_token_acc = 0.0
    total_stage_acc = 0.0
    total_dim_acc = torch.zeros(7, device=device)

    n = 0
    dim_weights = DIM_WEIGHTS.to(device)

    for batch in loader:
        images = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        proprio = batch["proprio"].to(device)
        stage_id = batch["stage_id"].to(device)
        tokens = batch["tokens"].to(device)

        with torch.set_grad_enabled(train):
            action_logits, stage_logits = model(images, text_ids, proprio, stage_id)

            action_loss = weighted_token_loss(
                logits=action_logits,
                targets=tokens,
                criterion=token_criterion,
                dim_weights=dim_weights,
            )

            stage_loss = stage_criterion(stage_logits, stage_id)

            loss = action_loss + 0.5 * stage_loss

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        dim_acc, token_acc = per_dim_accuracy(action_logits, tokens)
        stage_acc = (stage_logits.argmax(dim=-1) == stage_id).float().mean()

        total_loss += loss.item()
        total_token_acc += token_acc.item()
        total_stage_acc += stage_acc.item()
        total_dim_acc += dim_acc
        n += 1

    return (
        total_loss / max(n, 1),
        total_token_acc / max(n, 1),
        total_stage_acc / max(n, 1),
        (total_dim_acc / max(n, 1)).detach().cpu(),
    )


def main():
    Path("checkpoints").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    dataset = LiftTokenDatasetV4(
        hdf5_path=args.data,
        camera_key=args.camera,
        num_bins=args.num_bins,
    )

    print("Dataset:", args.data)
    print("Samples:", len(dataset))
    print("Image:", dataset[0]["image"].shape)
    print("Proprio:", dataset[0]["proprio"].shape)
    print("Stage id:", dataset[0]["stage_id"].item())
    print("Example tokens:", dataset[0]["tokens"].tolist())
    print("Dim weights:", DIM_WEIGHTS.tolist())

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = MiniRT2StagePolicyV4(
        vocab_size=128,
        num_bins=args.num_bins,
        action_dim=7,
        num_stages=5,
        proprio_dim=10,
        hidden_dim=384,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    token_criterion = nn.CrossEntropyLoss()
    stage_criterion = nn.CrossEntropyLoss()

    best_score = -1.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_token_acc, train_stage_acc, train_dim_acc = run_epoch(
            model, train_loader, optimizer, token_criterion, stage_criterion, device, train=True
        )

        val_loss, val_token_acc, val_stage_acc, val_dim_acc = run_epoch(
            model, val_loader, optimizer, token_criterion, stage_criterion, device, train=False
        )

        dt = time.time() - t0

        hard_score = float((val_dim_acc[[0, 1, 2, 6]]).mean())
        combined_score = 0.8 * hard_score + 0.2 * val_stage_acc

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_tok={train_token_acc:.3f} train_stage={train_stage_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_tok={val_token_acc:.3f} val_stage={val_stage_acc:.3f} | "
            f"hard={hard_score:.3f} score={combined_score:.3f} | "
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
            "num_stages": 5,
            "hidden_dim": 384,
            "camera": args.camera,
            "data": args.data,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_token_acc": val_token_acc,
            "val_stage_acc": val_stage_acc,
            "val_dim_acc": val_dim_acc.tolist(),
            "hard_score": hard_score,
            "combined_score": combined_score,
            "dim_weights": DIM_WEIGHTS.tolist(),
        }

        torch.save(ckpt, "checkpoints/mini_rt2_v4_latest.pt")

        if combined_score > best_score:
            best_score = combined_score
            torch.save(ckpt, "checkpoints/mini_rt2_v4_best.pt")
            print("  saved best checkpoint")

    print("Done.")
    print("Best checkpoint: checkpoints/mini_rt2_v4_best.pt")


if __name__ == "__main__":
    main()
