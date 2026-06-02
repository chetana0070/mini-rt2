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
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--num_bins", type=int, default=256)
parser.add_argument("--no_preview", action="store_true")
args = parser.parse_args()

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from src.data.lift_token_dataset import LiftTokenDataset
from src.models.mini_rt2_policy import MiniRT2TokenPolicy


def token_accuracy(logits, targets):
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float()
    return correct.mean().item(), preds


def show_preview(image_tensor, target_tokens, pred_tokens, epoch, step, loss, acc):
    img = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)

    # RGB to BGR for OpenCV and flip like previous robosuite camera display
    img = img[::-1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)

    text1 = f"Mini RT-2 training | epoch {epoch} step {step}"
    text2 = f"loss={loss:.4f} token_acc={acc:.3f}"
    text3 = f"target={target_tokens.tolist()}"
    text4 = f"pred  ={pred_tokens.tolist()}"

    cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(img, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(img, text3, (10, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(img, text4, (10, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    cv2.imshow("Mini RT-2 Training Preview", img)
    key = cv2.waitKey(1) & 0xFF

    return key == ord("q") or key == 27


def run_epoch(model, loader, optimizer, criterion, device, train=True, epoch=0, preview=False):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    stop_preview = False

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        text_ids = batch["text_ids"].to(device)
        tokens = batch["tokens"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(images, text_ids)

            loss = criterion(
                logits.reshape(-1, logits.shape[-1]),
                tokens.reshape(-1),
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        acc, preds = token_accuracy(logits, tokens)

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        if preview and step % 5 == 0:
            stop_preview = show_preview(
                image_tensor=images[0],
                target_tokens=tokens[0].detach().cpu(),
                pred_tokens=preds[0].detach().cpu(),
                epoch=epoch,
                step=step,
                loss=loss.item(),
                acc=acc,
            )

        if stop_preview:
            preview = False

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1)


def main():
    Path("checkpoints").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))

    dataset = LiftTokenDataset(
        hdf5_path=args.data,
        camera_key=args.camera,
        num_bins=args.num_bins,
    )

    print("Dataset:", args.data)
    print("Camera:", args.camera)
    print("Samples:", len(dataset))
    print("Image shape:", dataset[0]["image"].shape)
    print("Token shape:", dataset[0]["tokens"].shape)
    print("Example tokens:", dataset[0]["tokens"].tolist())

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

    model = MiniRT2TokenPolicy(
        vocab_size=128,
        num_bins=args.num_bins,
        action_dim=7,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    preview = not args.no_preview

    print()
    print("Training started.")
    print("Preview:", preview)
    print("Press q in preview window to close preview only.")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=True,
            epoch=epoch,
            preview=preview,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            train=False,
            epoch=epoch,
            preview=False,
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
            "camera": args.camera,
            "data": args.data,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
        }

        torch.save(ckpt, "checkpoints/mini_rt2_latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, "checkpoints/mini_rt2_best.pt")
            print("  saved best checkpoint")

    cv2.destroyAllWindows()

    print()
    print("Training done.")
    print("Best checkpoint: checkpoints/mini_rt2_best.pt")
    print("Latest checkpoint: checkpoints/mini_rt2_latest.pt")


if __name__ == "__main__":
    main()
