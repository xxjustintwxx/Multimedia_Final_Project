"""Phase 1: pretrain the image branch with a temporary classification head."""

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_ROOT  = Path(__file__).resolve().parent.parent
_TRAIN = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_TRAIN))

from model.architecture import Phase1Model, model_size_mb
from dataset import GestureDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=20)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--workers",    type=int,   default=8)
    p.add_argument("--run-name",   type=str,   default=None,
                   help="Run folder name under runs/. Auto-timestamped if omitted.")
    return p.parse_args()


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = total_acc = n = 0
    with torch.set_grad_enabled(train):
        for imgs, _, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy(logits, labels) * bs
            n          += bs
    return total_loss / n, total_acc / n


def main():
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("run_%m%d_%H%M")
    run_dir  = _ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    with open(run_dir / "phase1_args.json", "w") as f:
        json.dump(vars(args) | {"run_name": run_name}, f, indent=2)

    # File logger in addition to stdout
    fh = logging.FileHandler(run_dir / "phase1.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    log.info(f"Run directory: {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Datasets
    train_ds = GestureDataset("train", augment=True)
    val_ds   = GestureDataset("val",   augment=False)
    log.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_ds.balanced_sampler(),
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = Phase1Model().to(device)
    log.info(f"Phase1Model size: {model_size_mb(model):.3f} MB")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        log.info(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f}  "
            f"val loss={vl_loss:.4f} acc={vl_acc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), run_dir / "phase1_best.pt")
            log.info(f"  ↑ saved best (val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), run_dir / "phase1_last.pt")
    log.info(f"Phase 1 done. Best val acc: {best_val_acc:.4f}")
    log.info(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()
