"""Phase 2: joint end-to-end training of the full dual-stream model."""

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

_ROOT  = Path(__file__).resolve().parent.parent
_TRAIN = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_TRAIN))

from model.architecture import GestureClassifier, Phase1Model, model_size_mb
from dataset import GestureDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr-backbone",    type=float, default=1e-4,
                   help="LR for pre-trained image branch")
    p.add_argument("--lr-new",         type=float, default=5e-4,
                   help="LR for landmark branch + fusion head")
    p.add_argument("--workers",        type=int,   default=8)
    p.add_argument("--run-name",       type=str,   default=None,
                   help="Run folder under runs/. Must match the phase-1 run.")
    p.add_argument("--phase1-weights", type=str,   default=None,
                   help="Override path to phase1_best.pt. Defaults to runs/<run-name>/phase1_best.pt")
    return p.parse_args()


# ── scoring ──────────────────────────────────────────────────────────────────

def contest_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
    conf_thresh: float = 0.5,
    margin_thresh: float = 0.2,
) -> int:
    """
    +1 for correct target class prediction, −2 for false trigger.
    Predicting N/A (0) never adds or subtracts points.
    """
    top2_vals, top2_idx = probs.topk(2, dim=1)
    score = 0
    for i in range(len(labels)):
        pred = top2_idx[i, 0].item()
        conf = top2_vals[i, 0].item()
        margin = (top2_vals[i, 0] - top2_vals[i, 1]).item()

        # N/A after threshold gates
        if pred == 0 or conf < conf_thresh or margin < margin_thresh:
            continue
        if pred == labels[i].item():
            score += 1
        else:
            score -= 2
    return score


# ── class weights ─────────────────────────────────────────────────────────────

def compute_class_weights(dataset: GestureDataset) -> torch.Tensor:
    counts = np.bincount(dataset.labels, minlength=6).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)
    total  = counts.sum()
    # Inverse-frequency weights
    weights = total / (6.0 * counts)
    # Slight boost to N/A: encourages conservative predictions
    weights[0] *= 1.5
    return torch.tensor(weights, dtype=torch.float32)


# ── train / val ───────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = total_correct = total_score = n = 0
    all_probs  = []
    all_labels = []

    with torch.set_grad_enabled(train):
        for imgs, lms, labels in loader:
            imgs, lms, labels = imgs.to(device), lms.to(device), labels.to(device)
            logits = model(imgs, lms)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = F.softmax(logits.detach(), dim=1)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

            bs = labels.size(0)
            total_loss    += loss.item() * bs
            total_correct += (logits.argmax(1) == labels).sum().item()
            n             += bs

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)
    score = contest_score(all_probs, all_labels)

    return total_loss / n, total_correct / n, score


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("run_%m%d_%H%M")
    run_dir  = _ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    p1_weights = Path(args.phase1_weights) if args.phase1_weights else run_dir / "phase1_best.pt"

    # Save run config
    with open(run_dir / "phase2_args.json", "w") as f:
        json.dump(vars(args) | {"run_name": run_name, "phase1_weights_resolved": str(p1_weights)}, f, indent=2)

    fh = logging.FileHandler(run_dir / "phase2.log")
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
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    # Build full model
    model = GestureClassifier().to(device)
    log.info(f"GestureClassifier size: {model_size_mb(model):.3f} MB")

    # Load phase-1 CNN weights
    if p1_weights.exists():
        p1_state = torch.load(p1_weights, map_location="cpu")
        # Phase1Model state keys start with "image_branch." or "head."
        cnn_state = {
            k.replace("image_branch.", ""): v
            for k, v in p1_state.items()
            if k.startswith("image_branch.")
        }
        model.image_branch.load_state_dict(cnn_state)
        log.info(f"Loaded image branch from {p1_weights}")
    else:
        log.warning(f"Phase-1 weights not found at {p1_weights}, training from scratch")

    # Differential learning rates: lower LR for pre-trained image branch
    optimizer = torch.optim.Adam([
        {"params": model.image_branch.parameters(),    "lr": args.lr_backbone},
        {"params": model.landmark_branch.parameters(), "lr": args.lr_new},
        {"params": model.fusion.parameters(),          "lr": args.lr_new},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Weighted cross-entropy
    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    log.info(f"Class weights: {class_weights.tolist()}")

    best_val_score = -999_999
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_score = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        vl_loss, vl_acc, vl_score = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )
        scheduler.step()

        log.info(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"train loss={tr_loss:.4f} acc={tr_acc:.4f} score={tr_score:+d}  "
            f"val loss={vl_loss:.4f} acc={vl_acc:.4f} score={vl_score:+d}  "
            f"lr_bb={scheduler.get_last_lr()[0]:.2e}"
        )

        if vl_score > best_val_score:
            best_val_score = vl_score
            torch.save(model.state_dict(), run_dir / "phase2_best.pt")
            log.info(f"  ↑ saved best (val_score={best_val_score:+d})")

    torch.save(model.state_dict(), run_dir / "phase2_last.pt")
    log.info(f"Phase 2 done. Best val contest score: {best_val_score:+d}")
    log.info(f"Checkpoints saved to: {run_dir}")


if __name__ == "__main__":
    main()
