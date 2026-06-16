"""Ablation: landmark-only baseline.

Trains a classifier that uses ONLY the 21 MediaPipe landmarks (no image
branch), so the contribution of each modality can be quantified:

    image-only   -> Phase 1 (train/train_phase1.py),  ~86.15% val acc
    landmark-only -> THIS script
    fusion        -> Phase 2 (train/train_phase2.py),  ~96.75% val acc

The model reuses LandmarkBranch from model/architecture.py and adds the same
6-way head as the full model. Training mirrors Phase 2 (epochs, batch size,
inverse-frequency class weights with the 1.5x N/A boost, cosine annealing,
contest-score tracking) so the comparison is apples-to-apples. No existing
file is modified.

Run from the repository root (dataset must be preprocessed, same as Phase 2):

    python train/ablation_landmark_only.py --run-name ablation_lm

Best validation accuracy and contest score are printed and logged; the
best-accuracy checkpoint is saved to runs/<run-name>/landmark_only_best.pt.
"""

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

from model.architecture import LandmarkBranch, model_size_mb
from dataset import GestureDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


class LandmarkOnlyModel(nn.Module):
    """LandmarkBranch (42 -> 64) followed by the same 6-way head as fusion."""
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.landmark_branch = LandmarkBranch()
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 6),
        )

    def forward(self, lm: torch.Tensor) -> torch.Tensor:
        return self.head(self.landmark_branch(lm))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=30)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--workers",    type=int,   default=8)
    p.add_argument("--run-name",   type=str,   default=None,
                   help="Run folder under runs/. Auto-timestamped if omitted.")
    return p.parse_args()


# ── scoring (identical to train_phase2.contest_score) ─────────────────────────

def contest_score(
    probs: torch.Tensor,
    labels: torch.Tensor,
    conf_thresh: float = 0.5,
    margin_thresh: float = 0.2,
) -> int:
    """+1 correct target, -2 false trigger, 0 for N/A output."""
    top2_vals, top2_idx = probs.topk(2, dim=1)
    score = 0
    for i in range(len(labels)):
        pred   = top2_idx[i, 0].item()
        conf   = top2_vals[i, 0].item()
        margin = (top2_vals[i, 0] - top2_vals[i, 1]).item()
        if pred == 0 or conf < conf_thresh or margin < margin_thresh:
            continue
        if pred == labels[i].item():
            score += 1
        else:
            score -= 2
    return score


def compute_class_weights(dataset: GestureDataset) -> torch.Tensor:
    """Inverse-frequency class weights with a 1.5x boost on N/A (as in Phase 2)."""
    counts = np.bincount(dataset.labels, minlength=6).astype(np.float64)
    counts = np.where(counts == 0, 1, counts)
    total  = counts.sum()
    weights = total / (6.0 * counts)
    weights[0] *= 1.5
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = total_correct = n = 0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(train):
        for _imgs, lms, labels in loader:
            lms, labels = lms.to(device), labels.to(device)
            logits = model(lms)
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


def main():
    args = parse_args()

    run_name = args.run_name or datetime.now().strftime("ablation_lm_%m%d_%H%M")
    run_dir  = _ROOT / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "ablation_lm_args.json", "w") as f:
        json.dump(vars(args) | {"run_name": run_name}, f, indent=2)

    fh = logging.FileHandler(run_dir / "ablation_lm.log")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(fh)

    log.info(f"Run directory: {run_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    train_ds = GestureDataset("train", augment=True)
    val_ds   = GestureDataset("val",   augment=False)
    log.info(f"Train: {len(train_ds):,}  Val: {len(val_ds):,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = LandmarkOnlyModel().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"LandmarkOnlyModel: {n_params:,} params, {model_size_mb(model):.3f} MB")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    class_weights = compute_class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    log.info(f"Class weights: {class_weights.tolist()}")

    best_val_acc   = 0.0
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
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        best_val_score = max(best_val_score, vl_score)
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), run_dir / "landmark_only_best.pt")
            log.info(f"  ↑ saved best (val_acc={best_val_acc:.4f})")

    torch.save(model.state_dict(), run_dir / "landmark_only_last.pt")
    log.info("Landmark-only ablation done.")
    log.info(f"  best val accuracy     : {best_val_acc:.4f}  ({100*best_val_acc:.2f}%)")
    log.info(f"  best val contest score: {best_val_score:+d}")
    log.info(f"Checkpoints saved to: {run_dir}")
    log.info("Report these two numbers as the landmark-only ablation row.")


if __name__ == "__main__":
    main()
