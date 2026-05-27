"""
Threshold calibration: grid search over (conf_thresh, margin_thresh)
on the validation set, optimising for the contest scoring formula.
Saves best thresholds to model/thresholds.json.
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_ROOT  = Path(__file__).resolve().parent.parent
_TRAIN = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_TRAIN))

from model.architecture import GestureClassifier
from dataset import GestureDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def collect_probs(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, lms, labels in loader:
            imgs, lms = imgs.to(device), lms.to(device)
            probs = F.softmax(model(imgs, lms), dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs), np.concatenate(all_labels)


def score_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    conf_thresh: float,
    margin_thresh: float,
) -> int:
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
    top1_class  = sorted_idx[:, 0]
    top1_conf   = probs[np.arange(len(probs)), top1_class]
    top2_conf   = probs[np.arange(len(probs)), sorted_idx[:, 1]]
    margin      = top1_conf - top2_conf

    score = 0
    for i in range(len(labels)):
        pred = top1_class[i]
        if pred == 0:               # predicted N/A — no points
            continue
        if top1_conf[i] < conf_thresh:
            continue
        if margin[i] < margin_thresh:
            continue
        if pred == labels[i]:
            score += 1
        else:
            score -= 2
    return score


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, required=True,
                   help="Run folder under runs/ that contains phase2_best.pt")
    args = parse_args = p.parse_args()

    run_dir      = _ROOT / "runs" / args.run_name
    weights_path = run_dir / "phase2_best.pt"
    out_path     = _ROOT / "model" / "thresholds.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GestureClassifier().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    log.info(f"Loaded weights from {weights_path}")

    val_ds = GestureDataset("val", augment=False)
    loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)

    log.info("Collecting val-set softmax outputs …")
    probs, labels = collect_probs(model, loader, device)
    log.info(f"Val samples: {len(labels):,}")

    # Grid search
    conf_grid   = np.linspace(0.30, 0.95, 14)
    margin_grid = np.linspace(0.00, 0.50, 11)

    best_score = -999_999
    best_conf  = 0.5
    best_margin = 0.2
    results = []

    for ct in conf_grid:
        for mt in margin_grid:
            s = score_thresholds(probs, labels, float(ct), float(mt))
            results.append((float(ct), float(mt), s))
            if s > best_score:
                best_score  = s
                best_conf   = float(ct)
                best_margin = float(mt)

    log.info(f"\nBest thresholds: conf={best_conf:.3f}  margin={best_margin:.3f}  score={best_score:+d}")

    # Show top-10 results
    results.sort(key=lambda x: -x[2])
    log.info("Top 10 threshold combinations:")
    for ct, mt, s in results[:10]:
        log.info(f"  conf={ct:.3f}  margin={mt:.3f}  score={s:+d}")

    thresholds = {"conf_thresh": best_conf, "margin_thresh": best_margin}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
