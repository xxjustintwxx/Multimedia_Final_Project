"""
Generate a heatmap of contest scores over the (conf_thresh, margin_thresh) grid.
Saves: runs/<run-name>/threshold_heatmap.png
"""

import sys
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def score_thresholds(probs, labels, conf_thresh, margin_thresh):
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
    top1_class  = sorted_idx[:, 0]
    top1_conf   = probs[np.arange(len(probs)), top1_class]
    top2_conf   = probs[np.arange(len(probs)), sorted_idx[:, 1]]
    margin      = top1_conf - top2_conf

    score = 0
    for i in range(len(labels)):
        pred = top1_class[i]
        if pred == 0:
            continue
        if top1_conf[i] < conf_thresh or margin[i] < margin_thresh:
            continue
        score += 1 if pred == labels[i] else -2
    return score


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--conf-steps",   type=int, default=14)
    p.add_argument("--margin-steps", type=int, default=11)
    args = p.parse_args()

    run_dir      = _ROOT / "runs" / args.run_name
    weights_path = run_dir / "phase2_best.pt"
    figures_dir  = _ROOT / "figures"
    figures_dir.mkdir(exist_ok=True)
    out_path     = figures_dir / "threshold_heatmap.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = GestureClassifier().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    log.info(f"Loaded {weights_path}")

    val_ds = GestureDataset("val", augment=False)
    loader = DataLoader(val_ds, batch_size=512, shuffle=False,
                        num_workers=8, pin_memory=True)
    log.info("Collecting val-set probabilities …")
    probs, labels = collect_probs(model, loader, device)
    log.info(f"Val samples: {len(labels):,}")

    conf_grid   = np.linspace(0.30, 0.95, args.conf_steps)
    margin_grid = np.linspace(0.00, 0.50, args.margin_steps)

    # scores[i, j] = score at conf_grid[i], margin_grid[j]
    scores = np.zeros((len(conf_grid), len(margin_grid)), dtype=np.int32)
    for i, ct in enumerate(conf_grid):
        for j, mt in enumerate(margin_grid):
            scores[i, j] = score_thresholds(probs, labels, float(ct), float(mt))

    best_idx = np.unravel_index(np.argmax(scores), scores.shape)
    best_conf   = conf_grid[best_idx[0]]
    best_margin = margin_grid[best_idx[1]]
    best_score  = scores[best_idx]
    log.info(f"Best: conf={best_conf:.3f}  margin={best_margin:.3f}  score={best_score:+d}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        scores,
        aspect="auto",
        origin="lower",
        cmap="RdYlGn",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Contest score (+1/−2 formula)")

    # Axis ticks
    ax.set_xticks(range(len(margin_grid)))
    ax.set_xticklabels([f"{v:.2f}" for v in margin_grid], rotation=45, ha="right")
    ax.set_yticks(range(len(conf_grid)))
    ax.set_yticklabels([f"{v:.2f}" for v in conf_grid])

    ax.set_xlabel("margin_thresh")
    ax.set_ylabel("conf_thresh")
    ax.set_title(
        f"Threshold grid search — val contest score\n"
        f"Best: conf={best_conf:.3f}, margin={best_margin:.3f} → score={best_score:+d}",
        fontsize=11,
    )

    # Annotate each cell with its score
    for i in range(len(conf_grid)):
        for j in range(len(margin_grid)):
            ax.text(j, i, str(scores[i, j]),
                    ha="center", va="center", fontsize=6.5,
                    color="black")

    # Mark best cell with a red border
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
        linewidth=2, edgecolor="red", facecolor="none",
    )
    ax.add_patch(rect)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    log.info(f"Saved heatmap to {out_path}")


if __name__ == "__main__":
    main()
