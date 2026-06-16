"""Render a qualitative figure of the classifier's two input modalities:
the MediaPipe RGB crop with its 21 crop-relative landmarks overlaid.

Reads the committed examples in sample_preprocessed_outputs/ and writes
report/figures/sample_inputs.png. Run from the repository root:

    python report/figures/make_sample_inputs.py
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
CROPS = REPO / "sample_preprocessed_outputs" / "crops"
LMS = REPO / "sample_preprocessed_outputs" / "landmarks"
OUT = REPO / "report" / "figures" / "sample_inputs.png"

# MediaPipe Hands 21-landmark skeleton connectivity.
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                               # palm base
]


def main() -> None:
    stems = sorted(p.stem for p in CROPS.glob("*.jpg"))[:4]
    n = len(stems)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.2))
    if n == 1:
        axes = [axes]

    for ax, stem in zip(axes, stems):
        img = np.asarray(Image.open(CROPS / f"{stem}.jpg").convert("RGB"))
        h, w = img.shape[:2]
        lm = np.load(LMS / f"{stem}.npy").astype(np.float32)  # (21, 2) in [0,1]
        px, py = lm[:, 0] * w, lm[:, 1] * h

        ax.imshow(img)
        for a, b in CONNECTIONS:
            ax.plot([px[a], px[b]], [py[a], py[b]],
                    color="#1bd96a", linewidth=1.3, alpha=0.9)
        ax.scatter(px, py, s=10, color="#ff3b3b", zorder=3,
                   edgecolors="white", linewidths=0.4)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.axis("off")

    fig.tight_layout(pad=0.3)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
