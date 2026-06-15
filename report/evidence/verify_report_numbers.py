"""Recompute model statistics quoted in the final report.

Run from the repository root:
    python report/evidence/verify_report_numbers.py
"""

import json
from pathlib import Path
import sys

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from model.architecture import GestureClassifier, model_size_mb  # noqa: E402


def main() -> None:
    model = GestureClassifier()
    weights_path = REPO_ROOT / "model" / "weights.pt"
    thresholds_path = REPO_ROOT / "model" / "thresholds.json"

    checkpoint_bytes = weights_path.stat().st_size
    checkpoint_mib = checkpoint_bytes / (1024**2)
    official_size_score = (10.0 - checkpoint_mib) * 3.0

    with thresholds_path.open(encoding="utf-8") as handle:
        thresholds = json.load(handle)

    values = {
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "tensor_footprint_mib": model_size_mb(model),
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_mib": checkpoint_mib,
        "size_score_using_checkpoint_mib": official_size_score,
        "confidence_threshold": thresholds["conf_thresh"],
        "margin_threshold": thresholds["margin_thresh"],
    }
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
