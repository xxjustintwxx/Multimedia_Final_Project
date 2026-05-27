"""Export: copy final weights to model/ and verify size is under budget."""

import sys
import shutil
import logging
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from model.architecture import GestureClassifier, model_size_mb

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

MAX_SIZE_MB = 10.0
TARGET_SIZE_MB = 2.0


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", type=str, required=True,
                   help="Run folder under runs/ that contains phase2_best.pt")
    args = p.parse_args()

    src = _ROOT / "runs" / args.run_name / "phase2_best.pt"
    dst = _ROOT / "model" / "weights.pt"
    dst.parent.mkdir(parents=True, exist_ok=True)

    # Load and verify
    device = torch.device("cpu")
    model = GestureClassifier()
    model.load_state_dict(torch.load(src, map_location=device))
    model.eval()

    in_memory_mb = model_size_mb(model)
    log.info(f"In-memory model size (FP32): {in_memory_mb:.4f} MB")

    # Copy weights file
    shutil.copy2(src, dst)
    file_mb = dst.stat().st_size / (1024 ** 2)
    log.info(f"Saved weights file: {dst}  ({file_mb:.4f} MB)")

    # Scoring preview
    if file_mb > MAX_SIZE_MB:
        log.error(f"FAIL: model size {file_mb:.2f} MB exceeds 10 MB limit!")
    else:
        pts = (MAX_SIZE_MB - file_mb) * 3
        log.info(f"Size score: ({MAX_SIZE_MB} - {file_mb:.3f}) × 3 = {pts:.2f} / 30 pts")
        if file_mb > TARGET_SIZE_MB:
            log.warning(f"Above {TARGET_SIZE_MB} MB target — consider quantization")

    # Smoke-test inference
    dummy_img = torch.zeros(1, 3, 64, 64)
    dummy_lm  = torch.zeros(1, 42)
    with torch.no_grad():
        out = model(dummy_img, dummy_lm)
    log.info(f"Smoke test output shape: {out.shape}  (expected [1, 6])")
    assert out.shape == (1, 6), "Output shape mismatch!"

    log.info("Export complete.")


if __name__ == "__main__":
    main()
