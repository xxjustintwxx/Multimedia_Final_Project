"""
Submission entry point — implements the required predict() interface.

predict(cropped_img, landmarks) -> int  in {0, 1, 2, 3, 4, 5}
  0 = N/A   1 = fist   2 = like   3 = ok   4 = one   5 = palm
"""

import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ── paths (all relative to this file) ────────────────────────────────────────
_DIR         = Path(__file__).parent
_WEIGHTS     = _DIR / "model" / "weights.pt"
_THRESHOLDS  = _DIR / "model" / "thresholds.json"
sys.path.insert(0, str(_DIR))

from model.architecture import GestureClassifier

# ── constants ─────────────────────────────────────────────────────────────────
IMG_SIZE = 64
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)

_PREPROCESS = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

# ── load model once at import time ────────────────────────────────────────────
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model  = GestureClassifier()
_model.load_state_dict(torch.load(_WEIGHTS, map_location=_device))
_model.to(_device).eval()

# ── load calibrated thresholds ────────────────────────────────────────────────
if _THRESHOLDS.exists():
    with open(_THRESHOLDS) as f:
        _thr = json.load(f)
    _CONF_THRESH   = float(_thr["conf_thresh"])
    _MARGIN_THRESH = float(_thr["margin_thresh"])
else:
    # Fallback defaults if calibration hasn't been run yet
    _CONF_THRESH   = 0.60
    _MARGIN_THRESH = 0.20


# ── landmark geometry sanity check ───────────────────────────────────────────

def _landmarks_valid(lm: np.ndarray) -> bool:
    """Return False if landmarks look like a failed/degenerate detection."""
    # All landmarks collapsed to a tiny region (detection noise)
    span = lm.max(axis=0) - lm.min(axis=0)  # (2,)
    if max(span[0], span[1]) < 0.05:
        return False
    return True


# ── predict ───────────────────────────────────────────────────────────────────

def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    Parameters
    ----------
    cropped_img : np.ndarray  H×W×3  uint8 RGB
    landmarks   : np.ndarray  (21, 2) float32, crop-relative [0, 1]

    Returns
    -------
    int in {0, 1, 2, 3, 4, 5}
    """
    # ── landmark sanity gate ──────────────────────────────────────────────────
    lm = np.asarray(landmarks, dtype=np.float32)
    if not _landmarks_valid(lm):
        return 0

    # ── preprocess image ──────────────────────────────────────────────────────
    pil_img = Image.fromarray(np.asarray(cropped_img, dtype=np.uint8)).convert("RGB")
    img_t   = _PREPROCESS(pil_img).unsqueeze(0).to(_device)   # (1, 3, 64, 64)
    lm_t    = torch.from_numpy(lm.flatten()).unsqueeze(0).to(_device)  # (1, 42)

    # ── forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = _model(img_t, lm_t)                    # (1, 6)
        probs  = F.softmax(logits, dim=1)[0]            # (6,)

    # ── N/A rejection heuristics ─────────────────────────────────────────────
    top2_vals, top2_idx = probs.topk(2)
    pred_class = top2_idx[0].item()
    confidence = top2_vals[0].item()
    margin     = (top2_vals[0] - top2_vals[1]).item()

    if pred_class == 0:
        return 0
    if confidence < _CONF_THRESH:
        return 0
    if margin < _MARGIN_THRESH:
        return 0

    return int(pred_class)
