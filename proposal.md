# Project Proposal
## Hand Gesture Classification on Edge Devices
**Course:** Multimedia (114-2) | **Challenge sponsor:** Microsoft

---

## 1. Problem Summary

Design a compact gesture classifier (≤ 10 MB) that receives a **cropped hand image** and **21 MediaPipe landmark coordinates** and outputs one of six classes:

| Class | Label |
|-------|-------|
| 0 | N/A (reject / unknown) |
| 1 | fist |
| 2 | like |
| 3 | ok |
| 4 | one |
| 5 | palm |

The classifier sits downstream of the TA-provided MediaPipe preprocessor; full-frame images are strictly prohibited as input.

---

## 2. Scoring Strategy

| Criterion | Points | Our target |
|-----------|--------|-----------|
| Model size ≤ 10 MB, Score = (10 − size_MB) × 3 | 30 | < 2 MB → ~24 pts |
| Basic Performance (HaGRIDv2 test set) | 20 | high recall on 5 classes, near-zero false triggers |
| Real-World Robustness (TA-shot dataset) | 40 | robust N/A rejection |
| Presentation | 30 | live demo + defense explanation |

**Key insight from scoring:** a false trigger costs **−2 pts** while a correct prediction earns only **+1 pt**. Robust N/A rejection is therefore as important as 5-class accuracy.

---

## 3. Proposed Architecture: Dual-Stream Fusion

```
cropped image (H×W×3)          landmarks (21×2 = 42 floats)
       │                                  │
  ┌────▼────────────┐             ┌───────▼──────┐
  │  Lightweight    │             │  Landmark    │
  │  CNN backbone   │             │  MLP         │
  │  (custom or     │             │  42→128→64   │
  │  MobileNetV2-   │             └───────┬──────┘
  │  0.25 trimmed)  │                     │
  │  → 64-dim feat  │                     │
  └────────┬────────┘                     │
           └──────────┬───────────────────┘
                      │  concat (128-dim)
                 ┌────▼────┐
                 │  Fusion │
                 │  head   │
                 │ 128→64→6│
                 └────┬────┘
                      │ softmax
               ┌──────▼──────┐
               │ N/A decision│  ← engineering heuristics applied here
               └─────────────┘
```

### 3.1 Image Branch

- **Backbone option A (preferred):** Custom 5-layer depthwise-separable CNN  
  Input: 64×64 RGB → ~0.3 MB weights
- **Backbone option B (fallback):** MobileNetV2 width=0.25, features truncated after `layer_7`  
  ImageNet-pretrained backbone is allowed; fine-tune final layers only

### 3.2 Landmark Branch

- Input: 21 (x, y) crop-relative coordinates → 42 floats
- Architecture: Linear(42→128) → BN → ReLU → Linear(128→64) → BN → ReLU
- Encodes structural hand shape independent of lighting/texture

### 3.3 Fusion & Classification Head

- Concatenate image and landmark features (128-dim total)
- Two FC layers (128→64→6) with dropout (p=0.3)
- Final softmax for probability distribution

### 3.4 N/A Rejection Strategy (Engineering Heuristics)

Because false triggers are penalized 2× more than correct predictions:

1. **Confidence gate:** if `max_softmax_prob < threshold` (e.g., 0.6), output class 0 (N/A)
2. **Top-2 margin:** if `prob[1st] − prob[2nd] < margin` (e.g., 0.2), output N/A
3. **Landmark geometry sanity checks:**
   - Finger extension ratio (spread vs. closed) to pre-screen ambiguous poses
   - Wrist–palm aspect ratio outside expected range → N/A
4. These thresholds will be tuned on a held-out validation split

---

## 4. Dataset Strategy

### 4.1 Target class samples
- Download **HaGRIDv2 512px** for the 5 target gestures: fist, like, ok, one, palm
- Use TA preprocessor to generate `(crop, landmarks)` pairs for all samples

### 4.2 N/A samples
- Sample single-hand, non-target gesture categories from HaGRIDv2 (e.g., call, four, mute, stop, etc.)
- Balance N/A count to roughly equal total target-class count to avoid class imbalance

### 4.3 Augmentation (to match TA test conditions)
Since TAs apply bbox jitter and blur at test time:
- **Bbox jitter:** random crop expansion ±10%
- **Gaussian blur:** kernel 3–7, σ 0.1–1.5
- **Color jitter:** brightness, contrast, saturation ±20%
- **Horizontal flip** (with mirrored landmark x-coordinates)
- **Random rotation** ±15°

---

## 5. Training Plan

| Phase | Details |
|-------|---------|
| Data prep | Preprocess full HaGRIDv2 splits with `hand_preprocess.py`, save `(crop, landmarks, label)` tuples |
| Pre-training (image branch) | Train CNN on 5-class + N/A with standard cross-entropy |
| Joint training | Train full dual-stream model end-to-end, weighted cross-entropy (up-weight N/A due to −2 penalty) |
| Threshold tuning | Grid search confidence threshold on validation set, optimizing for contest scoring formula |
| Quantization | Post-training INT8 quantization to further reduce model size |

**Loss function:** weighted cross-entropy with class weights reflecting the asymmetric scoring (+1 / −2).

---

## 6. Model Size Budget

| Component | Estimated size |
|-----------|---------------|
| CNN backbone (custom 5-layer DW-sep) | ~0.3 MB |
| Landmark MLP | ~0.1 MB |
| Fusion head | ~0.05 MB |
| **Total (FP32)** | **~0.45 MB** |
| After INT8 quantization | **~0.15 MB** |

Targeting well under 2 MB → model size score ~24/30.

---

## 7. Inference Interface

```python
# inference.py
def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    cropped_img: H×W×3 uint8 RGB
    landmarks:   (21, 2) float32, crop-relative
    returns:     int in {0,1,2,3,4,5}
    """
    # 1. Preprocess image (resize to 64×64, normalize)
    # 2. Forward pass through dual-stream model
    # 3. Apply N/A rejection heuristics on softmax output
    # 4. Return final_decision_class
```

All model paths are relative to `inference.py`. Weights loaded from `model/` directory. No runtime downloads.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Model size exceeds 10 MB | Aggressive pruning, quantization, or fallback to landmark-only MLP (< 0.2 MB) |
| High false trigger rate on real-world data | Tune confidence threshold conservatively; add landmark geometry guards |
| Augmented test images degrade accuracy | Apply matching augmentations during training |
| HaGRIDv2 download too large | Use 512px version only; stream-process or use a sampled subset for N/A class |

---

## 9. Allowed Resources

- HaGRIDv2 dataset (official)
- ImageNet-pretrained backbones (fine-tuned)
- MediaPipe Hand Landmarker (provided by TAs)
- **Not allowed:** pretrained models trained on HaGRID / any gesture recognition dataset
