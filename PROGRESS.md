# Multimedia Final Project — Hand Gesture Classification on Edge Devices
**Course:** Multimedia (114-2) | **Challenge sponsor:** Microsoft  
**Leaderboard evaluations:** Mon / Wed / Fri / Sun, starting May 29

---

## Overview

Build a compact gesture classifier (≤ 10 MB) that takes a **MediaPipe-cropped hand image** and **21 landmark coordinates** and outputs one of six classes:

| Label | Gesture |
|-------|---------|
| 0 | N/A (reject / unknown) |
| 1 | fist |
| 2 | like |
| 3 | ok |
| 4 | one |
| 5 | palm |

**Key scoring rule:** false trigger = **−2 pts**, correct target prediction = **+1 pt**.  
Robust N/A rejection is as important as 5-class accuracy.

---

## Scoring Breakdown (120 pts total)

| Criterion | Points | Formula / Notes |
|-----------|--------|-----------------|
| Model Size ≤ 10 MB | 30 | `(10 − size_MB) × 3`; target < 2 MB → ~24 pts |
| Basic Performance (HaGRIDv2 test) | 20 | +1 correct, −2 false trigger; TAs apply bbox jitter + blur |
| Real-World Robustness (TA-shot dataset) | 40 | 50 N/A interference + 50 target-class images; same scoring |
| Presentation | 30 | Live demo + defense explanation |

---

## Submission Format

```
team_X.zip
├── inference.py          ← must implement predict(crop, landmarks) → int
├── model/
│   └── weights_int8.pt   ← quantized model weights
├── requirements.txt      ← inference-only dependencies (Colab-compatible)
└── README.md             ← environment setup and usage notes
```

**Interface contract:**
```python
def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    cropped_img : H×W×3  uint8 RGB
    landmarks   : (21, 2) float32, crop-relative [0,1]
    returns     : int in {0, 1, 2, 3, 4, 5}
    """
```

Evaluation runs in a **fresh Google Colab runtime** — no manual modifications allowed.

---

## Architecture — Dual-Stream Fusion

```
cropped image (64×64×3)          landmarks (21×2 = 42 floats)
        │                                    │
 ┌──────▼──────────────┐           ┌─────────▼──────┐
 │  Image Branch       │           │  Landmark MLP  │
 │  5-layer DW-sep CNN │           │  42→128→64     │
 │  → 64-dim features  │           │  BN + ReLU     │
 └──────────┬──────────┘           └─────────┬──────┘
            └──────────────┬─────────────────┘
                           │  concat → 128-dim
                      ┌────▼────┐
                      │ Fusion  │
                      │ 128→64→6│
                      │ dropout │
                      └────┬────┘
                           │ softmax
                    ┌──────▼──────┐
                    │ N/A heuristics │  ← post-processing
                    └─────────────┘
```

### Image Branch — Custom DW-sep CNN
- Input: 64×64 RGB, normalized to `[−1, 1]`
- 5 depthwise-separable conv blocks, each: `DWConv → PWConv → BN → ReLU`
- Channel progression: 3→16→32→64→64→64
- Global average pooling → 64-dim feature vector
- **Target size: ~0.3 MB FP32**

### Landmark Branch — MLP
- Input: 21 × (x, y) crop-relative coords → 42 floats
- `Linear(42→128) → BN → ReLU → Linear(128→64) → BN → ReLU`
- **Target size: ~0.1 MB FP32**

### Fusion Head
- `concat(64, 64) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→6)`
- **Target size: ~0.05 MB FP32**

### Total Size Budget
| Component | FP32 | INT8 (after quantization) |
|-----------|------|--------------------------|
| CNN backbone | ~0.3 MB | ~0.08 MB |
| Landmark MLP | ~0.1 MB | ~0.03 MB |
| Fusion head | ~0.05 MB | ~0.01 MB |
| **Total** | **~0.45 MB** | **~0.12 MB** |

---

## Dataset

**Source:** `testdummyvt/hagRIDv2_512px` (HuggingFace)  
**Preprocessed output:** `/work/xxjustin77xx/Multimedia_Final_Project/data/`

### Class distribution (after MediaPipe preprocessing)

| Split | N/A (0) | fist (1) | like (2) | ok (3) | one (4) | palm (5) | Total |
|-------|---------|---------|---------|--------|---------|---------|-------|
| train | 487,094 | 21,079 | 20,421 | 21,646 | 21,267 | 22,196 | 593,703 |
| val   | 58,407  | 2,669  | 2,625  | 2,816  | 2,661  | 2,804  | 71,982 |
| test  | 99,385  | 4,602  | 4,568  | 4,808  | 4,633  | 4,784  | 122,780 |

**N/A class includes:** call, dislike, four, mute, peace, rock, stop, three, two_up, no_gesture, etc. (all single-handed non-target gestures).  
**Excluded (two-handed):** grip, hand_heart, holy, take_picture, timeout, xsign.

> ⚠️ N/A is ~8× larger than any target class — must be handled with weighted loss.

### Augmentation (matching TA test conditions)
| Transform | Parameters |
|-----------|-----------|
| Bbox jitter | random crop expansion ±10% |
| Gaussian blur | kernel 3–7, σ 0.1–1.5 |
| Color jitter | brightness/contrast/saturation ±20% |
| Horizontal flip | flip image + mirror landmark x: `x' = 1 − x` |
| Rotation | ±15°, rotate landmark coords around crop center (0.5, 0.5) |

**Landmark rotation formula:**
```
shift:    x' = x − 0.5,  y' = y − 0.5
rotate:   x'' =  x'·cosθ − y'·sinθ
          y'' =  x'·sinθ + y'·cosθ
shift back: x_out = x'' + 0.5,  y_out = y'' + 0.5
clip to [0, 1]
```

---

## Implementation Steps

### Step 1 — Model Architecture ✅
- [x] `model/architecture.py` — DW-sep CNN + landmark MLP + fusion head
- [x] 36,039 params, **0.142 MB FP32** → size score **29.57 / 30 pts**

### Step 2 — Dataset & DataLoader ✅
- [x] `train/augment.py` — `RandomHorizontalFlipLM`, `RandomRotationLM`, `RandomBboxJitterLM`, `RandomGaussianBlur`
- [x] `train/dataset.py` — `GestureDataset` + `balanced_sampler()`

### Step 3 — Phase 1 Training (image branch only)
- [x] `train/train_phase1.py` — ready to run
- Train `Phase1Model` (CNN + temporary `Linear(64→6)` head, no landmarks)
- Loss: standard cross-entropy
- Balanced `WeightedRandomSampler` — each class equally likely per batch
- Optimizer: Adam lr=1e-3, CosineAnnealingLR over 20 epochs, batch=256
- Goal: warm-start CNN so it learns visual features before fusion
- Saves: `runs/<run-name>/phase1_best.pt`, `phase1_last.pt`, `phase1.log`, `phase1_args.json`
- [ ] **Job not yet submitted**

### Step 4 — Phase 2 Training (joint end-to-end)
- [x] `train/train_phase2.py` — ready to run
- Loads Phase 1 CNN weights into `GestureClassifier.image_branch`
- Trains full dual-stream model (CNN + landmark MLP + fusion head) end-to-end
- Loss: **weighted cross-entropy** — inverse-frequency weights + 1.5× N/A boost to reflect −2 false trigger cost
- Differential LR: image_branch=1e-4 (pre-trained), landmark_branch+fusion=5e-4 (new)
- CosineAnnealingLR over 30 epochs, batch=256, `drop_last=True`
- Saves best checkpoint by **validation contest score** (+1/−2 formula with default gates 0.5/0.2)
- Saves: `runs/<run-name>/phase2_best.pt`, `phase2_last.pt`, `phase2.log`, `phase2_args.json`
- [ ] **Job not yet submitted**

### Step 5 — Threshold Calibration
- [x] `train/calibrate.py` — ready to run
- Loads `runs/<run-name>/phase2_best.pt`, runs full val set inference (no augmentation)
- Grid search: `conf_thresh ∈ linspace(0.30, 0.95, 14)` × `margin_thresh ∈ linspace(0.0, 0.5, 11)` = **154 combinations**
- Scores each pair with exact contest formula: `+1 correct target, −2 false trigger, 0 for N/A prediction`
- Logs top-10 threshold combinations and saves winner to `model/thresholds.json`
- [ ] **Run after Phase 2 completes**

### Step 6 — N/A Heuristics ✅
- [x] Implemented in `inference.py`
  1. **Confidence gate:** `if max(p) < conf_thresh → N/A`
  2. **Top-2 margin:** `if p[1st] − p[2nd] < margin_thresh → N/A`
  3. **Landmark spread check:** if all 21 landmarks span < 5% of crop width/height → N/A (degenerate detection)

### Step 7 — Export
- [x] `train/export.py` — ready to run
- Copies `runs/<run-name>/phase2_best.pt` → `model/weights.pt`
- Prints size score preview: `(10 − size_MB) × 3`
- Runs smoke-test forward pass to confirm the file loads correctly
- [ ] **Run after calibration**

### Step 8 — Submission Files ✅
- [x] `inference.py` — `predict()` with N/A heuristics, model loaded once at import time
- [x] `requirements.txt` — torch, torchvision, numpy, Pillow (Colab-compatible)
- [x] `README.md` — setup and usage
- [ ] Package as `team_X.zip` after export

---

## File Structure (final)

```
Multimedia_Final_Project/
├── PROGRESS.md                  ← this file
├── proposal.md
├── hand_preprocess.py           ← TA-provided MediaPipe preprocessor
│
├── model/
│   ├── architecture.py          ← model definition (shared by train + inference)
│   ├── weights.pt               ← FP32 trained weights (copied by export.py)
│   └── thresholds.json          ← calibrated conf_thresh + margin_thresh
│
├── train/
│   ├── dataset.py               ← dataloader + augmentation
│   ├── augment.py               ← landmark-aware transforms
│   ├── train_phase1.py          ← image branch pretraining
│   ├── train_phase2.py          ← joint end-to-end training
│   ├── calibrate.py             ← threshold grid search
│   └── export.py                ← size check + copy to model/
│
├── runs/                        ← one subfolder per experiment
│   └── run_MMDD_HHMM/
│       ├── phase1_args.json     ← saved hyperparameters
│       ├── phase1_best.pt       ← best phase-1 checkpoint
│       ├── phase1_last.pt
│       ├── phase1.log
│       ├── phase2_args.json
│       ├── phase2_best.pt       ← best phase-2 checkpoint (→ model/weights.pt)
│       ├── phase2_last.pt
│       └── phase2.log
│
├── inference.py                 ← submission entry point
├── requirements.txt             ← submission dependencies
├── README.md                    ← submission readme
│
├── data/                        ← preprocessed HaGRIDv2
│   ├── crops/<split>/<label>/
│   └── landmarks/<split>/<label>/
│
└── download_and_preprocess.py   ← dataset pipeline (already done ✅)
```

---

## Training Jobs (SLURM)

All jobs submitted from `/work/xxjustin77xx/` via `srun_args.sh`.  
SLURM logs: `/work/xxjustin77xx/results/job_log/job-<id>.{out,err}`  
Run logs: `Multimedia_Final_Project/runs/<run-name>/phase{1,2}.log`  
Conda env: `multimedia`

### Commands

```bash
# Phase 1 — pick a run name once and reuse it for all steps
RUN=run_0527_1800   # example — change to your actual timestamp

# Step 3: Phase 1 training (20 epochs, ~1 GPU)
bash srun_args.sh 1 "python /work/xxjustin77xx/Multimedia_Final_Project/train/train_phase1.py --run-name $RUN"

# Step 4: Phase 2 training (30 epochs, ~1 GPU) — run after Phase 1 finishes
bash srun_args.sh 1 "python /work/xxjustin77xx/Multimedia_Final_Project/train/train_phase2.py --run-name $RUN"

# Step 5: Threshold calibration (CPU-fast, ~1 min) — run after Phase 2
bash srun_args.sh 1 "python /work/xxjustin77xx/Multimedia_Final_Project/train/calibrate.py --run-name $RUN"

# Step 7: Export weights to model/
bash srun_args.sh 1 "python /work/xxjustin77xx/Multimedia_Final_Project/train/export.py --run-name $RUN"
```

### Job Status

| Step | Job ID | Run name | Status |
|------|--------|----------|--------|
| Dataset download + preprocess | 217075 | — | ✅ Done |
| Phase 1 training | — | — | ⬜ Pending |
| Phase 2 training | — | — | ⬜ Pending |
| Threshold calibration | — | — | ⬜ Pending |
| Export | — | — | ⬜ Pending |

---

## Known Issues / Bugs
<!-- Add entries here as discovered -->
- None yet

---

## Results Log
<!-- Add validation scores and submission results here -->

| Date | Checkpoint | Val contest score | Leaderboard score | Notes |
|------|-----------|-------------------|-------------------|-------|
| — | — | — | — | — |
