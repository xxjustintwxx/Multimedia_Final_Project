# Slide Deck: Hand Gesture Classification on Edge Devices
> Instructions for Gemini: Generate one slide per `---` section. Use the **Title** as the slide heading. Use bullet points, tables, and diagrams as specified. Keep text concise — prefer visuals over paragraphs.

---

## Slide 1 — Cover

**Title:** Hand Gesture Classification on Edge Devices

**Subtitle:** Microsoft Challenge — Multimedia Final Project (114-2)

**Content:**
- Compact gesture classifier for real-world edge deployment
- Team 14

**Visual suggestion:** Full-bleed background of a hand performing gestures, dark overlay, white text.

---

## Slide 2 — 報告流程 (Agenda)

**Title:** 報告流程

**Content — numbered list:**
1. 問題定義 & 評分規則 (Problem & Scoring)
2. 模型架構 (Model Architecture)
3. 訓練流程 (Training Pipeline)
   - Phase 1 — Image Branch Pretraining
   - Phase 2 — Joint End-to-End Training
   - Calibration — Threshold Tuning
4. 數據結果 (Dataset & Results)
5. Live Demo

**Visual suggestion:** Clean timeline or numbered icon strip across the slide.

---

## Slide 3 — 問題定義 (Problem Definition)

**Title:** 問題定義

**Content:**

Input (two streams):
- **Cropped hand image** — RGB, variable size (MediaPipe pre-cropped)
- **21 landmark coordinates** — (x, y) crop-relative [0, 1], from MediaPipe Hand Landmarker

Output: one integer in {0, 1, 2, 3, 4, 5}

| Class | Gesture |
|-------|---------|
| 0 | N/A (reject / unknown) |
| 1 | fist |
| 2 | like |
| 3 | ok |
| 4 | one |
| 5 | palm |

Key constraint: **Full-frame images are strictly prohibited.**

**Visual suggestion:** Diagram showing MediaPipe preprocessing → cropped image + 21 dots → classifier → class label.

---

## Slide 4 — 評分規則 (Scoring Rules)

**Title:** 評分規則 — Why N/A Rejection Matters Most

**Content:**

| Criterion | Points | Formula |
|-----------|--------|---------|
| Model Size ≤ 10 MB | 30 | `(10 − size_MB) × 3` |
| Basic Performance (HaGRIDv2 test) | 20 | +1 correct, **−2 false trigger** |
| Real-World Robustness (TA-shot dataset) | 40 | 50 N/A + 50 target images; same scoring |
| Presentation | 30 | Live demo + explanation |

**Key insight (highlight in red box):**
> False trigger = **−2 pts** &nbsp;|&nbsp; Correct prediction = **+1 pt**
> One wrong trigger cancels two correct predictions.
> → Conservative N/A rejection is the highest-leverage optimization.

**Our model size:** 0.142 MB → Size score = **(10 − 0.142) × 3 = 29.57 / 30 pts**

---

## Slide 5 — 模型架構 Overview (Model Architecture Overview)

**Title:** 模型架構 — Dual-Stream Fusion

**Content — architecture diagram (text art, Gemini should render as a flow diagram):**

```
 Cropped Image (64×64×3)        Landmarks (21×2 = 42 floats)
         │                                  │
 ┌───────▼──────────────┐         ┌─────────▼──────────┐
 │   Image Branch       │         │  Landmark Branch   │
 │   5-layer DW-sep CNN │         │  MLP: 42→128→64    │
 │   3→16→32→64→64→64  │         │  BN + ReLU ×2      │
 │   → 64-dim feature   │         │  → 64-dim feature  │
 └──────────┬───────────┘         └──────────┬─────────┘
            └────────────┬────────────────────┘
                         │  concat → 128-dim
                    ┌────▼─────────────┐
                    │  Fusion Head     │
                    │  128→64→6        │
                    │  ReLU + Dropout  │
                    └────┬─────────────┘
                         │  softmax
                  ┌──────▼───────────┐
                  │  N/A Heuristics  │  ← post-processing layer
                  └──────────────────┘
```

**Stats box:**
- Total parameters: **36,039**
- Model size (FP32): **0.142 MB**
- Size score: **29.57 / 30 pts**

---

## Slide 6 — Image Branch (CNN Detail)

**Title:** Image Branch — Depthwise-Separable CNN

**Content:**

Each block = **DWConv → BN → ReLU → PWConv → BN → ReLU**
(MobileNet-style depthwise-separable: fewer params, same receptive field)

| Block | Input Size | Output Size | Channels | Stride |
|-------|-----------|------------|----------|--------|
| Block 1 | 64×64 | 32×32 | 3 → 16 | 2 |
| Block 2 | 32×32 | 16×16 | 16 → 32 | 2 |
| Block 3 | 16×16 | 8×8 | 32 → 64 | 2 |
| Block 4 | 8×8 | 8×8 | 64 → 64 | 1 |
| Block 5 | 8×8 | 8×8 | 64 → 64 | 1 |
| GAP | 8×8 | 1×1 | 64 | — |

Output: **64-dim feature vector** capturing visual texture and shape.

**Why DW-sep?** ~8–9× fewer multiply-adds than standard conv — critical for staying under 10 MB.

---

## Slide 7 — Landmark Branch & Fusion Head

**Title:** Landmark Branch + Fusion Head

**Content:**

**Landmark Branch (MLP):**
- Input: 21 keypoints × (x, y) = **42 floats**, crop-relative coordinates [0, 1]
- Architecture: `Linear(42→128) → BN → ReLU → Linear(128→64) → BN → ReLU`
- Encodes **structural geometry** independent of lighting, color, and texture

**Why two streams?**
- Image branch: captures texture (finger skin creases, hand outline)
- Landmark branch: captures structure (finger angles, spread, curl)
- Together they are **complementary** — neither alone achieves the same accuracy

**Fusion Head:**
- `concat(64+64=128) → Linear(128→64) → ReLU → Dropout(0.3) → Linear(64→6)`
- Dropout prevents co-adaptation between the two streams

---

## Slide 8 — N/A Rejection Heuristics

**Title:** N/A Rejection — 3-Layer Defense

**Content:**

Because false triggers cost **−2 pts**, we apply three post-processing guards on top of the neural network output:

**Layer 1 — Confidence Gate**
```
if max(softmax_probs) < conf_thresh  →  output N/A
```
Rejects predictions where the network is uncertain.

**Layer 2 — Top-2 Margin Check**
```
if prob[1st class] − prob[2nd class] < margin_thresh  →  output N/A
```
Rejects predictions where two classes are nearly tied (ambiguous gesture).

**Layer 3 — Landmark Spread Check**
```
if all 21 landmarks span < 5% of crop width AND height  →  output N/A
```
Rejects degenerate detections (e.g., only fingertips detected, or bad MediaPipe output).

**Note:** Both `conf_thresh` and `margin_thresh` are not hardcoded — they are tuned via grid search on the validation set (see Calibration slide).

---

## Slide 9 — 訓練流程 Overview (Training Pipeline Overview)

**Title:** 訓練流程 — Three-Stage Pipeline

**Content — horizontal flow diagram:**

```
  [HaGRIDv2 Dataset]
         │
         ▼
  ┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
  │  Phase 1        │ ──▶ │  Phase 2             │ ──▶ │  Calibration     │
  │  Image branch   │     │  Full dual-stream    │     │  Threshold grid  │
  │  pretraining    │     │  joint training      │     │  search on val   │
  │  (CNN only)     │     │  (CNN + MLP + fusion)│     │  set             │
  │  20 epochs      │     │  30 epochs           │     │  154 combos      │
  │  val acc 86.15% │     │  val acc 96.75%      │     │  → thresholds.json│
  └─────────────────┘     └──────────────────────┘     └──────────────────┘
```

**Why three stages?**
- Phase 1 gives the CNN a head start on visual features before landmark fusion is introduced
- Phase 2 jointly optimizes both streams with scoring-aware loss
- Calibration sets conservative thresholds to minimize false triggers

---

## Slide 10 — Phase 1 Training (Image Branch Pretraining)

**Title:** Phase 1 — Image Branch Pretraining

**Content:**

**Goal:** Warm-start the CNN so it learns visual hand features before being fused with landmarks.

**Setup:**
- Model: `Phase1Model` = ImageBranch + temporary `Linear(64→6)` head (landmarks ignored)
- Loss: standard cross-entropy
- Optimizer: Adam, lr = 1e-3
- Scheduler: CosineAnnealingLR over 20 epochs
- Batch size: 256
- Sampler: **WeightedRandomSampler** — each class equally likely per batch (balances the 8× N/A imbalance)

**Augmentation applied:**
- Bbox jitter ±10%, Gaussian blur, color jitter ±20%, horizontal flip, rotation ±15°

**Result:**
| Metric | Value |
|--------|-------|
| Epochs | 20 |
| Best validation accuracy | **86.15%** (epoch 20) |

After Phase 1, the CNN weights are frozen-copied into the full `GestureClassifier` for Phase 2.

---

## Slide 11 — Phase 2 Training (Joint End-to-End)

**Title:** Phase 2 — Joint End-to-End Training

**Content:**

**Goal:** Train the full dual-stream model, fine-tuning CNN with lower LR to preserve learned features.

**Setup:**
- Model: full `GestureClassifier` (CNN + Landmark MLP + Fusion Head)
- CNN weights initialized from Phase 1 best checkpoint
- **Differential learning rates:**
  - Image branch (pre-trained): lr = **1e-4**
  - Landmark branch + fusion head (new): lr = **5e-4**
- Scheduler: CosineAnnealingLR over 30 epochs
- Batch size: 256, `drop_last=True`

**Loss: Weighted Cross-Entropy**
- Weights = inverse class frequency
- N/A class weight × **1.5 boost** → reflects the asymmetric −2 false trigger penalty
- Encourages the model to be conservative when uncertain

**Best checkpoint selected by:** validation contest score (+1/−2 formula), not just accuracy.

**Result:**
| Metric | Value |
|--------|-------|
| Epochs | 30 |
| Best validation accuracy | **96.75%** |
| Best validation contest score | **+11,539** (epoch 28) |

---

## Slide 12 — Calibration (Threshold Tuning)

**Title:** Calibration — Grid Search for Optimal Thresholds

**Content:**

**Goal:** Find the (conf_thresh, margin_thresh) pair that maximizes contest score on validation set.

**Method:**
- Run full validation set inference (no augmentation) → collect softmax probabilities
- Grid search over **14 × 11 = 154 combinations:**
  - `conf_thresh` ∈ linspace(0.30, 0.95, 14)
  - `margin_thresh` ∈ linspace(0.00, 0.50, 11)
- Score each pair with exact contest formula: +1 correct, −2 false trigger, 0 for N/A prediction

**Scoring function:**
```python
for each sample:
    if pred_class == 0:             skip (N/A, no points)
    if max_prob < conf_thresh:      skip (treated as N/A)
    if margin < margin_thresh:      skip (treated as N/A)
    if pred == true_label:          score += 1
    else:                           score -= 2
```

**Output:** Best thresholds saved to `model/thresholds.json`, loaded at inference time.

**Visual suggestion:** 2D heatmap with conf_thresh on x-axis, margin_thresh on y-axis, contest score as color intensity.

---

## Slide 13 — Dataset

**Title:** 數據集 — HaGRIDv2 512px

**Content:**

**Source:** HaGRIDv2 (HuggingFace: `testdummyvt/hagRIDv2_512px`), preprocessed with MediaPipe

**Class distribution:**

| Split | N/A (0) | fist (1) | like (2) | ok (3) | one (4) | palm (5) | Total |
|-------|---------|---------|---------|--------|---------|---------|-------|
| Train | 487,094 | 21,079 | 20,421 | 21,646 | 21,267 | 22,196 | **593,703** |
| Val | 58,407 | 2,669 | 2,625 | 2,816 | 2,661 | 2,804 | **71,982** |
| Test | 99,385 | 4,602 | 4,568 | 4,808 | 4,633 | 4,784 | **122,780** |

**N/A class includes:** call, dislike, four, mute, peace, rock, stop, three, two_up, no_gesture (all single-handed non-target gestures)
**Excluded (two-handed):** grip, hand_heart, holy, take_picture, timeout, xsign

**⚠ N/A is ~8× larger than any target class** → handled via WeightedRandomSampler (Phase 1) and weighted loss (Phase 2).

---

## Slide 14 — Data Augmentation

**Title:** Data Augmentation — Matching TA Test Conditions

**Content:**

TAs apply bbox jitter and blur at test time — we replicate these in training:

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| Bbox jitter | Random crop expansion ±10% | Simulate TA bbox perturbation |
| Gaussian blur | Kernel 3–7, σ 0.1–1.5 | Simulate TA blur augmentation |
| Color jitter | Brightness/contrast/saturation ±20% | Lighting variation |
| Horizontal flip | Flip image + mirror landmark x: `x' = 1 − x` | Handedness invariance |
| Rotation | ±15°, rotate landmarks around crop center (0.5, 0.5) | Tilt robustness |

**Landmark rotation formula (applied alongside image rotation):**
```
x' = (x − 0.5)·cosθ − (y − 0.5)·sinθ + 0.5
y' = (x − 0.5)·sinθ + (y − 0.5)·cosθ + 0.5
```
Landmark coordinates are rotated in sync with the image — they stay geometrically consistent.

---

## Slide 15 — 數據結果 (Results)

**Title:** 數據結果 — Summary

**Content:**

**Training results:**

| Stage | Val Accuracy | Val Contest Score |
|-------|-------------|------------------|
| Phase 1 (CNN only, no landmarks) | 86.15% | — |
| Phase 2 (dual-stream, epoch 28) | **96.75%** | **+11,539** |

**Model size:**

| Metric | Value |
|--------|-------|
| Parameters | 36,039 |
| FP32 model size | **0.142 MB** |
| Size score | **(10 − 0.142) × 3 = 29.57 / 30 pts** |

**Inference pipeline:**
1. Resize crop to 64×64, normalize to [−1, 1]
2. Flatten landmarks to 42 floats
3. Forward pass → softmax probs
4. Apply 3-layer N/A heuristics → final class

**Runs in fresh Google Colab without any manual modifications.**

---

## Slide 16 — Live Demo

**Title:** Live Demo

**Content:**

**Demo setup:**
- Webcam feed → MediaPipe (TA preprocessor) → cropped image + 21 landmarks
- → our `predict()` function → displayed class label in real time

**What to show:**
1. Five target gestures recognized correctly: fist, like, ok, one, palm
2. Non-target gestures correctly rejected as N/A (e.g., peace sign, random hand pose)
3. Ambiguous / distorted poses → N/A (confidence gate kicks in)
4. Model file size shown: `du -sh model/weights.pt` → 0.142 MB

**Engineering strategies to highlight:**
- Dual-stream fusion: why two modalities beat one
- Asymmetric loss weighting: why we boost N/A in training
- Three-layer N/A defense: confidence gate → margin check → landmark spread
- Calibration via grid search on contest scoring formula (not accuracy)

**Visual suggestion:** Split screen — left: live webcam with landmark overlay, right: softmax probability bar chart updating in real time, top class label displayed large.
