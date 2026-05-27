# Hand Gesture Classification on Edge Devices

Multimedia (114-2) — Microsoft Challenge  
6-class gesture classifier: N/A · fist · like · ok · one · palm

## Environment

```bash
pip install -r requirements.txt
```

Tested on Python 3.11, PyTorch 2.0+. Runs on CPU (Colab) or CUDA.

## Inference interface

```python
import numpy as np
from inference import predict

# cropped_img : np.ndarray  H×W×3  uint8 RGB   (MediaPipe crop)
# landmarks   : np.ndarray  (21, 2) float32    (crop-relative [0,1])
label = predict(cropped_img, landmarks)
# returns int in {0=N/A, 1=fist, 2=like, 3=ok, 4=one, 5=palm}
```

## File layout

```
inference.py          ← entry point, implements predict()
model/
  architecture.py     ← model definition
  weights.pt          ← trained FP32 weights (~0.14 MB)
  thresholds.json     ← calibrated N/A rejection thresholds
requirements.txt
README.md
```

## Model

Dual-stream fusion network:
- **Image branch**: 5-layer depthwise-separable CNN, 64×64 RGB → 64-dim
- **Landmark branch**: MLP 42→128→64
- **Fusion head**: 128→64→6, softmax

Post-softmax N/A rejection:
1. Confidence gate: `max_prob < conf_thresh` → N/A
2. Top-2 margin: `p1 − p2 < margin_thresh` → N/A
3. Landmark sanity: collapsed detection → N/A

Model size: ~0.14 MB FP32 → size score ≈ 29.6 / 30 pts
