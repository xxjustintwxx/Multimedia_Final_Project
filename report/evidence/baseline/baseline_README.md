# Hand Gesture Classifier

Dual-path model: small CNN (image crop) + MLP (21 landmarks) → 6-class classifier.

## Classes
| Index | Gesture |
|-------|---------|
| 0     | N/A (all non-target gestures) |
| 1     | fist |
| 2     | like |
| 3     | ok |
| 4     | one |
| 5     | palm |

## Environment
```bash
pip install -r requirements.txt
```
Tested on Python 3.10, PyTorch 2.x. CPU and CUDA both work.

## Usage
```python
from inference import predict
import numpy as np

# cropped_img : H×W×3 uint8 RGB numpy array
# landmarks   : (21, 2) float32 numpy array, crop-relative [0, 1]
result = predict(cropped_img, landmarks)
print(result)  # int in {0,1,2,3,4,5}
```

## Model
- Architecture: ImageBranch (4-layer CNN → 128d) + LandmarkBranch (MLP → 32d) → fused → 6 logits
- Weights: `model/gesture_net.pth` (~1.2 MB)
- N/A rejection: confidence threshold at 0.60 (tunable in `inference.py`)
