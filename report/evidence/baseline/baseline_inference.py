"""
Hand Gesture Classifier — inference interface.
Implements: predict(cropped_img, landmarks) -> int

Classes: 0=N/A, 1=fist, 2=like, 3=ok, 4=one, 5=palm
All paths are relative to this file.
"""

import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

# ── model definition (must match training exactly) ────────────────────────────

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class ImageBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBnRelu(3,   32,  stride=2),   # 64×64
            ConvBnRelu(32,  64,  stride=2),   # 32×32
            ConvBnRelu(64,  128, stride=2),   # 16×16
            ConvBnRelu(128, 128, stride=2),   #  8×8
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, 128)

    def forward(self, x):
        return F.relu(self.proj(self.net(x).flatten(1)))


class LandmarkBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        return self.net(x)


class GestureNet(nn.Module):
    def __init__(self, num_classes=6, dropout=0.4):
        super().__init__()
        self.image_branch    = ImageBranch()
        self.landmark_branch = LandmarkBranch()
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, img, lm):
        return self.classifier(
            torch.cat([self.image_branch(img), self.landmark_branch(lm)], dim=1)
        )


# ── load model once at import time ────────────────────────────────────────────

_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_WEIGHTS = Path(__file__).parent / "model" / "gesture_net.pth"

_model = GestureNet().to(_DEVICE)
_model.load_state_dict(torch.load(_WEIGHTS, map_location=_DEVICE, weights_only=True))
_model.eval()

_transform = T.Compose([
    T.Resize((128, 128)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

# Softmax confidence below this → predict N/A (class 0).
# Keeps false-trigger rate low at the cost of a few missed positives.
_CONFIDENCE_THRESHOLD = 0.60


# ── public interface ──────────────────────────────────────────────────────────

def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    Args:
        cropped_img : H×W×3 uint8 RGB numpy array  (hand crop from MediaPipe)
        landmarks   : (21, 2) float32 numpy array  (crop-relative, normalised [0,1])
    Returns:
        int in {0, 1, 2, 3, 4, 5}
        0=N/A, 1=fist, 2=like, 3=ok, 4=one, 5=palm
    """
    img = _transform(Image.fromarray(cropped_img)).unsqueeze(0).to(_DEVICE)
    lm  = torch.from_numpy(landmarks.flatten().astype(np.float32)).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        probs = F.softmax(_model(img, lm), dim=1)[0]   # (6,)

    confidence, cls = probs.max(0)
    if confidence.item() < _CONFIDENCE_THRESHOLD:
        return 0   # not confident enough → N/A

    return int(cls.item())
