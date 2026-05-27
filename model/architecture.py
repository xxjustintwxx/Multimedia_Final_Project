"""Dual-stream gesture classifier: DW-sep CNN + landmark MLP + fusion head."""

import torch
import torch.nn as nn


class DWSepBlock(nn.Module):
    """Depthwise-separable conv block (standard MobileNet style)."""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw  = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw  = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.dw(x)))
        x = self.relu(self.bn2(self.pw(x)))
        return x


class ImageBranch(nn.Module):
    """5-layer DW-sep CNN: (B, 3, 64, 64) → (B, 64)."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            DWSepBlock(3,  16, stride=2),   # 64×64 → 32×32
            DWSepBlock(16, 32, stride=2),   # 32×32 → 16×16
            DWSepBlock(32, 64, stride=2),   # 16×16 →  8×8
            DWSepBlock(64, 64, stride=1),   #  8×8  →  8×8
            DWSepBlock(64, 64, stride=1),   #  8×8  →  8×8
        )
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gap(self.blocks(x)).flatten(1)  # (B, 64)


class LandmarkBranch(nn.Module):
    """MLP: (B, 42) → (B, 64)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(42, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 64)


class GestureClassifier(nn.Module):
    """Full dual-stream model (phase-2 training + inference)."""
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.image_branch    = ImageBranch()
        self.landmark_branch = LandmarkBranch()
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 6),
        )

    def forward(self, img: torch.Tensor, lm: torch.Tensor) -> torch.Tensor:
        img_feat = self.image_branch(img)       # (B, 64)
        lm_feat  = self.landmark_branch(lm)     # (B, 64)
        return self.fusion(torch.cat([img_feat, lm_feat], dim=1))  # (B, 6)


class Phase1Model(nn.Module):
    """Image-branch-only model for phase-1 pretraining."""
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.image_branch = ImageBranch()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(64, 6),
        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        return self.head(self.image_branch(img))


def model_size_mb(model: nn.Module) -> float:
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total / (1024 ** 2)
