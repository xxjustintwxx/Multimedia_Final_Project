"""Dataset for preprocessed HaGRIDv2 crops and landmarks."""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image
import torchvision.transforms as T

_TRAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_TRAIN_DIR))
from augment import (
    ComposeWithLandmarks, RandomHorizontalFlipLM,
    RandomRotationLM, RandomBboxJitterLM, RandomGaussianBlur,
)

DATA_ROOT = Path("/work/xxjustin77xx/Multimedia_Final_Project/data")
IMG_SIZE  = 64
MEAN = (0.5, 0.5, 0.5)
STD  = (0.5, 0.5, 0.5)
NUM_CLASSES = 6


class GestureDataset(Dataset):
    def __init__(self, split: str, augment: bool = False):
        assert split in ("train", "val", "test")
        self.augment = augment
        self.samples: list[tuple[str, str, int]] = []  # (crop_path, lm_path, label)

        crops_root = DATA_ROOT / "crops"    / split
        lm_root    = DATA_ROOT / "landmarks" / split

        for label in range(NUM_CLASSES):
            crop_dir = crops_root / str(label)
            lm_dir   = lm_root    / str(label)
            if not crop_dir.exists():
                continue
            for entry in os.scandir(str(crop_dir)):
                if entry.name.endswith(".jpg"):
                    stem    = entry.name[:-4]
                    lm_path = str(lm_dir / f"{stem}.npy")
                    self.samples.append((entry.path, lm_path, label))

        self.labels = [s[2] for s in self.samples]

        # Landmark-aware spatial transforms (applied together to img + lm)
        self.lm_aug = ComposeWithLandmarks([
            RandomBboxJitterLM(jitter=0.10),
            RandomHorizontalFlipLM(p=0.5),
            RandomRotationLM(max_degrees=15),
        ])

        # Image-only colour transforms
        self.img_colour_aug = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ])
        self.blur_aug = RandomGaussianBlur(sigma_range=(0.1, 1.5))

        # Final normalisation (applied always)
        self.to_tensor = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD),
        ])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        crop_path, lm_path, label = self.samples[idx]
        img = Image.open(crop_path).convert("RGB")
        lm  = np.load(lm_path).astype(np.float32)   # (21, 2)

        if self.augment:
            img, lm = self.lm_aug(img, lm)
            img = self.blur_aug(img)
            img = self.img_colour_aug(img)

        img_tensor = self.to_tensor(img)         # (3, 64, 64)
        lm_tensor  = torch.from_numpy(lm.flatten())  # (42,)
        return img_tensor, lm_tensor, label

    # ------------------------------------------------------------------
    def balanced_sampler(self) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler that gives each class equal weight."""
        counts = np.bincount(self.labels, minlength=NUM_CLASSES).astype(np.float64)
        counts = np.where(counts == 0, 1, counts)          # avoid div-by-zero
        class_w = 1.0 / counts
        sample_w = [class_w[l] for l in self.labels]
        return WeightedRandomSampler(
            weights=sample_w,
            num_samples=len(sample_w),
            replacement=True,
        )
