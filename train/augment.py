"""Landmark-aware augmentation transforms."""

import math
import random
from PIL import Image, ImageFilter
import numpy as np
import torchvision.transforms.functional as TF


class RandomHorizontalFlipLM:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img: Image.Image, lm: np.ndarray):
        if random.random() < self.p:
            img = TF.hflip(img)
            lm = lm.copy()
            lm[:, 0] = 1.0 - lm[:, 0]
        return img, lm


class RandomRotationLM:
    """Rotate image and landmarks around the crop center (0.5, 0.5)."""
    def __init__(self, max_degrees: float = 15.0):
        self.max_degrees = max_degrees

    def __call__(self, img: Image.Image, lm: np.ndarray):
        angle = random.uniform(-self.max_degrees, self.max_degrees)
        img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)

        theta = math.radians(angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        lm = lm.copy()
        x = lm[:, 0] - 0.5
        y = lm[:, 1] - 0.5
        lm[:, 0] = x * cos_t - y * sin_t + 0.5
        lm[:, 1] = x * sin_t + y * cos_t + 0.5
        np.clip(lm, 0.0, 1.0, out=lm)
        return img, lm


class RandomBboxJitterLM:
    """Simulate MediaPipe bbox jitter: randomly zoom in or out by ±jitter."""
    def __init__(self, jitter: float = 0.10):
        self.jitter = jitter

    def __call__(self, img: Image.Image, lm: np.ndarray):
        w, h = img.size
        delta = random.uniform(-self.jitter, self.jitter)
        pad = int(abs(delta) * min(w, h))
        if pad == 0:
            return img, lm

        lm = lm.copy()

        if delta > 0:
            # Zoom out: pad image edges, scale landmarks inward
            padded = Image.new("RGB", (w + 2 * pad, h + 2 * pad), (0, 0, 0))
            padded.paste(img, (pad, pad))
            img = padded.resize((w, h), Image.BILINEAR)
            scale = w / (w + 2 * pad)
            offset = pad / (w + 2 * pad)
            lm[:, 0] = lm[:, 0] * scale + offset
            lm[:, 1] = lm[:, 1] * scale + offset
        else:
            # Zoom in: crop a random inner region
            left = random.randint(0, pad)
            top  = random.randint(0, pad)
            right  = w - (pad - left)
            bottom = h - (pad - top)
            img = img.crop((left, top, right, bottom)).resize((w, h), Image.BILINEAR)
            crop_w = right - left
            crop_h = bottom - top
            lm[:, 0] = (lm[:, 0] * w - left) / crop_w
            lm[:, 1] = (lm[:, 1] * h - top)  / crop_h
            np.clip(lm, 0.0, 1.0, out=lm)

        return img, lm


class RandomGaussianBlur:
    """PIL-based Gaussian blur with random sigma."""
    def __init__(self, sigma_range: tuple = (0.1, 1.5)):
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        sigma = random.uniform(*self.sigma_range)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))


class ComposeWithLandmarks:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, img: Image.Image, lm: np.ndarray):
        for t in self.transforms:
            img, lm = t(img, lm)
        return img, lm
