"""Image -> normalized tensor, with augmentation for training."""
from __future__ import annotations

import random

import cv2
import numpy as np
import torch

PIXEL_MEAN = np.array([127.5, 127.5, 127.5], dtype=np.float32)
PIXEL_STD = np.array([128.0, 128.0, 128.0], dtype=np.float32)


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """HWC uint8 RGB -> CHW float32 tensor in [-1, 1], matching the ONNX contract."""
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(img))


class _EvalTransform:
    def __init__(self, size: tuple[int, int]):
        self.size = size

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        if img.shape[:2] != self.size:
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)
        return _to_tensor(img)


class _TrainTransform:
    def __init__(self, size: tuple[int, int], level: str = "medium"):
        self.size = size
        self.level = level

    def __call__(self, img: np.ndarray) -> torch.Tensor:
        if img.shape[:2] != self.size:
            img = cv2.resize(img, self.size[::-1], interpolation=cv2.INTER_LINEAR)

        if self.level == "none":
            return _to_tensor(img)

        if random.random() < 0.5:  # horizontal flip
            img = np.ascontiguousarray(img[:, ::-1, :])

        if self.level in ("light", "medium", "heavy"):
            if random.random() < 0.3:
                alpha = random.uniform(0.85, 1.15)  # brightness/contrast
                beta = random.uniform(-15, 15)
                img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        if self.level in ("medium", "heavy"):
            if random.random() < 0.2:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)
            if random.random() < 0.15:
                noise = np.random.normal(0, 8, img.shape).astype(np.float32)
                img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        if self.level == "heavy":
            if random.random() < 0.2:
                h, w = img.shape[:2]
                ch, cw = int(h * 0.25), int(w * 0.25)
                y0 = random.randint(0, h - ch)
                x0 = random.randint(0, w - cw)
                img = img.copy()
                img[y0:y0 + ch, x0:x0 + cw] = 0  # random erasing

        return _to_tensor(img)


def build_eval_transform(size: tuple[int, int]):
    return _EvalTransform(size)


def build_train_transform(size: tuple[int, int], level: str = "medium"):
    return _TrainTransform(size, level)


def preprocess_batch(images: list[np.ndarray], size: tuple[int, int]) -> torch.Tensor:
    """List of HWC uint8 RGB crops -> one NCHW float32 tensor in [-1, 1]."""
    tfm = _EvalTransform(size)
    tensors = [tfm(img) for img in images]
    return torch.stack(tensors, dim=0)