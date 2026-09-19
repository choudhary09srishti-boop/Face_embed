"""WIDER FACE reading + detection augmentation.

Two annotation formats are accepted:

1. RetinaFace-style `label.txt` (has 5 landmarks - preferred, because the
   detector's landmark head is what makes alignment possible):

       # 0--Parade/0_Parade_marchingband_1_849.jpg
       449 330 122 149 488.9 373.6 0.0 542.0 376.4 0.0 515.0 412.8 0.0 485.1 425.8 0.0 538.3 431.4 0.0 0.82

   i.e. `x y w h` then 5 x (lx, ly, unused), then a confidence. Missing
   landmarks are written as -1.

2. Original WIDER `wider_face_train_bbx_gt.txt` (boxes only):

       0--Parade/0_Parade_marchingband_1_849.jpg
       1
       449 330 122 149 0 0 0 0 0 0

   Faces here have no landmark targets; those samples train the box and class
   heads only and are skipped by the landmark loss.

Augmentation is the SSD recipe adapted to faces: random square crop, photometric
distortion, pad-to-square, horizontal mirror (with landmark pair swapping), then
resize to a fixed square input. Boxes and landmarks are carried through every
step and returned in normalised [0,1] coordinates.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..data.dataset import read_image_rgb
from ..data.transforms import PIXEL_MEAN, PIXEL_STD

log = logging.getLogger(__name__)

# Landmark index pairs swapped by a horizontal flip: (left eye, right eye) and
# (left mouth corner, right mouth corner). The nose (index 2) stays put.
FLIP_PAIRS = [(0, 1), (3, 4)]


def parse_annotations(label_file: str | Path, images_root: str | Path) -> list[dict]:
    """-> [{"path": abs path, "boxes": (M,4) xyxy, "landmarks": (M,10), "labels": (M,)}]."""
    label_file = Path(label_file)
    images_root = Path(images_root)
    text = label_file.read_text(encoding="utf-8").splitlines()
    retina_style = any(line.startswith("#") for line in text[:50])
    records: list[dict] = []

    if retina_style:
        cur: dict | None = None
        for line in text:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if cur is not None:
                    records.append(cur)
                rel = line[1:].strip()
                cur = {"path": str(images_root / rel), "rows": []}
            elif cur is not None:
                cur["rows"].append([float(v) for v in line.split()])
        if cur is not None:
            records.append(cur)
    else:
        i = 0
        while i < len(text):
            rel = text[i].strip()
            i += 1
            if not rel:
                continue
            if i >= len(text):
                break
            try:
                count = int(text[i].strip())
            except ValueError:
                continue
            i += 1
            rows = []
            # A count of 0 is still followed by one dummy line in WIDER's file.
            for _ in range(max(count, 1)):
                if i < len(text):
                    rows.append([float(v) for v in text[i].split()])
                    i += 1
            records.append({"path": str(images_root / rel), "rows": rows if count > 0 else []})

    out: list[dict] = []
    for rec in records:
        boxes, lmks, labels = [], [], []
        for row in rec["rows"]:
            if len(row) < 4:
                continue
            x, y, w, h = row[0], row[1], row[2], row[3]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            if len(row) >= 19:  # retinaface layout: 4 + 15 (+1 conf)
                pts = [row[4], row[5], row[7], row[8], row[10], row[11],
                       row[13], row[14], row[16], row[17]]
                lmks.append(pts if pts[0] >= 0 else [-1.0] * 10)
            else:
                lmks.append([-1.0] * 10)
            labels.append(1.0)
        if not boxes:
            continue
        out.append({
            "path": rec["path"],
            "boxes": np.asarray(boxes, dtype=np.float32),
            "landmarks": np.asarray(lmks, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.float32),
        })
    log.info("parsed %d annotated images from %s", len(out), label_file)
    return out


# ----------------------------------------------------------------- augmentation
def _distort(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    if random.random() < 0.5:
        img *= random.uniform(0.7, 1.3)                      # brightness
    if random.random() < 0.5:
        m = img.mean()
        img = (img - m) * random.uniform(0.7, 1.3) + m        # contrast
    img = np.clip(img, 0, 255).astype(np.uint8)
    if random.random() < 0.5:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[..., 0] = (hsv[..., 0] + random.uniform(-9, 9)) % 180
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img


def _random_square_crop(img, boxes, lmks, labels, max_tries: int = 50):
    """Crop a random square that keeps at least one face centre inside.

    Cropping (rather than only resizing) is what teaches scale invariance: the
    same face appears at many pixel sizes across epochs.
    """
    h, w = img.shape[:2]
    if boxes.shape[0] == 0:
        return img, boxes, lmks, labels
    for _ in range(max_tries):
        scale = random.choice([0.3, 0.45, 0.6, 0.8, 1.0])
        side = int(min(h, w) * scale)
        if side < 16:
            continue
        x0 = random.randint(0, w - side) if w > side else 0
        y0 = random.randint(0, h - side) if h > side else 0
        roi = np.array([x0, y0, x0 + side, y0 + side], dtype=np.float32)

        centres = (boxes[:, :2] + boxes[:, 2:]) / 2
        inside = ((centres[:, 0] > roi[0]) & (centres[:, 0] < roi[2]) &
                  (centres[:, 1] > roi[1]) & (centres[:, 1] < roi[3]))
        if not inside.any():
            continue

        crop = img[y0:y0 + side, x0:x0 + side]
        b = boxes[inside].copy()
        l = lmks[inside].copy()
        lab = labels[inside].copy()

        b[:, 0::2] = np.clip(b[:, 0::2] - roi[0], 0, side - 1)
        b[:, 1::2] = np.clip(b[:, 1::2] - roi[1], 0, side - 1)

        valid_l = l[:, 0] >= 0
        if valid_l.any():
            l[valid_l, 0::2] -= roi[0]
            l[valid_l, 1::2] -= roi[1]
            # A landmark pushed outside the crop makes the whole 5-point set useless.
            outside = ((l[:, 0::2] < 0) | (l[:, 0::2] > side - 1) |
                       (l[:, 1::2] < 0) | (l[:, 1::2] > side - 1)).any(axis=1)
            l[valid_l & outside] = -1.0
        # Drop degenerate boxes produced by the clipping above.
        keep = ((b[:, 2] - b[:, 0]) > 2) & ((b[:, 3] - b[:, 1]) > 2)
        if not keep.any():
            continue
        return crop, b[keep], l[keep], lab[keep]
    return img, boxes, lmks, labels


def _mirror(img, boxes, lmks):
    h, w = img.shape[:2]
    img = np.ascontiguousarray(img[:, ::-1])
    b = boxes.copy()
    b[:, 0] = w - boxes[:, 2]
    b[:, 2] = w - boxes[:, 0]
    l = lmks.copy()
    valid = l[:, 0] >= 0
    if valid.any():
        pts = l[valid].reshape(-1, 5, 2)
        pts[:, :, 0] = w - pts[:, :, 0]
        for a, bb in FLIP_PAIRS:
            pts[:, [a, bb]] = pts[:, [bb, a]]
        l[valid] = pts.reshape(-1, 10)
    return img, b, l


def _pad_to_square(img, boxes, lmks, fill: int = 114):
    h, w = img.shape[:2]
    if h == w:
        return img, boxes, lmks
    side = max(h, w)
    out = np.full((side, side, 3), fill, dtype=img.dtype)
    out[:h, :w] = img  # top-left placement keeps coordinates unchanged
    return out, boxes, lmks


def _resize_norm(img, boxes, lmks, size: int):
    h, w = img.shape[:2]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    b = boxes.copy()
    b[:, 0::2] /= float(w)
    b[:, 1::2] /= float(h)
    l = lmks.copy()
    valid = l[:, 0] >= 0
    if valid.any():
        l[valid, 0::2] /= float(w)
        l[valid, 1::2] /= float(h)
    return img, b, l


class WiderFaceDataset(Dataset):
    def __init__(self, label_file, images_root, img_size: int = 640, train: bool = True,
                 min_face_px: int = 6):
        self.records = parse_annotations(label_file, images_root)
        if not self.records:
            raise RuntimeError("no usable annotations found in %s" % label_file)
        self.img_size = img_size
        self.train = train
        self.min_face_px = min_face_px

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        for _ in range(10):
            rec = self.records[idx]
            img = read_image_rgb(rec["path"])
            if img is None:
                log.warning("unreadable detection image: %s", rec["path"])
                idx = random.randrange(len(self.records))
                continue

            boxes = rec["boxes"].copy()
            lmks = rec["landmarks"].copy()
            labels = rec["labels"].copy()

            if self.train:
                img, boxes, lmks, labels = _random_square_crop(img, boxes, lmks, labels)
                img = _distort(img)
                img, boxes, lmks = _pad_to_square(img, boxes, lmks)
                if random.random() < 0.5:
                    img, boxes, lmks = _mirror(img, boxes, lmks)
            else:
                img, boxes, lmks = _pad_to_square(img, boxes, lmks)

            img, boxes, lmks = _resize_norm(img, boxes, lmks, self.img_size)

            # Drop faces that are now smaller than the finest anchor can represent.
            wpx = (boxes[:, 2] - boxes[:, 0]) * self.img_size
            hpx = (boxes[:, 3] - boxes[:, 1]) * self.img_size
            keep = (wpx >= self.min_face_px) & (hpx >= self.min_face_px)
            boxes, lmks, labels = boxes[keep], lmks[keep], labels[keep]
            if boxes.shape[0] == 0:
                idx = random.randrange(len(self.records))
                continue

            tensor = torch.from_numpy(
                ((img.astype(np.float32) - PIXEL_MEAN) / PIXEL_STD).transpose(2, 0, 1).copy()
            )
            target = np.concatenate([boxes, lmks, labels[:, None]], axis=1).astype(np.float32)
            return tensor, torch.from_numpy(target)
        raise RuntimeError("could not produce a valid detection sample after 10 tries")


def detection_collate(batch):
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = [b[1] for b in batch]
    return images, targets
