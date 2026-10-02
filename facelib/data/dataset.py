"""Identity-folder dataset: root/<identity>/*.jpg -> (image, label)."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image_rgb(path) -> np.ndarray | None:
    """Load an image as RGB uint8, or None if unreadable. Unicode-path safe on Windows."""
    path = str(path)
    try:
        data = np.fromfile(path, dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


def _scan_identities(root: Path, min_images: int, max_images: int | None) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for ident_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in ident_dir.rglob("*") if f.suffix.lower() in IMG_EXTS)
        if max_images:
            files = files[:max_images]
        if len(files) >= min_images:
            out[ident_dir.name] = [str(f) for f in files]
    return out


class FaceFolderDataset(Dataset):
    """root/<identity>/*.jpg -> (image tensor, label). One identity folder = one class."""

    def __init__(self, root, transform=None, min_images: int = 1,
                 max_images: int | None = None, index_cache: str | None = None,
                 class_to_idx: dict[str, int] | None = None):
        self.root = Path(root)
        self.transform = transform

        index = None
        if index_cache and Path(index_cache).exists():
            try:
                index = json.loads(Path(index_cache).read_text(encoding="utf-8"))
                log.info("loaded dataset index from %s (%d identities)", index_cache, len(index))
            except Exception:
                index = None

        if index is None:
            index = _scan_identities(self.root, min_images, max_images)
            if index_cache:
                Path(index_cache).parent.mkdir(parents=True, exist_ok=True)
                Path(index_cache).write_text(json.dumps(index), encoding="utf-8")

        if not index:
            raise RuntimeError("no identities with >= %d images found under %s" % (min_images, root))

        names = sorted(index.keys())
        self.class_to_idx = class_to_idx or {n: i for i, n in enumerate(names)}
        self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)

        self.samples: list[tuple[str, int]] = []
        for name, files in index.items():
            if name not in self.class_to_idx:
                continue  # e.g. class_to_idx restricted to a train-only identity split
            label = self.class_to_idx[name]
            for f in files:
                self.samples.append((f, label))

        if not self.samples:
            raise RuntimeError("no samples matched class_to_idx under %s" % root)

        self.targets = [label for _, label in self.samples]
        self.num_classes = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, label = self.samples[i]
        img = read_image_rgb(path)
        if img is None:
            log.warning("unreadable image, skipping: %s", path)
            return self.__getitem__((i + 1) % len(self.samples))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def save_classes(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.classes, indent=2), encoding="utf-8")