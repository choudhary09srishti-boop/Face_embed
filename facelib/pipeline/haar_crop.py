"""Embed faces from uncropped photos: Haar box -> margin-1.7 crop -> embedding.

Margin 1.7 reproduces the framing of the LFW crops the model was trained on
(measured: median cosine 0.88 vs. the 'whole' path on the same photo).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..align.warp import crop_bbox
from ..data.dataset import read_image_rgb
from ..detect.bootstrap import build_bootstrap_detector

MARGIN = 1.7


@dataclass
class CropFace:
    bbox: np.ndarray
    embedding: np.ndarray

    @property
    def area(self) -> float:
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


class HaarCropEmbedder:
    def __init__(self, embedder, margin: float = MARGIN):
        self.embedder = embedder
        self.margin = margin
        self.haar = build_bootstrap_detector("haar")

    def embed_image(self, image: np.ndarray) -> list[CropFace]:
        """image: HxWx3 uint8 RGB. One CropFace per detected face, largest first."""
        det = self.haar.detect(image)
        if len(det) == 0:
            return []
        boxes = det.boxes
        order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        boxes = boxes[order]
        size = self.embedder.align_size
        crops = [crop_bbox(image, b, size, self.margin) for b in boxes]
        embs = self.embedder.embed_crops(crops)
        return [CropFace(bbox=boxes[i].astype(np.float32), embedding=embs[i])
                for i in range(len(crops))]

    def embed_path(self, path: str | Path) -> list[CropFace]:
        return self.embed_image(read_image_rgb(path))

    def embed_largest_face(self, image: np.ndarray) -> CropFace | None:
        faces = self.embed_image(image)
        return faces[0] if faces else None