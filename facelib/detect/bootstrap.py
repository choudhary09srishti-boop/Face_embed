"""Bootstrap detectors - how you get aligned crops *before* TinyFace is trained.

Training the embedding network needs aligned crops, and aligning needs a face
detector, so there is a chicken-and-egg problem on day one. Three ways out, in
order of preference:

  * `WholeImageDetector` - your images are already tight face crops (most
    curated datasets are). Nothing is detected; the whole image is the face.
  * `HaarDetector` - OpenCV's bundled Haar cascade. This is the one component
    that is *not* written from scratch: it is a classical (non-neural) detector
    shipped inside opencv, used only to bootstrap a dataset. Optionally locates
    the two eyes and aligns on them, which is much better than a bare box crop.
  * `FaceDetector` (detect/infer.py) - the real thing, once you have trained it.

All three expose the same `detect(image_rgb) -> DetectionResult` interface, so
`FaceEmbedder` does not care which one it is handed. Landmarks are returned as
-1 when unknown, and the alignment code falls back to a box crop in that case.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..align.umeyama import similarity_matrix_2x3
from ..align.warp import reference_landmarks
from .infer import DetectionResult

log = logging.getLogger(__name__)

NO_LANDMARKS = -1.0


class WholeImageDetector:
    """Treats the entire image as a single face. For pre-cropped datasets."""

    def detect(self, image: np.ndarray) -> DetectionResult:
        h, w = image.shape[:2]
        return DetectionResult(
            boxes=np.array([[0, 0, w - 1, h - 1]], dtype=np.float32),
            scores=np.array([1.0], dtype=np.float32),
            landmarks=np.full((1, 5, 2), NO_LANDMARKS, dtype=np.float32),
        )

    def detect_many(self, images: list[np.ndarray]) -> list[DetectionResult]:
        return [self.detect(im) for im in images]


class HaarDetector:
    def __init__(
        self,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: int = 40,
        use_eyes: bool = True,
    ):
        base = Path(cv2.data.haarcascades)
        self.face = cv2.CascadeClassifier(str(base / "haarcascade_frontalface_default.xml"))
        if self.face.empty():
            raise RuntimeError("could not load opencv's frontal face cascade")
        self.eyes = None
        if use_eyes:
            eyes = cv2.CascadeClassifier(str(base / "haarcascade_eye.xml"))
            self.eyes = None if eyes.empty() else eyes
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, image: np.ndarray) -> DetectionResult:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self.face.detectMultiScale(
            gray, scaleFactor=self.scale_factor, minNeighbors=self.min_neighbors,
            minSize=(self.min_size, self.min_size),
        )
        if len(rects) == 0:
            return DetectionResult(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                                   np.zeros((0, 5, 2), np.float32))
        boxes, lmks = [], []
        for (x, y, w, h) in rects:
            boxes.append([x, y, x + w, y + h])
            lmks.append(self._eye_landmarks(gray, x, y, w, h))
        # Haar gives no score; use box area so the largest face sorts first.
        areas = np.array([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes], dtype=np.float32)
        order = areas.argsort()[::-1]
        boxes = np.asarray(boxes, dtype=np.float32)[order]
        lmks = np.asarray(lmks, dtype=np.float32)[order]
        scores = (areas[order] / areas.max()).astype(np.float32)
        return DetectionResult(boxes, scores, lmks.reshape(-1, 5, 2))

    def _eye_landmarks(self, gray: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
        """Return 5 pseudo-landmarks, or -1s if the two eyes were not found.

        Only the eye pair is real; the remaining three points are placed by the
        canonical template's proportions after the eye line fixes rotation and
        scale. That is enough for `align_pseudo`, which solves the similarity
        transform from the two eyes alone.
        """
        blank = np.full((5, 2), NO_LANDMARKS, dtype=np.float32)
        if self.eyes is None:
            return blank
        roi = gray[y:y + int(h * 0.6), x:x + w]
        if roi.size == 0:
            return blank
        found = self.eyes.detectMultiScale(roi, 1.1, 6, minSize=(max(w // 10, 8),) * 2)
        if len(found) < 2:
            return blank
        found = sorted(found, key=lambda r: r[2] * r[3], reverse=True)[:2]
        centres = [(x + ex + ew / 2.0, y + ey + eh / 2.0) for (ex, ey, ew, eh) in found]
        centres.sort(key=lambda p: p[0])  # left eye first, in image coordinates
        out = blank.copy()
        out[0] = centres[0]
        out[1] = centres[1]
        return out

    def detect_many(self, images: list[np.ndarray]) -> list[DetectionResult]:
        return [self.detect(im) for im in images]


def has_full_landmarks(landmarks: np.ndarray) -> bool:
    return bool(np.all(np.asarray(landmarks) >= 0))


def has_eye_landmarks(landmarks: np.ndarray) -> bool:
    lmk = np.asarray(landmarks).reshape(-1, 2)
    return bool(np.all(lmk[:2] >= 0))


def align_by_eyes(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity transform from the 2 eye points onto the canonical template.

    Two point pairs determine rotation, uniform scale and translation exactly, so
    this removes head roll and normalises inter-ocular distance - most of what
    full 5-point alignment buys you.
    """
    lmk = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)[:2]
    ref = reference_landmarks(size)[:2]
    M = similarity_matrix_2x3(lmk, ref)
    return cv2.warpAffine(image, M, (size, size), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def build_bootstrap_detector(kind: str, **kw):
    kind = (kind or "whole").lower()
    if kind in ("whole", "wholeimage", "none"):
        return WholeImageDetector()
    if kind == "haar":
        return HaarDetector(**kw)
    raise ValueError("unknown bootstrap detector %r (use 'whole' or 'haar')" % (kind,))
