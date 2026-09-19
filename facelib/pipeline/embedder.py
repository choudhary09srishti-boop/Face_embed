"""End-to-end pipeline: image (any size, any number of faces) -> one embedding per face.

    detect  ->  5 landmarks  ->  similarity-align to 112x112  ->  backbone  ->  L2-normalised 512-d

Every face in the image is processed, and the crops are batched through the
backbone in one forward pass, so a group photo costs barely more than a portrait.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from ..align.warp import align_face, alignment_error, crop_bbox
from ..data.dataset import read_image_rgb
from ..data.transforms import preprocess_batch
from ..detect.bootstrap import (align_by_eyes, build_bootstrap_detector, has_eye_landmarks,
                                has_full_landmarks)
from ..nn.builder import load_embedding_model

log = logging.getLogger(__name__)


@dataclass
class Face:
    """One detected face and its embedding."""

    bbox: np.ndarray                  # (4,) x1, y1, x2, y2 in source-image pixels
    det_score: float
    landmarks: np.ndarray | None       # (5, 2) or None when the detector gave none
    embedding: np.ndarray              # (D,) float32, L2-normalised
    align_error: float = -1.0          # landmark reprojection residual, px (-1 if unknown)
    align_mode: str = "landmarks"      # landmarks | eyes | bbox
    crop: np.ndarray | None = field(default=None, repr=False)

    @property
    def area(self) -> float:
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))

    def to_dict(self, include_embedding: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "bbox": [round(float(v), 2) for v in self.bbox],
            "det_score": round(float(self.det_score), 4),
            "landmarks": (None if self.landmarks is None
                          else [[round(float(x), 2), round(float(y), 2)] for x, y in self.landmarks]),
            "align_mode": self.align_mode,
            "align_error": round(float(self.align_error), 3),
        }
        if include_embedding:
            out["embedding"] = [float(v) for v in self.embedding]
            out["embedding_dim"] = int(self.embedding.shape[0])
        return out


class FaceEmbedder:
    def __init__(
        self,
        embedding_checkpoint: str | Path,
        detector: Any = "whole",
        device: str | torch.device | None = None,
        align_size: int = 112,
        flip_tta: bool = False,
        min_det_score: float = 0.6,
        max_faces: int | None = None,
        batch_size: int = 32,
        prefer_ema: bool = True,
        keep_crops: bool = False,
        max_align_error: float = -1.0,
    ):
        """`detector` may be: a ready detector object, a path to a TinyFace checkpoint,
        or the strings "whole" / "haar" (see detect/bootstrap.py)."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.meta = load_embedding_model(
            embedding_checkpoint, device=self.device, prefer_ema=prefer_ema, normalize=True
        )
        self.embedding_dim = int(self.meta["embedding_dim"])
        self.align_size = int(self.meta.get("input_size") or align_size)
        self.flip_tta = flip_tta
        self.min_det_score = min_det_score
        self.max_faces = max_faces
        self.batch_size = batch_size
        self.keep_crops = keep_crops
        self.max_align_error = max_align_error
        self.detector = self._resolve_detector(detector)

    def _resolve_detector(self, detector: Any):
        if detector is None:
            return build_bootstrap_detector("whole")
        if isinstance(detector, (str, Path)):
            name = str(detector)
            if name.lower() in ("whole", "wholeimage", "none", "haar"):
                return build_bootstrap_detector(name.lower())
            from ..detect.infer import FaceDetector  # local import: torch checkpoint path
            return FaceDetector(name, device=self.device, score_threshold=self.min_det_score)
        return detector  # already a detector-like object

    # ------------------------------------------------------------------ crops
    def _align(self, image: np.ndarray, box: np.ndarray, lmk: np.ndarray | None):
        """-> (crop, mode, residual). Falls back from 5-point to eyes to plain box crop."""
        if lmk is not None and has_full_landmarks(lmk):
            crop = align_face(image, lmk, self.align_size)
            return crop, "landmarks", alignment_error(lmk, self.align_size)
        if lmk is not None and has_eye_landmarks(lmk):
            return align_by_eyes(image, lmk, self.align_size), "eyes", -1.0
        return crop_bbox(image, box, self.align_size), "bbox", -1.0

    # -------------------------------------------------------------- embedding
    @torch.no_grad()
    def embed_crops(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Aligned uint8 RGB crops -> (N, D) L2-normalised embeddings."""
        if len(crops) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        out: list[np.ndarray] = []
        for start in range(0, len(crops), self.batch_size):
            chunk = crops[start:start + self.batch_size]
            batch = preprocess_batch(chunk, (self.align_size, self.align_size)).to(self.device)
            emb = self.model(batch)
            if self.flip_tta:
                # Average the embedding of the mirrored crop: a mirrored face is the
                # same identity, so this halves the variance for free.
                emb = emb + self.model(torch.flip(batch, dims=[3]))
                emb = torch.nn.functional.normalize(emb, dim=1)
            out.append(emb.float().cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32)

    def embed_image(self, image: np.ndarray) -> list[Face]:
        """image: HxWx3 uint8 RGB. Returns one Face per detection, largest first."""
        det = self.detector.detect(image)
        if len(det) == 0:
            return []
        keep = det.scores >= self.min_det_score if det.scores.size else np.ones(0, bool)
        boxes, scores, lmks = det.boxes[keep], det.scores[keep], det.landmarks[keep]
        if boxes.shape[0] == 0:
            return []

        order = np.argsort(-((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])))
        boxes, scores, lmks = boxes[order], scores[order], lmks[order]
        if self.max_faces is not None:
            boxes, scores, lmks = boxes[:self.max_faces], scores[:self.max_faces], lmks[:self.max_faces]

        crops, modes, errors = [], [], []
        for i in range(boxes.shape[0]):
            crop, mode, err = self._align(image, boxes[i], lmks[i] if lmks.size else None)
            crops.append(crop)
            modes.append(mode)
            errors.append(err)

        embeddings = self.embed_crops(crops)

        faces: list[Face] = []
        for i in range(len(crops)):
            bad_align = errors[i] >= 0 and errors[i] > self.max_align_error
            if self.max_align_error > 0 and bad_align:
                log.debug("dropping face with alignment residual %.2f px", errors[i])
                continue
            faces.append(Face(
                bbox=boxes[i].astype(np.float32),
                det_score=float(scores[i]),
                landmarks=(None if not lmks.size or not has_full_landmarks(lmks[i])
                           else lmks[i].astype(np.float32)),
                embedding=embeddings[i],
                align_error=float(errors[i]),
                align_mode=modes[i],
                crop=crops[i] if self.keep_crops else None,
            ))
        return faces

    def embed_path(self, path: str | Path) -> list[Face]:
        img = read_image_rgb(path)
        if img is None:
            raise FileNotFoundError("could not read image: %s" % path)
        return self.embed_image(img)

    def embed_paths(self, paths: Iterable[str | Path], skip_errors: bool = True):
        """Yields (path, faces) so a long batch job can stream results to disk."""
        for p in paths:
            try:
                yield p, self.embed_path(p)
            except Exception as exc:  # a corrupt file should not end a 100k-image run
                if not skip_errors:
                    raise
                log.warning("skipping %s: %s", p, exc)
                yield p, []

    def embed_largest_face(self, image: np.ndarray) -> Face | None:
        faces = self.embed_image(image)
        return faces[0] if faces else None

    def describe(self) -> dict[str, Any]:
        return {
            "backbone": self.meta["model_cfg"].get("name"),
            "embedding_dim": self.embedding_dim,
            "align_size": self.align_size,
            "detector": type(self.detector).__name__,
            "flip_tta": self.flip_tta,
            "device": str(self.device),
            "trained_epochs": self.meta.get("epoch"),
            "best_metric": self.meta.get("best_metric"),
        }
