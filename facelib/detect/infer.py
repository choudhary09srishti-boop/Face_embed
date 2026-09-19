"""Inference wrapper for TinyFaceDetector: image in, boxes + 5 landmarks out."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from ..data.transforms import PIXEL_MEAN, PIXEL_STD
from .anchors import (DEFAULT_MIN_SIZES, DEFAULT_STRIDES, DEFAULT_VARIANCES,
                      decode_boxes, decode_landmarks, generate_anchors)
from .model import build_detector
from .nms import clip_boxes, filter_small, nms, soft_nms

log = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    boxes: np.ndarray       # (N, 4) x1,y1,x2,y2 in original-image pixels
    scores: np.ndarray      # (N,)
    landmarks: np.ndarray   # (N, 5, 2) in original-image pixels

    def __len__(self) -> int:
        return int(self.boxes.shape[0])


class FaceDetector:
    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device = "cpu",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        max_size: int = 1024,
        min_face: int = 12,
        use_soft_nms: bool = False,
        top_k: int = 500,
    ):
        ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        cfg = ckpt.get("model_cfg", {})
        self.model = build_detector(cfg)
        state = ckpt.get("ema") or ckpt["model"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(device)

        self.device = device
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.max_size = max_size
        self.min_face = min_face
        self.use_soft_nms = use_soft_nms
        self.top_k = top_k
        self.strides = [int(s) for s in cfg.get("strides", DEFAULT_STRIDES)]
        self.min_sizes = [list(map(int, s)) for s in cfg.get("min_sizes", DEFAULT_MIN_SIZES)]
        self.variances = [float(v) for v in cfg.get("variances", DEFAULT_VARIANCES)]
        self._anchor_cache: dict[tuple[int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------ helpers
    def _anchors(self, hw: tuple[int, int]) -> torch.Tensor:
        """Anchors depend only on the padded input size, so cache them per size."""
        if hw not in self._anchor_cache:
            self._anchor_cache[hw] = generate_anchors(hw, self.strides, self.min_sizes,
                                                      device=self.device)
        return self._anchor_cache[hw]

    def _preprocess(self, image: np.ndarray):
        """Resize so the long side <= max_size, then pad to a multiple of the coarsest stride."""
        h, w = image.shape[:2]
        scale = min(self.max_size / max(h, w), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        rh, rw = image.shape[:2]
        step = max(self.strides)
        ph = int(np.ceil(rh / step) * step)
        pw = int(np.ceil(rw / step) * step)
        canvas = np.full((ph, pw, 3), 114.0, dtype=np.float32)  # same pad value as training
        canvas[:rh, :rw] = image.astype(np.float32)
        tensor = torch.from_numpy(((canvas - PIXEL_MEAN) / PIXEL_STD).transpose(2, 0, 1)).unsqueeze(0)
        return tensor, scale, (ph, pw)

    # -------------------------------------------------------------------- detect
    @torch.no_grad()
    def detect(self, image: np.ndarray) -> DetectionResult:
        """image: HxWx3 uint8 RGB."""
        if image is None or image.size == 0:
            return DetectionResult(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                                   np.zeros((0, 5, 2), np.float32))
        h0, w0 = image.shape[:2]
        tensor, scale, (ph, pw) = self._preprocess(image)
        tensor = tensor.to(self.device)
        loc, logits, lmk = self.model(tensor)

        anchors = self._anchors((ph, pw))
        scores = torch.softmax(logits[0].float(), dim=1)[:, 1]
        boxes = decode_boxes(loc[0].float(), anchors, self.variances)
        lmks = decode_landmarks(lmk[0].float(), anchors, self.variances)

        keep = scores > self.score_threshold
        if keep.sum() == 0:
            return DetectionResult(np.zeros((0, 4), np.float32), np.zeros((0,), np.float32),
                                   np.zeros((0, 5, 2), np.float32))
        scores = scores[keep].cpu().numpy()
        boxes = boxes[keep].cpu().numpy()
        lmks = lmks[keep].cpu().numpy()

        # Normalised (padded-image) coords -> original image pixels.
        boxes[:, 0::2] *= pw / scale
        boxes[:, 1::2] *= ph / scale
        lmks[:, 0::2] *= pw / scale
        lmks[:, 1::2] *= ph / scale

        order = scores.argsort()[::-1][:self.top_k]
        boxes, scores, lmks = boxes[order], scores[order], lmks[order]

        if self.use_soft_nms:
            kept, scores_kept = soft_nms(boxes, scores, score_threshold=self.score_threshold)
            boxes, lmks = boxes[kept], lmks[kept]
            scores = scores_kept
        else:
            kept = nms(boxes, scores, self.nms_threshold)
            boxes, scores, lmks = boxes[kept], scores[kept], lmks[kept]

        boxes = clip_boxes(boxes, h0, w0)
        if self.min_face > 0 and len(boxes):
            sel = filter_small(boxes, self.min_face)
            boxes, scores, lmks = boxes[sel], scores[sel], lmks[sel]

        return DetectionResult(
            boxes.astype(np.float32),
            scores.astype(np.float32),
            lmks.reshape(-1, 5, 2).astype(np.float32),
        )

    def detect_many(self, images: list[np.ndarray]) -> list[DetectionResult]:
        return [self.detect(im) for im in images]
