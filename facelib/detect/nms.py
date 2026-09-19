"""Non-maximum suppression, written out rather than pulled from torchvision.

`nms` is the standard greedy version: sort by score, keep the top box, drop
everything that overlaps it beyond `iou_threshold`, repeat.

`soft_nms` decays neighbour scores instead of deleting them, which recovers
genuinely overlapping faces (a crowd, or a face in front of another) that greedy
NMS throws away. Worth switching on for group photos.
"""
from __future__ import annotations

import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4,
        top_k: int = -1) -> np.ndarray:
    """boxes: (N, 4) corner form. Returns kept indices, highest score first."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1e-9) * (y2 - y1 + 1e-9)
    order = scores.argsort()[::-1]
    if top_k > 0:
        order = order[:top_k]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def soft_nms(boxes: np.ndarray, scores: np.ndarray, sigma: float = 0.5,
             score_threshold: float = 0.05, method: str = "gaussian"):
    """Returns (kept_indices, decayed_scores_for_those_indices)."""
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)
    idx = np.arange(len(scores))
    s = scores.astype(np.float64).copy()
    b = boxes.astype(np.float64).copy()
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    keep: list[int] = []
    kept_scores: list[float] = []

    while len(s) > 0:
        m = int(np.argmax(s))
        if s[m] < score_threshold:
            break
        keep.append(int(idx[m]))
        kept_scores.append(float(s[m]))
        # Swap the maximum to the front, then compare the tail against it.
        for arr in (s, idx, areas):
            arr[[0, m]] = arr[[m, 0]]
        b[[0, m]] = b[[m, 0]]
        if len(s) == 1:
            break
        xx1 = np.maximum(b[0, 0], b[1:, 0])
        yy1 = np.maximum(b[0, 1], b[1:, 1])
        xx2 = np.minimum(b[0, 2], b[1:, 2])
        yy2 = np.minimum(b[0, 3], b[1:, 3])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[0] + areas[1:] - inter + 1e-9)
        if method == "linear":
            decay = np.where(iou > 0.3, 1.0 - iou, 1.0)
        else:  # gaussian
            decay = np.exp(-(iou ** 2) / sigma)
        s = s[1:] * decay
        b = b[1:]
        idx = idx[1:]
        areas = areas[1:]
        alive = s >= score_threshold
        s, b, idx, areas = s[alive], b[alive], idx[alive], areas[alive]
    return np.asarray(keep, dtype=np.int64), np.asarray(kept_scores, dtype=np.float32)


def clip_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] = out[:, 0].clip(0, width - 1)
    out[:, 1] = out[:, 1].clip(0, height - 1)
    out[:, 2] = out[:, 2].clip(0, width - 1)
    out[:, 3] = out[:, 3].clip(0, height - 1)
    return out


def filter_small(boxes: np.ndarray, min_size: float) -> np.ndarray:
    """Indices of boxes whose shorter side is at least `min_size` pixels."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.nonzero(np.minimum(w, h) >= min_size)[0]
