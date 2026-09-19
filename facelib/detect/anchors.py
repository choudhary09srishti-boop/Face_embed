"""Anchor (prior) generation, box encoding/decoding and ground-truth matching.

The detector is a single-shot anchor-based net. For every cell of every feature
level we place square anchors of fixed sizes; the network then predicts, per
anchor, (a) face/background logits, (b) a 4-number offset to the true box and
(c) 10 numbers for the 5 landmarks. Everything here is in *normalised* image
coordinates (0..1) so a trained model works at any input resolution.

Offsets are divided by `variances` (0.1 for centre/size, 0.2 for log-scale).
That is only a fixed rescaling to put the regression targets near unit variance,
which keeps the smooth-L1 loss balanced against the classification loss.
"""
from __future__ import annotations

import math
from typing import Sequence

import torch

# Anchor sizes per level, matched to strides 8 / 16 / 32. Two anchors per cell.
DEFAULT_MIN_SIZES: list[list[int]] = [[16, 32], [64, 128], [256, 512]]
DEFAULT_STRIDES: list[int] = [8, 16, 32]
DEFAULT_VARIANCES: list[float] = [0.1, 0.2]


def generate_anchors(
    image_size: tuple[int, int],
    strides: Sequence[int] = DEFAULT_STRIDES,
    min_sizes: Sequence[Sequence[int]] = DEFAULT_MIN_SIZES,
    device: torch.device | str = "cpu",
    clip: bool = False,
) -> torch.Tensor:
    """-> (num_anchors, 4) tensor of centre-form anchors (cx, cy, w, h), normalised.

    Ordering is level -> row -> col -> anchor size, which must match the order in
    which the heads flatten their predictions.
    """
    h, w = image_size
    anchors: list[list[float]] = []
    for level, stride in enumerate(strides):
        fh = int(math.ceil(h / stride))
        fw = int(math.ceil(w / stride))
        for i in range(fh):
            for j in range(fw):
                # Cell centre, expressed in normalised coordinates.
                cy = (i + 0.5) * stride / h
                cx = (j + 0.5) * stride / w
                for size in min_sizes[level]:
                    anchors.append([cx, cy, size / w, size / h])
    out = torch.tensor(anchors, dtype=torch.float32, device=device)
    if clip:
        out.clamp_(max=1.0, min=0.0)
    return out


def num_anchors_for(image_size: tuple[int, int], strides=DEFAULT_STRIDES,
                    min_sizes=DEFAULT_MIN_SIZES) -> int:
    h, w = image_size
    total = 0
    for level, stride in enumerate(strides):
        total += int(math.ceil(h / stride)) * int(math.ceil(w / stride)) * len(min_sizes[level])
    return total


# ------------------------------------------------------------------ conversions
def center_to_corner(boxes: torch.Tensor) -> torch.Tensor:
    """(cx, cy, w, h) -> (x1, y1, x2, y2)."""
    return torch.cat([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], dim=1)


def corner_to_center(boxes: torch.Tensor) -> torch.Tensor:
    """(x1, y1, x2, y2) -> (cx, cy, w, h)."""
    return torch.cat([(boxes[:, 2:] + boxes[:, :2]) / 2, boxes[:, 2:] - boxes[:, :2]], dim=1)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """IoU matrix (A, B) for corner-form boxes."""
    A, B = boxes_a.size(0), boxes_b.size(0)
    lt = torch.max(boxes_a[:, :2].unsqueeze(1).expand(A, B, 2),
                   boxes_b[:, :2].unsqueeze(0).expand(A, B, 2))
    rb = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                   boxes_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])).unsqueeze(1).expand_as(inter)
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])).unsqueeze(0).expand_as(inter)
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


# --------------------------------------------------------------------- encoding
def encode_boxes(matched: torch.Tensor, anchors: torch.Tensor,
                 variances: Sequence[float] = DEFAULT_VARIANCES) -> torch.Tensor:
    """matched: (N, 4) corner-form GT assigned to each anchor -> (N, 4) regression targets."""
    g_cxcy = (matched[:, :2] + matched[:, 2:]) / 2 - anchors[:, :2]
    g_cxcy = g_cxcy / (variances[0] * anchors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / anchors[:, 2:]
    g_wh = torch.log(g_wh.clamp(min=1e-9)) / variances[1]
    return torch.cat([g_cxcy, g_wh], dim=1)


def decode_boxes(loc: torch.Tensor, anchors: torch.Tensor,
                 variances: Sequence[float] = DEFAULT_VARIANCES) -> torch.Tensor:
    """Inverse of `encode_boxes`; returns corner-form boxes."""
    cxcy = anchors[:, :2] + loc[:, :2] * variances[0] * anchors[:, 2:]
    wh = anchors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
    boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], dim=1)
    return boxes


def encode_landmarks(matched: torch.Tensor, anchors: torch.Tensor,
                     variances: Sequence[float] = DEFAULT_VARIANCES) -> torch.Tensor:
    """matched: (N, 10) landmark coords -> offsets from the anchor centre, in anchor units."""
    pts = matched.view(matched.size(0), 5, 2)
    centre = anchors[:, :2].unsqueeze(1)
    size = anchors[:, 2:].unsqueeze(1)
    return ((pts - centre) / (variances[0] * size)).view(matched.size(0), 10)


def decode_landmarks(pre: torch.Tensor, anchors: torch.Tensor,
                     variances: Sequence[float] = DEFAULT_VARIANCES) -> torch.Tensor:
    pts = pre.view(pre.size(0), 5, 2)
    centre = anchors[:, :2].unsqueeze(1)
    size = anchors[:, 2:].unsqueeze(1)
    return (centre + pts * variances[0] * size).view(pre.size(0), 10)


# --------------------------------------------------------------------- matching
def match_anchors(
    truths: torch.Tensor,
    landmarks: torch.Tensor,
    anchors: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.35,
    variances: Sequence[float] = DEFAULT_VARIANCES,
):
    """Assign ground-truth faces to anchors.

    truths:    (M, 4) corner-form, normalised
    landmarks: (M, 10) normalised; a row of -1 means "no landmark annotation"
    labels:    (M,) 1 for a face, 0 for an ignored/invalid annotation

    Returns (loc_t (A,4), conf_t (A,), landm_t (A,10), landm_valid (A,) bool).

    Two rules, in order:
      1. every anchor whose best IoU with any face exceeds the threshold becomes
         a positive for that face;
      2. every face additionally claims its single best anchor, even below the
         threshold - otherwise tiny faces (WIDER FACE is full of them) would
         never receive a positive anchor and the model would learn to ignore them.
    """
    A = anchors.size(0)
    device = anchors.device
    if truths.numel() == 0:
        return (torch.zeros(A, 4, device=device),
                torch.zeros(A, dtype=torch.long, device=device),
                torch.zeros(A, 10, device=device),
                torch.zeros(A, dtype=torch.bool, device=device))

    overlaps = box_iou(truths, center_to_corner(anchors))  # (M, A)

    best_anchor_overlap, best_anchor_idx = overlaps.max(dim=1)   # per GT
    best_truth_overlap, best_truth_idx = overlaps.max(dim=0)     # per anchor

    # Discard GT boxes that could not be matched to any anchor at all.
    valid_gt = best_anchor_overlap >= 0.1
    if valid_gt.sum() == 0:
        return (torch.zeros(A, 4, device=device),
                torch.zeros(A, dtype=torch.long, device=device),
                torch.zeros(A, 10, device=device),
                torch.zeros(A, dtype=torch.bool, device=device))
    best_anchor_idx = best_anchor_idx[valid_gt]
    best_anchor_overlap = best_anchor_overlap[valid_gt]
    gt_indices = torch.nonzero(valid_gt, as_tuple=False).squeeze(1)

    # Rule 2: force each surviving GT's best anchor to point at it.
    best_truth_overlap.index_fill_(0, best_anchor_idx, 2.0)
    best_truth_idx[best_anchor_idx] = gt_indices

    matched = truths[best_truth_idx]
    conf = labels[best_truth_idx].clone()
    conf[best_truth_overlap < iou_threshold] = 0

    loc_t = encode_boxes(matched, anchors, variances)

    matched_lmk = landmarks[best_truth_idx]
    landm_t = encode_landmarks(matched_lmk, anchors, variances)
    # Mark rows whose GT has no landmark annotation; the loss skips them.
    no_lmk = (matched_lmk[:, 0] < 0)
    landm_t[no_lmk] = 0.0
    conf[no_lmk & (conf > 0)] = 1  # still a face for classification, just no landmark target
    landm_valid = (~no_lmk) & (conf > 0)
    return loc_t, conf, landm_t, landm_valid
