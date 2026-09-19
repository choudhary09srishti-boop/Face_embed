"""Landmark-based face alignment to a canonical crop.

The canonical 5-point template below is the de-facto standard for 112x112 face
recognition crops (left eye, right eye, nose tip, left mouth corner, right mouth
corner). Training and inference must use the *same* template and the same crop
size, otherwise embeddings from the two paths are not comparable - that is the
single most common cause of "my accuracy collapsed at deployment".
"""
from __future__ import annotations

import cv2
import numpy as np

from .umeyama import apply_affine, invert_affine, similarity_matrix_2x3, transform_residual

# Reference landmarks for a 112x112 crop, in (x, y) pixels.
REF_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)


def reference_landmarks(size: int = 112) -> np.ndarray:
    """Template scaled to a square crop of `size` pixels."""
    if size == 112:
        return REF_LANDMARKS_112.copy()
    return REF_LANDMARKS_112 * (float(size) / 112.0)


def align_face(
    image: np.ndarray,
    landmarks: np.ndarray,
    size: int = 112,
    border_value: int = 0,
    return_matrix: bool = False,
):
    """Warp a face onto the canonical template.

    image: HxWx3 (RGB or BGR - the warp is colour agnostic)
    landmarks: (5, 2) array of (x, y) in image coordinates
    """
    lmk = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if lmk.shape[0] != 5:
        raise ValueError("align_face expects 5 landmarks, got %d" % lmk.shape[0])
    ref = reference_landmarks(size)
    M = similarity_matrix_2x3(lmk, ref)
    crop = cv2.warpAffine(
        image, M, (size, size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(border_value,) * 3,
    )
    if return_matrix:
        return crop, M
    return crop


def alignment_error(landmarks: np.ndarray, size: int = 112) -> float:
    """Reprojection residual of the 5 landmarks, in template pixels."""
    lmk = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    ref = reference_landmarks(size)
    M = similarity_matrix_2x3(lmk, ref)
    return transform_residual(lmk, ref, M)


def crop_bbox(image: np.ndarray, bbox, size: int = 112, margin: float = 0.25) -> np.ndarray:
    """Landmark-free fallback: square-expand the box, pad if it leaves the image, resize.

    Embeddings from this path are measurably worse than aligned ones (no roll
    correction, no scale normalisation), so it is only used when a detector
    returns a box without landmarks.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * (1.0 + margin)
    half = side / 2.0
    sx1, sy1 = int(round(cx - half)), int(round(cy - half))
    sx2, sy2 = int(round(cx + half)), int(round(cy + half))

    pad_l, pad_t = max(0, -sx1), max(0, -sy1)
    pad_r, pad_b = max(0, sx2 - w), max(0, sy2 - h)
    sx1, sy1 = max(sx1, 0), max(sy1, 0)
    sx2, sy2 = min(sx2, w), min(sy2, h)
    if sx2 <= sx1 or sy2 <= sy1:
        return np.zeros((size, size, 3), dtype=image.dtype)
    patch = image[sy1:sy2, sx1:sx2]
    if pad_l or pad_t or pad_r or pad_b:
        patch = cv2.copyMakeBorder(patch, pad_t, pad_b, pad_l, pad_r,
                                   cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if patch.shape[0] > size else cv2.INTER_LINEAR
    return cv2.resize(patch, (size, size), interpolation=interp)


def align_or_crop(image: np.ndarray, bbox, landmarks: np.ndarray | None, size: int = 112,
                  margin: float = 0.25) -> np.ndarray:
    if landmarks is not None and np.asarray(landmarks).size == 10:
        return align_face(image, landmarks, size)
    return crop_bbox(image, bbox, size, margin)


def landmarks_to_original(landmarks_in_crop: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Map points from the aligned crop back into the source image."""
    return apply_affine(landmarks_in_crop, invert_affine(M))


def draw_faces(image: np.ndarray, boxes, landmarks=None, labels=None,
               color=(0, 255, 0)) -> np.ndarray:
    """Debug overlay - boxes, 5 landmarks and optional text per face."""
    out = image.copy()
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, boxes.shape[-1] if hasattr(boxes, "shape") else 4)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if labels is not None and i < len(labels):
            text = str(labels[i])
            cv2.putText(out, text, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
        if landmarks is not None and i < len(landmarks):
            for (lx, ly) in np.asarray(landmarks[i], dtype=np.float32).reshape(-1, 2):
                cv2.circle(out, (int(round(lx)), int(round(ly))), 2, (255, 0, 0), -1)
    return out
