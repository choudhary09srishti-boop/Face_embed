"""Least-squares similarity transform (Umeyama, 1991), implemented with numpy only.

Given N corresponding 2D points, find the rotation R, isotropic scale c and
translation t that minimise  sum || dst_i - (c R src_i + t) ||^2.

This is the exact estimator behind landmark-based face alignment: 5 detected
landmarks are mapped onto a canonical 5-point template, and only rotation,
uniform scale and translation are allowed - no shear, no perspective - so the
face geometry that identifies a person is preserved while pose-induced roll and
scale are removed.
"""
from __future__ import annotations

import numpy as np


def umeyama(src: np.ndarray, dst: np.ndarray, estimate_scale: bool = True) -> np.ndarray:
    """Returns the (dim+1, dim+1) homogeneous transform mapping src -> dst.

    src, dst: (N, dim) float arrays with N >= dim.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.ndim != 2:
        raise ValueError("src and dst must be matching (N, dim) arrays")
    num, dim = src.shape
    if num < dim:
        raise ValueError("need at least %d point pairs, got %d" % (dim, num))

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Cross-covariance of the centred point sets.
    A = (dst_demean.T @ src_demean) / num

    # A reflection would mirror the face; force a proper rotation (det > 0).
    d = np.ones((dim,), dtype=np.float64)
    if np.linalg.det(A) < 0:
        d[dim - 1] = -1.0

    T = np.eye(dim + 1, dtype=np.float64)
    U, S, Vt = np.linalg.svd(A)
    rank = np.linalg.matrix_rank(A)
    if rank == 0:
        return np.full((dim + 1, dim + 1), np.nan)
    if rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(Vt) > 0:
            T[:dim, :dim] = U @ Vt
        else:
            saved = d[dim - 1]
            d[dim - 1] = -1.0
            T[:dim, :dim] = U @ np.diag(d) @ Vt
            d[dim - 1] = saved
    else:
        T[:dim, :dim] = U @ np.diag(d) @ Vt

    if estimate_scale:
        var = src_demean.var(axis=0).sum()
        scale = 1.0 if var < 1e-12 else (1.0 / var) * (S @ d)
    else:
        scale = 1.0

    T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean)
    T[:dim, :dim] *= scale
    return T


def similarity_matrix_2x3(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """The 2x3 affine matrix cv2.warpAffine expects."""
    T = umeyama(src, dst, estimate_scale=True)
    return T[:2, :].astype(np.float32)


def apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 2x3 matrix to (N, 2) points."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return (pts @ M[:, :2].T + M[:, 2]).astype(np.float32)


def invert_affine(M: np.ndarray) -> np.ndarray:
    """Inverse of a 2x3 affine - maps aligned-crop coordinates back to the original image."""
    A = np.asarray(M, dtype=np.float64)[:, :2]
    b = np.asarray(M, dtype=np.float64)[:, 2]
    Ainv = np.linalg.inv(A)
    out = np.zeros((2, 3), dtype=np.float64)
    out[:, :2] = Ainv
    out[:, 2] = -Ainv @ b
    return out.astype(np.float32)


def transform_residual(src: np.ndarray, dst: np.ndarray, M: np.ndarray) -> float:
    """Mean reprojection error in pixels - a cheap alignment-quality signal.

    A large residual means the 5 landmarks do not fit a rigid+scale model, which
    in practice means an extreme pose or a bad detection. Worth logging or using
    to reject a face before embedding it.
    """
    pred = apply_affine(src, M)
    return float(np.linalg.norm(pred - np.asarray(dst, dtype=np.float32), axis=1).mean())
