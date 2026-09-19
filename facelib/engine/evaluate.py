"""Verification evaluation - the metric that actually tracks embedding quality.

Protocol (the LFW protocol, implemented here rather than imported):
  1. embed both images of every pair, L2-normalise, take the cosine similarity;
  2. split the pairs into 10 folds; for each fold, pick the threshold that
     maximises accuracy on the *other nine*, then score the held-out fold.
     Choosing the threshold on the same data you report is the classic way to
     overstate face-recognition accuracy, so it is done properly here;
  3. also report threshold-free numbers: ROC-AUC, EER, and TAR at fixed FAR.

TAR@FAR is the number to quote for a real deployment: it answers "how many
genuine matches do I catch if I am willing to accept one impostor in 1,000?".

ROC, AUC, EER and the fold logic are computed with numpy directly - no sklearn.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.pairs import PairsDataset, read_pairs
from ..data.transforms import build_eval_transform

log = logging.getLogger(__name__)


# --------------------------------------------------------------------- scoring
@torch.no_grad()
def score_pairs(
    model: torch.nn.Module,
    pairs: list[tuple[str, str, int]],
    root: str | Path,
    device: torch.device | str = "cpu",
    input_size: int = 112,
    batch_size: int = 64,
    num_workers: int = 4,
    flip_tta: bool = True,
    normalize_output: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """-> (cosine scores, labels). `model` must map NCHW [-1,1] -> (N, D)."""
    ds = PairsDataset(pairs, root, build_eval_transform((input_size, input_size)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=torch.device(device).type == "cuda")
    was_training = model.training
    model.eval()
    scores: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    def embed(x: torch.Tensor) -> torch.Tensor:
        e = model(x.to(device, non_blocking=True)).float()
        if flip_tta:
            e = e + model(torch.flip(x.to(device, non_blocking=True), dims=[3])).float()
        return torch.nn.functional.normalize(e, dim=1) if normalize_output or flip_tta else e

    for a, b, y in loader:
        ea, eb = embed(a), embed(b)
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    if was_training:
        model.train()
    return np.concatenate(scores), np.concatenate(labels).astype(np.int64)


# --------------------------------------------------------------------- metrics
def accuracy_at(threshold: float, scores: np.ndarray, labels: np.ndarray) -> float:
    pred = scores >= threshold
    return float((pred == labels.astype(bool)).mean())


def best_threshold(scores: np.ndarray, labels: np.ndarray, num_steps: int = 2000) -> tuple[float, float]:
    """Grid search over the observed score range. Returns (threshold, accuracy)."""
    lo, hi = float(scores.min()), float(scores.max())
    if hi - lo < 1e-9:
        return lo, accuracy_at(lo, scores, labels)
    grid = np.linspace(lo, hi, num_steps)
    accs = np.array([accuracy_at(t, scores, labels) for t in grid])
    i = int(accs.argmax())
    return float(grid[i]), float(accs[i])


def roc_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (fpr, tpr, thresholds), computed by sorting scores descending."""
    order = np.argsort(-scores)
    s = scores[order]
    y = labels[order].astype(bool)
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(int((~y).sum()), 1)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # Prepend the (0,0) point so the curve starts at "accept nothing".
    return (np.concatenate([[0.0], fpr]), np.concatenate([[0.0], tpr]),
            np.concatenate([[s[0] + 1e-6], s]))


def auc_score(fpr: np.ndarray, tpr: np.ndarray) -> float:
    trapz = getattr(np, "trapezoid", None) or np.trapz  # renamed in numpy 2.0
    return float(trapz(tpr, fpr))


def equal_error_rate(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray) -> tuple[float, float]:
    """EER: the operating point where FAR == FRR. Returns (eer, threshold)."""
    frr = 1.0 - tpr
    i = int(np.argmin(np.abs(fpr - frr)))
    return float((fpr[i] + frr[i]) / 2.0), float(thresholds[i])


def tar_at_far(fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray,
               target_far: float) -> tuple[float, float]:
    """Highest TAR whose FAR does not exceed `target_far`. Returns (tar, threshold)."""
    ok = np.nonzero(fpr <= target_far)[0]
    if ok.size == 0:
        return 0.0, float(thresholds[0])
    i = int(ok[-1])  # fpr is non-decreasing, so the last index is the loosest allowed
    return float(tpr[i]), float(thresholds[i])


def verification_metrics(scores: np.ndarray, labels: np.ndarray, folds: int = 10,
                         fars: tuple[float, ...] = (1e-1, 1e-2, 1e-3, 1e-4)) -> dict:
    n = len(scores)
    if n == 0:
        raise ValueError("no pairs to evaluate")
    folds = max(min(folds, n), 1)

    # Cross-validated accuracy: threshold fitted on the other folds, never on the fold itself.
    idx = np.arange(n)
    bounds = np.linspace(0, n, folds + 1).astype(int)
    fold_acc, fold_thr = [], []
    for k in range(folds):
        test = idx[bounds[k]:bounds[k + 1]]
        train = np.concatenate([idx[:bounds[k]], idx[bounds[k + 1]:]])
        if train.size == 0 or test.size == 0:
            continue
        thr, _ = best_threshold(scores[train], labels[train])
        fold_thr.append(thr)
        fold_acc.append(accuracy_at(thr, scores[test], labels[test]))

    fpr, tpr, thr = roc_curve(scores, labels)
    global_thr, global_acc = best_threshold(scores, labels)
    eer, eer_thr = equal_error_rate(fpr, tpr, thr)

    out = {
        "num_pairs": int(n),
        "num_positive": int(labels.sum()),
        "accuracy": float(np.mean(fold_acc)) if fold_acc else global_acc,
        "accuracy_std": float(np.std(fold_acc)) if fold_acc else 0.0,
        "threshold": float(np.mean(fold_thr)) if fold_thr else global_thr,
        "accuracy_no_cv": global_acc,
        "auc": auc_score(fpr, tpr),
        "eer": eer,
        "eer_threshold": eer_thr,
        "mean_cos_same": float(scores[labels == 1].mean()) if (labels == 1).any() else 0.0,
        "mean_cos_diff": float(scores[labels == 0].mean()) if (labels == 0).any() else 0.0,
    }
    for far in fars:
        tar, t = tar_at_far(fpr, tpr, thr, far)
        key = "tar@far%g" % far
        out[key] = tar
        out[key + "_threshold"] = t
    return out


def format_metrics(m: dict) -> str:
    parts = [
        "acc %.4f +- %.4f (thr %.3f)" % (m["accuracy"], m["accuracy_std"], m["threshold"]),
        "auc %.4f" % m["auc"],
        "eer %.4f" % m["eer"],
        "tar@1e-3 %.4f" % m.get("tar@far0.001", 0.0),
        "cos same/diff %.3f/%.3f" % (m["mean_cos_same"], m["mean_cos_diff"]),
    ]
    return " | ".join(parts)


def evaluate_pairs_file(
    model: torch.nn.Module,
    pairs_file: str | Path,
    root: str | Path,
    device: torch.device | str = "cpu",
    input_size: int = 112,
    batch_size: int = 64,
    num_workers: int = 4,
    flip_tta: bool = True,
    folds: int = 10,
) -> dict:
    pairs = read_pairs(pairs_file)
    scores, labels = score_pairs(model, pairs, root, device, input_size, batch_size,
                                 num_workers, flip_tta)
    return verification_metrics(scores, labels, folds)


# ------------------------------------------------------- identification (rank-N)
def identification_metrics(gallery_emb: np.ndarray, gallery_labels: list[str],
                           probe_emb: np.ndarray, probe_labels: list[str],
                           ranks: tuple[int, ...] = (1, 5)) -> dict:
    """Closed-set rank-N accuracy: is the right identity in the top N matches?"""
    g = gallery_emb / (np.linalg.norm(gallery_emb, axis=1, keepdims=True) + 1e-12)
    p = probe_emb / (np.linalg.norm(probe_emb, axis=1, keepdims=True) + 1e-12)
    sims = p @ g.T
    order = np.argsort(-sims, axis=1)
    gl = np.asarray(gallery_labels)
    out = {}
    for r in ranks:
        r_eff = min(r, g.shape[0])
        hit = 0
        for i, pl in enumerate(probe_labels):
            if pl in set(gl[order[i, :r_eff]]):
                hit += 1
        out["rank%d" % r] = hit / max(len(probe_labels), 1)
    return out
