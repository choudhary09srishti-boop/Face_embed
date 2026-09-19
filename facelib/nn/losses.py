"""Losses and metrics for embedding training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothCE(nn.Module):
    """Cross-entropy with label smoothing.

    Mild smoothing (0.05-0.1) helps when the identity labels are scraped/noisy:
    it stops the margin head from driving one logit to infinity on a mislabelled
    image. Set 0.0 for clean, curated datasets.
    """

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.smoothing <= 0:
            return F.cross_entropy(logits, target)
        n = logits.size(1)
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        smooth = -logp.mean(dim=1)
        return ((1 - self.smoothing) * nll + self.smoothing * smooth).mean()


class FocalCE(nn.Module):
    """Focal cross-entropy: down-weights already-correct samples.

    Useful late in training on long-tailed identity distributions, where most of
    the batch is easy and the gradient signal from rare identities gets drowned.
    """

    def __init__(self, gamma: float = 2.0, smoothing: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits.float(), dim=1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = torch.exp(-nll)
        loss = ((1 - pt) ** self.gamma) * nll
        if self.smoothing > 0:
            loss = (1 - self.smoothing) * loss + self.smoothing * (-logp.mean(dim=1))
        return loss.mean()


def build_criterion(name: str = "ce", smoothing: float = 0.0, gamma: float = 2.0) -> nn.Module:
    name = (name or "ce").lower()
    if name in ("ce", "cross_entropy", "label_smooth"):
        return LabelSmoothCE(smoothing)
    if name == "focal":
        return FocalCE(gamma, smoothing)
    raise ValueError("unknown criterion %r" % (name,))


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1,)) -> list[float]:
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(target.view(-1, 1))
    out = []
    for k in topk:
        k = min(k, maxk)
        out.append(correct[:, :k].any(dim=1).float().mean().item() * 100.0)
    return out


@torch.no_grad()
def embedding_stats(embeddings: torch.Tensor) -> dict[str, float]:
    """Diagnostics that catch a collapsing embedding before the eval does.

    `norm_mean` drifting toward 0 or `pairwise_cos_mean` toward 1.0 means every
    face is mapping to the same direction - usually lr too high or margin too
    large for the number of identities.
    """
    e = embeddings.detach().float()
    norms = e.norm(dim=1)
    en = F.normalize(e, dim=1)
    sim = en @ en.t()
    n = sim.size(0)
    off = sim[~torch.eye(n, dtype=torch.bool, device=sim.device)]
    return {
        "norm_mean": norms.mean().item(),
        "norm_std": norms.std(unbiased=False).item() if n > 1 else 0.0,
        "pairwise_cos_mean": off.mean().item() if off.numel() else 0.0,
        "pairwise_cos_max": off.max().item() if off.numel() else 0.0,
    }
