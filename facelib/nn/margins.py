"""Angular margin classification heads - implemented from the formulas.

A face embedding is trained as a classification problem over training identities,
but with the logit replaced by an *angle*. Both the weight column of a class and
the embedding are L2-normalised, so the logit is exactly cos(theta):

    logit_j = s * cos(theta_j),      theta_j = angle(embedding, W_j)

and for the ground-truth class a margin is inserted:

    logit_y = s * ( cos(m1 * theta_y + m2) - m3 )

  m1 (multiplicative, SphereFace), m2 (additive angular, ArcFace),
  m3 (additive cosine, CosFace). One head covers all three:

    ArcFace    m1=1.00  m2=0.50  m3=0.00
    CosFace    m1=1.00  m2=0.00  m3=0.35
    SphereFace m1=1.35  m2=0.00  m3=0.00
    NormFace   m1=1.00  m2=0.00  m3=0.00   (no margin - useful for warmup)

Why it works: pushing the correct class *past* a margin in angle forces intra-class
angles to shrink and inter-class angles to grow, which is precisely what makes
cosine similarity of the resulting embeddings a usable identity metric at test time.

`sub_centers > 1` (sub-center ArcFace) keeps K weight columns per identity and
takes the closest one, so label noise and extreme poses get their own centre
instead of dragging the main one.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS: dict[str, tuple[float, float, float]] = {
    "arcface": (1.0, 0.5, 0.0),
    "cosface": (1.0, 0.0, 0.35),
    "sphereface": (1.35, 0.0, 0.0),
    "normface": (1.0, 0.0, 0.0),
    "combined": (1.0, 0.3, 0.2),  # the "glint360k" style combination
}


class CombinedMarginHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        m1: float = 1.0,
        m2: float = 0.5,
        m3: float = 0.0,
        sub_centers: int = 1,
        interclass_filtering: float = 0.0,
        margin_warmup_steps: int = 0,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("need at least 2 identities to train a margin head")
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.m1, self.m2, self.m3 = m1, m2, m3
        self.K = max(int(sub_centers), 1)
        self.interclass_filtering = interclass_filtering
        self.margin_warmup_steps = max(int(margin_warmup_steps), 0)
        self.register_buffer("_step", torch.zeros((), dtype=torch.long), persistent=True)

        self.weight = nn.Parameter(torch.empty(num_classes * self.K, embedding_dim))
        nn.init.normal_(self.weight, std=0.01)

    # ------------------------------------------------------------------ helpers
    @classmethod
    def from_preset(cls, name: str, embedding_dim: int, num_classes: int, **kw) -> "CombinedMarginHead":
        key = name.lower()
        if key not in PRESETS:
            raise ValueError("unknown margin preset %r (have %s)" % (name, sorted(PRESETS)))
        m1, m2, m3 = PRESETS[key]
        kw.setdefault("m1", m1)
        kw.setdefault("m2", m2)
        kw.setdefault("m3", m3)
        return cls(embedding_dim, num_classes, **kw)

    def _margin_factor(self) -> float:
        """0 -> 1 ramp over `margin_warmup_steps`.

        Starting at full margin from random weights makes the target logit the
        *smallest* one for a while and the loss plateaus; ramping avoids that
        without needing a separate warmup stage.
        """
        if self.margin_warmup_steps == 0:
            return 1.0
        return float(min(1.0, (self._step.item() + 1) / self.margin_warmup_steps))

    def cosine(self, embedding: torch.Tensor) -> torch.Tensor:
        """(N, D) -> (N, C) cosine similarity to every class centre."""
        x = F.normalize(embedding.float(), dim=1)
        w = F.normalize(self.weight.float(), dim=1)
        cos = x @ w.t()
        if self.K > 1:
            cos = cos.view(-1, self.num_classes, self.K).amax(dim=2)
        return cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    # ------------------------------------------------------------------ forward
    def forward(self, embedding: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Returns scaled logits ready for cross-entropy."""
        cos = self.cosine(embedding)

        if self.training and self.interclass_filtering > 0:
            cos = self._filter_interclass(cos, labels)

        idx = torch.arange(cos.size(0), device=cos.device)
        target_cos = cos[idx, labels]

        f = self._margin_factor()
        m1 = 1.0 + (self.m1 - 1.0) * f
        m2, m3 = self.m2 * f, self.m3 * f

        if m1 == 1.0 and m2 == 0.0:
            new_target = target_cos - m3
        else:
            theta = torch.acos(target_cos)
            new_theta = (m1 * theta + m2).clamp(max=math.pi - 1e-6)
            new_target = torch.cos(new_theta) - m3

        logits = cos.clone()
        logits[idx, labels] = new_target.to(logits.dtype)
        if self.training:
            self._step += 1
        return logits * self.scale

    def _filter_interclass(self, cos: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Drop the top-`interclass_filtering` fraction of non-target classes.

        Very similar non-target identities are usually the same person under a
        different label (scraped datasets are noisy); masking them out stops the
        margin from fighting over a duplicate identity.
        """
        with torch.no_grad():
            k = int(self.num_classes * self.interclass_filtering)
            if k < 1:
                return cos
            idx = torch.arange(cos.size(0), device=cos.device)
            masked = cos.clone()
            masked[idx, labels] = -2.0
            topk = masked.topk(k, dim=1).indices
            drop = torch.zeros_like(cos, dtype=torch.bool).scatter_(1, topk, True)
            drop[idx, labels] = False
        return cos.masked_fill(drop, -1.0)

    def extra_repr(self) -> str:
        return "dim=%d, classes=%d, K=%d, s=%.1f, m1=%.2f, m2=%.2f, m3=%.2f" % (
            self.embedding_dim, self.num_classes, self.K, self.scale, self.m1, self.m2, self.m3)


class LinearClassifier(nn.Module):
    """Plain softmax head. Only here as an ablation baseline - it produces noticeably
    worse embeddings than any margin head, which is the whole point of the margin."""

    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, embedding: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(embedding)


def build_head(name: str, embedding_dim: int, num_classes: int, **kw) -> nn.Module:
    name = name.lower()
    if name == "linear":
        return LinearClassifier(embedding_dim, num_classes)
    return CombinedMarginHead.from_preset(name, embedding_dim, num_classes, **kw)
