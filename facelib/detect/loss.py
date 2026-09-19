"""Multi-task detection loss: box regression + face/background classification + landmarks.

The hard part of a single-shot detector is the class imbalance: a 640x640 input
has ~16k anchors and a photo typically contains a handful of faces, so >99.9% of
anchors are negative. Two standard remedies are implemented:

  * hard negative mining (default) - keep only the `neg_pos_ratio` highest-loss
    negatives per image, so easy background is not allowed to average the loss
    down to nothing;
  * focal loss (`use_focal: true`) - keep every anchor but down-weight the easy
    ones. Slower per step, usually a little better on tiny faces.
"""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchors import DEFAULT_VARIANCES, match_anchors


class MultiTaskDetectionLoss(nn.Module):
    def __init__(
        self,
        iou_threshold: float = 0.35,
        neg_pos_ratio: int = 7,
        variances: Sequence[float] = DEFAULT_VARIANCES,
        loc_weight: float = 2.0,
        landmark_weight: float = 1.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
    ):
        super().__init__()
        self.iou_threshold = iou_threshold
        self.neg_pos_ratio = neg_pos_ratio
        self.variances = list(variances)
        self.loc_weight = loc_weight
        self.landmark_weight = landmark_weight
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

    def forward(self, predictions, anchors: torch.Tensor, targets: list[torch.Tensor]):
        """predictions: (loc (N,A,4), logits (N,A,2), lmk (N,A,10)) from TinyFaceDetector.
        anchors: (A, 4) centre-form, normalised.
        targets: list of (M_i, 15) tensors - [x1,y1,x2,y2, 10 landmark coords, label].
        """
        loc_p, logits_p, lmk_p = predictions
        N, A = loc_p.shape[0], loc_p.shape[1]
        device = loc_p.device

        loc_t = torch.zeros(N, A, 4, device=device)
        lmk_t = torch.zeros(N, A, 10, device=device)
        conf_t = torch.zeros(N, A, dtype=torch.long, device=device)
        lmk_valid = torch.zeros(N, A, dtype=torch.bool, device=device)

        with torch.no_grad():
            for i, tgt in enumerate(targets):
                if tgt.numel() == 0:
                    continue
                tgt = tgt.to(device)
                truths = tgt[:, 0:4]
                lmks = tgt[:, 4:14]
                labels = tgt[:, 14].long()
                l, c, lm, lv = match_anchors(truths, lmks, anchors, labels,
                                             self.iou_threshold, self.variances)
                loc_t[i], conf_t[i], lmk_t[i], lmk_valid[i] = l, c, lm, lv

        pos = conf_t > 0
        num_pos = int(pos.sum().item())

        # ------------------------------------------------------------- box loss
        if num_pos > 0:
            loss_loc = F.smooth_l1_loss(loc_p[pos], loc_t[pos], reduction="sum")
        else:
            loss_loc = loc_p.sum() * 0.0

        # -------------------------------------------------------- landmark loss
        if lmk_valid.any():
            loss_lmk = F.smooth_l1_loss(lmk_p[lmk_valid], lmk_t[lmk_valid], reduction="sum")
            num_lmk = int(lmk_valid.sum().item())
        else:
            loss_lmk = lmk_p.sum() * 0.0
            num_lmk = 1

        # -------------------------------------------------- classification loss
        if self.use_focal:
            loss_cls = self._focal(logits_p, conf_t)
        else:
            loss_cls = self._mined_ce(logits_p, conf_t, pos, num_pos)

        denom = max(num_pos, 1)
        loss_loc = self.loc_weight * loss_loc / denom
        loss_cls = loss_cls / denom
        loss_lmk = self.landmark_weight * loss_lmk / max(num_lmk, 1)
        total = loss_loc + loss_cls + loss_lmk
        stats = {
            "loss": float(total.detach()),
            "loss_box": float(loss_loc.detach()),
            "loss_cls": float(loss_cls.detach()),
            "loss_lmk": float(loss_lmk.detach()),
            "num_pos": float(num_pos),
        }
        return total, stats

    # ------------------------------------------------------------------ helpers
    def _mined_ce(self, logits: torch.Tensor, conf_t: torch.Tensor,
                  pos: torch.Tensor, num_pos: int) -> torch.Tensor:
        N, A, C = logits.shape
        flat = logits.view(-1, C)
        # Per-anchor "how wrong is background" score, used only for ranking.
        with torch.no_grad():
            loss_all = F.cross_entropy(flat, conf_t.view(-1), reduction="none").view(N, A)
            loss_neg = loss_all.clone()
            loss_neg[pos] = 0.0
            _, idx = loss_neg.sort(dim=1, descending=True)
            rank = idx.argsort(dim=1)
            num_neg = torch.clamp(self.neg_pos_ratio * pos.sum(dim=1, keepdim=True),
                                  min=1, max=A - 1)
            neg = rank < num_neg
        keep = pos | neg
        return F.cross_entropy(logits[keep], conf_t[keep], reduction="sum")

    def _focal(self, logits: torch.Tensor, conf_t: torch.Tensor) -> torch.Tensor:
        N, A, C = logits.shape
        flat = logits.view(-1, C)
        target = conf_t.view(-1)
        logp = F.log_softmax(flat, dim=1)
        nll = -logp.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = torch.exp(-nll)
        alpha = torch.where(target > 0,
                            torch.full_like(nll, self.focal_alpha),
                            torch.full_like(nll, 1.0 - self.focal_alpha))
        return (alpha * (1 - pt) ** self.focal_gamma * nll).sum()
