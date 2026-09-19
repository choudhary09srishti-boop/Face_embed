"""TinyFace: a single-shot face detector with 5-point landmark regression.

Architecture (all written here, no torchvision backbones, no pretrained weights):

    depthwise-separable backbone  ->  C3 (stride 8), C4 (stride 16), C5 (stride 32)
    FPN                           ->  P3, P4, P5, all `fpn_channels` wide
    SSH context module per level  ->  widens the receptive field cheaply
    three 1x1 heads per level     ->  class logits (2), box offsets (4), landmarks (10)

Why landmarks are predicted here rather than by a separate model: the embedding
network needs an *aligned* crop, alignment needs 5 points, and running a second
network per face would dominate the cost. Regressing them from the same features
is nearly free.

Outputs are concatenated across levels in the same order `generate_anchors`
produces anchors, so prediction i corresponds to anchor i.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..nn.layers import ConvBnAct, init_weights


class DepthwiseSeparable(nn.Sequential):
    """3x3 depthwise + 1x1 pointwise - ~8x cheaper than a dense 3x3 at the same width."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, act: str = "leaky"):
        super().__init__(
            ConvBnAct(in_ch, in_ch, 3, stride, 1, groups=in_ch, act=act),
            ConvBnAct(in_ch, out_ch, 1, 1, 0, act=act),
        )


class TinyBackbone(nn.Module):
    """Returns three feature maps at strides 8, 16, 32."""

    def __init__(self, width: float = 0.25, act: str = "leaky"):
        super().__init__()

        def ch(c: int) -> int:
            return max(int(round(c * width / 8) * 8), 8)

        self.stage1 = nn.Sequential(
            ConvBnAct(3, ch(32), 3, 2, 1, act=act),        # /2
            DepthwiseSeparable(ch(32), ch(64), 1, act),
            DepthwiseSeparable(ch(64), ch(128), 2, act),   # /4
            DepthwiseSeparable(ch(128), ch(128), 1, act),
            DepthwiseSeparable(ch(128), ch(256), 2, act),  # /8
            DepthwiseSeparable(ch(256), ch(256), 1, act),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparable(ch(256), ch(512), 2, act),  # /16
            *[DepthwiseSeparable(ch(512), ch(512), 1, act) for _ in range(5)],
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparable(ch(512), ch(1024), 2, act),  # /32
            DepthwiseSeparable(ch(1024), ch(1024), 1, act),
        )
        self.out_channels = [ch(256), ch(512), ch(1024)]

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        c3 = self.stage1(x)
        c4 = self.stage2(c3)
        c5 = self.stage3(c4)
        return [c3, c4, c5]


class FPN(nn.Module):
    """Top-down feature pyramid: adds coarse semantic context to the fine levels
    that actually have to find 16-pixel faces."""

    def __init__(self, in_channels: list[int], out_ch: int = 64, act: str = "leaky"):
        super().__init__()
        self.lateral = nn.ModuleList([ConvBnAct(c, out_ch, 1, 1, 0, act=act) for c in in_channels])
        self.smooth = nn.ModuleList([ConvBnAct(out_ch, out_ch, 3, 1, 1, act=act)
                                     for _ in range(len(in_channels) - 1)])

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="nearest")
            laterals[i - 1] = self.smooth[i - 1](laterals[i - 1] + up)
        return laterals


class SSH(nn.Module):
    """Context module: concatenates 3x3, 5x5 and 7x7 receptive fields (the larger
    ones built from stacked 3x3 convs) so one head sees several face scales."""

    def __init__(self, channels: int, act: str = "leaky"):
        super().__init__()
        assert channels % 4 == 0, "SSH needs channels divisible by 4"
        q = channels // 4
        self.conv3 = ConvBnAct(channels, channels // 2, 3, 1, 1, act="none")
        self.conv5a = ConvBnAct(channels, q, 3, 1, 1, act=act)
        self.conv5b = ConvBnAct(q, q, 3, 1, 1, act="none")
        self.conv7a = ConvBnAct(q, q, 3, 1, 1, act=act)
        self.conv7b = ConvBnAct(q, q, 3, 1, 1, act="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b3 = self.conv3(x)
        a = self.conv5a(x)
        b5 = self.conv5b(a)
        b7 = self.conv7b(self.conv7a(a))
        return F.relu(torch.cat([b3, b5, b7], dim=1), inplace=True)


class _Head(nn.Module):
    def __init__(self, in_ch: int, num_anchors: int, outputs_per_anchor: int):
        super().__init__()
        self.outputs = outputs_per_anchor
        self.conv = nn.Conv2d(in_ch, num_anchors * outputs_per_anchor, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, A*k, H, W) -> (N, H*W*A, k); channel layout must match anchor ordering.
        out = self.conv(x).permute(0, 2, 3, 1).contiguous()
        return out.view(out.size(0), -1, self.outputs)


class TinyFaceDetector(nn.Module):
    def __init__(
        self,
        width: float = 0.25,
        fpn_channels: int = 64,
        anchors_per_level: int = 2,
        num_levels: int = 3,
        act: str = "leaky",
    ):
        super().__init__()
        self.backbone = TinyBackbone(width=width, act=act)
        self.fpn = FPN(self.backbone.out_channels, fpn_channels, act=act)
        self.ssh = nn.ModuleList([SSH(fpn_channels, act=act) for _ in range(num_levels)])
        self.cls_heads = nn.ModuleList([_Head(fpn_channels, anchors_per_level, 2) for _ in range(num_levels)])
        self.box_heads = nn.ModuleList([_Head(fpn_channels, anchors_per_level, 4) for _ in range(num_levels)])
        self.lmk_heads = nn.ModuleList([_Head(fpn_channels, anchors_per_level, 10) for _ in range(num_levels)])
        self.anchors_per_level = anchors_per_level
        self.num_levels = num_levels

        init_weights(self)
        # Bias the classifier toward "background": >99% of anchors are negative, and
        # without this the first iterations are dominated by a huge, useless loss.
        for head in self.cls_heads:
            nn.init.constant_(head.conv.bias[1::2], -4.0)

    def forward(self, x: torch.Tensor):
        """-> (box_offsets (N,A,4), class_logits (N,A,2), landmarks (N,A,10))."""
        feats = self.fpn(self.backbone(x))
        feats = [ssh(f) for ssh, f in zip(self.ssh, feats)]
        boxes = torch.cat([h(f) for h, f in zip(self.box_heads, feats)], dim=1)
        logits = torch.cat([h(f) for h, f in zip(self.cls_heads, feats)], dim=1)
        lmks = torch.cat([h(f) for h, f in zip(self.lmk_heads, feats)], dim=1)
        return boxes, logits, lmks


def build_detector(cfg: dict) -> TinyFaceDetector:
    return TinyFaceDetector(
        width=float(cfg.get("width", 0.25)),
        fpn_channels=int(cfg.get("fpn_channels", 64)),
        anchors_per_level=int(cfg.get("anchors_per_level", 2)),
        num_levels=int(cfg.get("num_levels", 3)),
        act=str(cfg.get("act", "leaky")),
    )
