"""MobileFaceNet - small, fast embedding backbone (~1M params, ~220 MFLOPs at 112x112).

Use this when you need CPU/edge inference or you only have a few thousand images:
an iresnet50 overfits a small dataset long before it converges, while this one
trains in minutes per epoch on a single cloud GPU.

Stage table (t = expansion, c = out channels, n = repeats, s = stride of first repeat):
    conv3x3      c=64        s=2   -> 56x56
    dw conv3x3   c=64        s=1
    bottleneck   t=2 c=64  n=5 s=2 -> 28x28
    bottleneck   t=4 c=128 n=1 s=2 -> 14x14
    bottleneck   t=2 c=128 n=6 s=1
    bottleneck   t=4 c=128 n=1 s=2 -> 7x7
    bottleneck   t=2 c=128 n=2 s=1
    conv1x1      c=512
    GDConv 7x7   (global depthwise, replaces avg-pool)
    linear       -> embedding_dim, then BatchNorm1d
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ConvBnAct, GDConv, InvertedResidual, init_weights

STAGES = [
    # (expansion, channels, repeats, stride)
    (2, 64, 5, 2),
    (4, 128, 1, 2),
    (2, 128, 6, 1),
    (4, 128, 1, 2),
    (2, 128, 2, 1),
]


def _round_ch(c: int, width: float, divisor: int = 8) -> int:
    c = c * width
    new = max(divisor, int(c + divisor / 2) // divisor * divisor)
    if new < 0.9 * c:  # never shrink a layer by more than 10% when rounding
        new += divisor
    return int(new)


class MobileFaceNet(nn.Module):
    def __init__(self, embedding_dim: int = 512, input_size: int = 112, width: float = 1.0,
                 dropout: float = 0.0, use_se: bool = False, act: str = "prelu",
                 fp16_backbone: bool = False):
        super().__init__()
        if input_size % 16 != 0:
            raise ValueError("input_size must be divisible by 16, got %d" % input_size)
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.fp16_backbone = fp16_backbone

        stem_ch = _round_ch(64, width)
        self.stem = ConvBnAct(3, stem_ch, 3, 2, 1, act=act)
        self.stem_dw = ConvBnAct(stem_ch, stem_ch, 3, 1, 1, groups=stem_ch, act=act)

        blocks: list[nn.Module] = []
        in_ch = stem_ch
        for t, c, n, s in STAGES:
            out_ch = _round_ch(c, width)
            for i in range(n):
                blocks.append(InvertedResidual(in_ch, out_ch, stride=s if i == 0 else 1,
                                               expansion=t, act=act, use_se=use_se))
                in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)

        last_ch = _round_ch(512, width)
        self.conv_last = ConvBnAct(in_ch, last_ch, 1, 1, 0, act=act)
        spatial = input_size // 16
        self.gdconv = GDConv(last_ch, spatial)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Conv2d(last_ch, embedding_dim, 1, 1, 0, bias=False)  # 1x1 conv == linear on 1x1 map
        self.bn = nn.BatchNorm1d(embedding_dim)

        init_weights(self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stem_dw(x)
        x = self.blocks(x)
        return self.conv_last(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp16_backbone:
            with torch.autocast(device_type=x.device.type, dtype=torch.float16):
                feat = self.forward_features(x)
            feat = feat.float()
        else:
            feat = self.forward_features(x)
        x = self.gdconv(feat)
        x = self.dropout(x)
        x = self.fc(x).flatten(1)
        return self.bn(x)
