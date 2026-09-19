"""IResNet - the "improved residual" backbone used by ArcFace-style face models.

Differences from a torchvision ResNet, all of which matter for face embeddings:
  * BatchNorm *before* the first conv of each block (pre-activation ordering).
  * PReLU instead of ReLU (keeps a little gradient on negative activations;
    faces are low-contrast and the dead-ReLU problem costs accuracy).
  * Stride-1 stem with no max-pool, so a 112x112 crop only downsamples 16x
    and the final map is 7x7 - fine detail survives to the embedding.
  * No global average pooling: a flatten + linear "BN neck" keeps spatial
    position, which is meaningful because the crop is landmark-aligned.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .layers import ConvBnAct, EmbeddingNeck, SEModule, init_weights

DEPTHS: dict[int, list[int]] = {
    18: [2, 2, 2, 2],
    34: [3, 4, 6, 3],
    50: [3, 4, 14, 3],
    100: [3, 13, 30, 3],
    200: [6, 26, 60, 6],
}


class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None, use_se: bool = False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.conv1 = nn.Conv2d(in_ch, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.se = SEModule(planes) if use_se else nn.Identity()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, depth: int = 50, embedding_dim: int = 512, input_size: int = 112,
                 dropout: float = 0.0, width: float = 1.0, use_se: bool = False,
                 fp16_backbone: bool = False):
        super().__init__()
        if depth not in DEPTHS:
            raise ValueError("iresnet depth must be one of %s" % sorted(DEPTHS))
        if input_size % 16 != 0:
            raise ValueError("input_size must be divisible by 16, got %d" % input_size)
        layers = DEPTHS[depth]
        chans = [int(c * width) for c in (64, 64, 128, 256, 512)]
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.fp16_backbone = fp16_backbone

        self.stem = nn.Sequential(
            nn.Conv2d(3, chans[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chans[0]),
            nn.PReLU(chans[0]),
        )
        self.in_ch = chans[0]
        self.layer1 = self._make_layer(chans[1], layers[0], stride=2, use_se=use_se)
        self.layer2 = self._make_layer(chans[2], layers[1], stride=2, use_se=use_se)
        self.layer3 = self._make_layer(chans[3], layers[2], stride=2, use_se=use_se)
        self.layer4 = self._make_layer(chans[4], layers[3], stride=2, use_se=use_se)

        spatial = input_size // 16
        self.neck = EmbeddingNeck(chans[4], (spatial, spatial), embedding_dim, dropout)

        init_weights(self)
        # Zero-init the last BN of every block: each residual branch starts as a
        # no-op, which lets very deep configs (r100/r200) train without warmup tricks.
        for m in self.modules():
            if isinstance(m, IBasicBlock):
                nn.init.zeros_(m.bn3.weight)

    def _make_layer(self, planes: int, blocks: int, stride: int, use_se: bool) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_ch != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_ch, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        blocks_list = [IBasicBlock(self.in_ch, planes, stride, downsample, use_se)]
        self.in_ch = planes
        for _ in range(1, blocks):
            blocks_list.append(IBasicBlock(self.in_ch, planes, 1, None, use_se))
        return nn.Sequential(*blocks_list)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fp16_backbone:
            with torch.autocast(device_type=x.device.type, dtype=torch.float16):
                feat = self.forward_features(x)
            feat = feat.float()
        else:
            feat = self.forward_features(x)
        # The neck stays in fp32: the embedding feeds an angular margin, and fp16
        # rounding there visibly costs accuracy.
        return self.neck(feat)


def iresnet18(**kw) -> IResNet:
    return IResNet(depth=18, **kw)


def iresnet34(**kw) -> IResNet:
    return IResNet(depth=34, **kw)


def iresnet50(**kw) -> IResNet:
    return IResNet(depth=50, **kw)


def iresnet100(**kw) -> IResNet:
    return IResNet(depth=100, **kw)
