"""Building blocks shared by the embedding backbones and the detector."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_activation(name: str, channels: int | None = None) -> nn.Module:
    name = (name or "prelu").lower()
    if name == "prelu":
        return nn.PReLU(channels) if channels else nn.PReLU()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "relu6":
        return nn.ReLU6(inplace=True)
    if name == "leaky":
        return nn.LeakyReLU(0.1, inplace=True)
    if name == "hswish":
        return nn.Hardswish(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name in ("none", "identity"):
        return nn.Identity()
    raise ValueError("unknown activation %r" % (name,))


class ConvBnAct(nn.Sequential):
    """Conv -> BatchNorm -> activation. `groups=in_ch` makes it depthwise."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1,
                 padding: int | None = None, groups: int = 1, act: str = "prelu",
                 bn: bool = True):
        if padding is None:
            padding = kernel // 2
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=not bn)
        ]
        if bn:
            layers.append(nn.BatchNorm2d(out_ch))
        act_mod = make_activation(act, out_ch)
        if not isinstance(act_mod, nn.Identity):
            layers.append(act_mod)
        super().__init__(*layers)


class SEModule(nn.Module):
    """Squeeze-and-excitation channel gate."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=True)
        self.act = nn.PReLU(hidden)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, 1)
        w = self.fc2(self.act(self.fc1(w)))
        return x * torch.sigmoid(w)


class InvertedResidual(nn.Module):
    """MobileNetV2 bottleneck: 1x1 expand -> depthwise -> 1x1 project (linear).

    The residual connection only exists when the shapes match, which is what
    keeps the MobileFaceNet stage definitions readable.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, expansion: int = 2,
                 kernel: int = 3, act: str = "prelu", use_se: bool = False):
        super().__init__()
        hidden = int(round(in_ch * expansion))
        self.use_residual = stride == 1 and in_ch == out_ch
        layers: list[nn.Module] = []
        if hidden != in_ch:
            layers.append(ConvBnAct(in_ch, hidden, 1, 1, 0, act=act))
        layers.append(ConvBnAct(hidden, hidden, kernel, stride, kernel // 2, groups=hidden, act=act))
        if use_se:
            layers.append(SEModule(hidden))
        layers.append(ConvBnAct(hidden, out_ch, 1, 1, 0, act="none"))  # linear projection
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        return x + out if self.use_residual else out


class GDConv(nn.Module):
    """Global depthwise conv: replaces average pooling in face nets.

    Average pooling weights every spatial position equally; a face crop is
    aligned, so position carries information (eyes vs chin). A depthwise kernel
    the size of the feature map learns that weighting instead.
    """

    def __init__(self, channels: int, kernel: int | tuple[int, int] = 7):
        super().__init__()
        k = (kernel, kernel) if isinstance(kernel, int) else kernel
        self.conv = nn.Conv2d(channels, channels, k, 1, 0, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class Flatten(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


class EmbeddingNeck(nn.Module):
    """feature map -> BN -> dropout -> flatten -> linear -> BN1d (no activation).

    The trailing BatchNorm1d without affine bias is the standard ArcFace "BN neck":
    it keeps the embedding distribution centred so cosine similarity behaves.
    """

    def __init__(self, in_ch: int, spatial: tuple[int, int], embedding_dim: int = 512,
                 dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.dropout = nn.Dropout(p=dropout, inplace=False) if dropout > 0 else nn.Identity()
        self.flatten = Flatten()
        self.fc = nn.Linear(in_ch * spatial[0] * spatial[1], embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return self.bn2(x)


def l2_normalize(x: torch.Tensor, dim: int = 1, eps: float = 1e-10) -> torch.Tensor:
    return x / x.norm(p=2, dim=dim, keepdim=True).clamp_min(eps)


def init_weights(module: nn.Module) -> None:
    """Kaiming for convs, xavier for linears, BN to (1, 0)."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.PReLU):
            nn.init.constant_(m.weight, 0.25)
