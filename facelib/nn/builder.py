"""Model factory + checkpoint loading for the embedding network."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .iresnet import IResNet
from .layers import l2_normalize
from .mobilefacenet import MobileFaceNet

log = logging.getLogger(__name__)


def build_embedding_model(cfg: dict | Any) -> nn.Module:
    """cfg keys: name, embedding_dim, input_size, dropout, width, use_se, fp16_backbone.

    `name` is one of: mobilefacenet, iresnet18/34/50/100/200 (or r18/r34/... for short).
    """
    name = str(cfg["name"]).lower().replace("-", "").replace("_", "")
    kw = dict(
        embedding_dim=int(cfg.get("embedding_dim", 512)),
        input_size=int(cfg.get("input_size", 112)),
        dropout=float(cfg.get("dropout", 0.0)),
        width=float(cfg.get("width", 1.0)),
        use_se=bool(cfg.get("use_se", False)),
        fp16_backbone=bool(cfg.get("fp16_backbone", False)),
    )
    if name in ("mobilefacenet", "mbf"):
        kw["act"] = str(cfg.get("act", "prelu"))
        return MobileFaceNet(**kw)
    if name.startswith("iresnet") or (name.startswith("r") and name[1:].isdigit()):
        depth = int(name.replace("iresnet", "").replace("r", ""))
        return IResNet(depth=depth, **kw)
    raise ValueError("unknown backbone %r" % (cfg["name"],))


class NormalizedEmbedding(nn.Module):
    """Backbone + L2 normalisation, i.e. exactly what inference and ONNX should expose.

    Downstream code compares embeddings with a dot product, so normalising inside
    the graph removes one way for a caller to get it wrong.
    """

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = getattr(backbone, "embedding_dim", None)
        self.input_size = getattr(backbone, "input_size", 112)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return l2_normalize(self.backbone(x).float(), dim=1)


def strip_prefix(state: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state):
        return state
    return {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state.items()}


def load_embedding_model(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
    prefer_ema: bool = True,
    normalize: bool = True,
) -> tuple[nn.Module, dict]:
    """Rebuild the backbone from the architecture block stored in the checkpoint.

    Returns (model_in_eval_mode, metadata). Metadata carries `model` (the arch
    config), `classes` when the training set labels were saved, `step`/`epoch`
    and the best verification score, so an embedding dump can always be traced
    back to the exact model that produced it.
    """
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    if "model_cfg" not in ckpt:
        raise KeyError(
            "checkpoint %s has no 'model_cfg' - it was not written by this project" % checkpoint
        )
    model = build_embedding_model(ckpt["model_cfg"])
    state = None
    if prefer_ema and ckpt.get("ema") is not None:
        state = ckpt["ema"]
        log.info("loading EMA weights")
    if state is None:
        state = ckpt["backbone"]
    state = strip_prefix(strip_prefix(state, "module."), "backbone.")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        log.warning("state_dict mismatch | missing=%s unexpected=%s",
                    list(missing)[:6], list(unexpected)[:6])
    if normalize:
        model = NormalizedEmbedding(model)
    model.eval().to(device)
    meta = {
        "model_cfg": ckpt["model_cfg"],
        "classes": ckpt.get("classes"),
        "epoch": ckpt.get("epoch"),
        "step": ckpt.get("step"),
        "best_metric": ckpt.get("best_metric"),
        "embedding_dim": ckpt["model_cfg"].get("embedding_dim", 512),
        "input_size": ckpt["model_cfg"].get("input_size", 112),
    }
    return model, meta


def save_classes_json(path: str | Path, classes: list[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"classes": classes}, f, indent=2)
