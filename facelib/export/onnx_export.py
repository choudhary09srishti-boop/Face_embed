"""Export the embedding model to ONNX (input NCHW [-1,1] RGB, output L2-normalised)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from ..nn.builder import load_embedding_model

log = logging.getLogger(__name__)


def export_embedding(checkpoint: str | Path, output: str | Path, opset: int = 17,
                     dynamic_batch: bool = True, verify: bool = True) -> dict:
    model, meta = load_embedding_model(checkpoint, device="cpu", normalize=True)
    size = int(meta["input_size"])
    dummy = torch.randn(1, 3, size, size)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model, dummy, str(output),
        input_names=["input"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch"}, "embedding": {0: "batch"}} if dynamic_batch else None,
        opset_version=opset, do_constant_folding=True,
    )
    info = {"onnx": str(output), "input_size": size, "embedding_dim": int(meta["embedding_dim"]),
            "normalization": "(rgb_uint8 - 127.5) / 128.0", "layout": "NCHW"}

    if verify:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            ref = model(dummy).detach().numpy()
            got = sess.run(["embedding"], {"input": dummy.numpy()})[0]
            info["max_abs_diff"] = float(np.abs(ref - got).max())
            log.info("onnx verified, max |torch - onnx| = %.2e", info["max_abs_diff"])
        except ImportError:
            log.warning("onnxruntime not installed; skipping verification")
    return info
