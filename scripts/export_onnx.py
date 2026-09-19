#!/usr/bin/env python
"""Export a trained embedding checkpoint to ONNX."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.export.onnx_export import export_embedding

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="weights/embedding.onnx")
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    print(json.dumps(export_embedding(a.checkpoint, a.out, a.opset), indent=2))
