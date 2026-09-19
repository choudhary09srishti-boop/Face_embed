#!/usr/bin/env python
"""Evaluate a checkpoint on a pair list: accuracy, AUC, EER, TAR@FAR, thresholds."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.engine.evaluate import evaluate_pairs_file, format_metrics
from facelib.nn.builder import load_embedding_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-flip", action="store_true")
    a = ap.parse_args()
    model, meta = load_embedding_model(a.checkpoint, a.device)
    m = evaluate_pairs_file(model, a.pairs, a.root, a.device, int(meta["input_size"]),
                            a.batch_size, a.workers, flip_tta=not a.no_flip)
    print(format_metrics(m))
    print(json.dumps(m, indent=2))
