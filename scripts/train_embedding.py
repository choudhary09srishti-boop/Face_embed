#!/usr/bin/env python
"""Train the embedding model.

  python scripts/train_embedding.py --config configs/embedding_mbf.yaml \
      data.root=data/aligned train.epochs=30

Multi-GPU:  torchrun --nproc_per_node=4 scripts/train_embedding.py --config ...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.engine.train_embedding import train
from facelib.utils.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*", help="dotted.key=value")
    a = ap.parse_args()
    print(train(load_config(a.config, a.overrides)))
