#!/usr/bin/env python
"""Train the TinyFace detector on WIDER FACE.

  python scripts/train_detector.py --config configs/detector.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.engine.train_detector import train
from facelib.utils.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args()
    print(train(load_config(a.config, a.overrides)))
