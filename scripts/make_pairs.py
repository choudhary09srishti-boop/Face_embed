#!/usr/bin/env python
"""Build a verification pair list from an aligned dataset."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.data.pairs import make_pairs, split_identities, write_pairs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data/pairs.txt")
    ap.add_argument("--num-pairs", type=int, default=6000)
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help=">0 restricts pairs to a held-out identity split")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ids = None
    if a.holdout_frac > 0:
        _, ids = split_identities(a.root, a.holdout_frac, a.seed)
        print("holdout identities: %d" % len(ids))
    pairs = make_pairs(a.root, a.num_pairs, a.seed, identities=ids)
    write_pairs(pairs, a.out)
    print("%d pairs -> %s" % (len(pairs), a.out))
