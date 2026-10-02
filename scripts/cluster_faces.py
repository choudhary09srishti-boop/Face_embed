#!/usr/bin/env python
"""Cluster face embeddings into groups of the same person (DBSCAN, cosine distance)."""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True, help="path to embeddings.npy")
    ap.add_argument("--csv", required=True, help="path to embeddings.csv (maps row -> image path)")
    ap.add_argument("--out", required=True, help="output folder for clustered copies")
    ap.add_argument("--eps", type=float, default=0.35,
                    help="max cosine distance within a cluster; lower = stricter grouping")
    ap.add_argument("--min-samples", type=int, default=2,
                    help="min faces to form a cluster; smaller groups become 'noise'")
    args = ap.parse_args()

    emb = np.load(args.embeddings)
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    rows = list(csv.DictReader(open(args.csv, newline="", encoding="utf-8")))
    paths = [r.get("image") or r.get("path") or r.get("file") for r in rows]

    db = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric="cosine")
    labels = db.fit_predict(emb)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print("found %d clusters, %d unclustered faces (out of %d)" % (n_clusters, n_noise, len(labels)))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for label, path in zip(labels, paths):
        if path is None:
            continue
        folder = out / ("person_%03d" % label if label != -1 else "unclustered")
        folder.mkdir(parents=True, exist_ok=True)
        src = Path(path)
        if src.exists():
            shutil.copy(src, folder / src.name)

    print("clustered photos copied -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())