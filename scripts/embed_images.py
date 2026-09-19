#!/usr/bin/env python
"""Batch embedding CLI: images in, embeddings + manifest out (multi-face aware).

  python scripts/embed_images.py --checkpoint runs/mbf/best.pt --images photos/ \
      --detector runs/detector/last.pt --out out/embeddings

Writes <out>.npy (N x D float32), <out>.csv (one row per face) and <out>.meta.json.
Add --jsonl for a streaming JSON file with embeddings inline.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.data.dataset import IMG_EXTS
from facelib.pipeline.embedder import FaceEmbedder


def collect(images: str) -> list[Path]:
    p = Path(images)
    if p.is_dir():
        return [f for f in sorted(p.rglob("*")) if f.suffix.lower() in IMG_EXTS]
    if p.suffix.lower() == ".txt":
        return [Path(l.strip()) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    return [p]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--images", required=True, help="directory, .txt list, or single file")
    ap.add_argument("--out", default="out/embeddings")
    ap.add_argument("--detector", default="haar", help="whole | haar | path/to/detector.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--max-faces", type=int, default=0, help="0 = all faces")
    ap.add_argument("--flip-tta", action="store_true")
    ap.add_argument("--jsonl", action="store_true")
    ap.add_argument("--parquet", action="store_true")
    a = ap.parse_args()

    files = collect(a.images)
    if not files:
        print("no images found at %s" % a.images)
        return 2

    emb = FaceEmbedder(a.checkpoint, detector=a.detector, device=a.device,
                       flip_tta=a.flip_tta, min_det_score=a.min_score,
                       max_faces=a.max_faces or None, batch_size=a.batch_size)
    print("pipeline: %s" % json.dumps(emb.describe()))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    vectors: list[np.ndarray] = []
    rows: list[dict] = []
    jf = open(out.with_suffix(".jsonl"), "w", encoding="utf-8") if a.jsonl else None

    for n, (path, faces) in enumerate(emb.embed_paths(files)):
        for k, f in enumerate(faces):
            vectors.append(f.embedding)
            rows.append({
                "row": len(vectors) - 1, "image": str(path), "face": k,
                "x1": round(float(f.bbox[0]), 1), "y1": round(float(f.bbox[1]), 1),
                "x2": round(float(f.bbox[2]), 1), "y2": round(float(f.bbox[3]), 1),
                "det_score": round(float(f.det_score), 4),
                "align_mode": f.align_mode, "align_error": round(float(f.align_error), 3),
            })
            if jf:
                jf.write(json.dumps({"image": str(path), "face": k, **f.to_dict()}) + "\n")
        if (n + 1) % 200 == 0:
            print("%d/%d images | %d faces" % (n + 1, len(files), len(vectors)))
    if jf:
        jf.close()

    matrix = (np.stack(vectors) if vectors
              else np.zeros((0, emb.embedding_dim), dtype=np.float32))
    np.save(out.with_suffix(".npy"), matrix)
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["row", "image"])
        w.writeheader()
        w.writerows(rows)
    meta = {"images": len(files), "faces": int(matrix.shape[0]),
            "embedding_dim": int(matrix.shape[1]) if matrix.size else emb.embedding_dim,
            "pipeline": emb.describe()}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if a.parquet and rows:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.table({**{k: [r[k] for r in rows] for k in rows[0]},
                              "embedding": [v.tolist() for v in vectors]})
            pq.write_table(table, out.with_suffix(".parquet"))
        except ImportError:
            print("pyarrow not installed; skipped parquet")

    print("\n%d faces from %d images -> %s.npy / .csv" % (matrix.shape[0], len(files), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
