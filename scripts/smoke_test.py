#!/usr/bin/env python
"""End-to-end smoke test on synthetic data - no dataset needed.

Generates fake "identities" (each a distinct colour/shape pattern plus noise),
trains for a couple of epochs, evaluates verification, runs the embedder on an
image with several pasted faces, and exercises the gallery. Use it to confirm the
whole pipeline is wired correctly on a fresh cloud box before touching real data.

  python scripts/smoke_test.py --out runs/smoke --identities 12 --per-identity 10
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.utils.config import Config


def synth_identity(rng: np.random.Generator, size: int = 112) -> np.ndarray:
    """A deterministic-per-identity pattern: base colour + two blobs + a bar."""
    img = np.zeros((size, size, 3), np.uint8)
    img[:] = rng.integers(40, 200, 3)
    for _ in range(2):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        centre = tuple(int(v) for v in rng.integers(size // 4, 3 * size // 4, 2))
        cv2.circle(img, centre, int(rng.integers(8, 22)), c, -1)
    cv2.rectangle(img, (int(rng.integers(0, size // 2)), int(rng.integers(0, size // 2))),
                  (int(rng.integers(size // 2, size)), int(rng.integers(size // 2, size))),
                  tuple(int(v) for v in rng.integers(0, 255, 3)), 3)
    return img


def make_dataset(root: Path, n_ids: int, per_id: int, size: int = 112) -> None:
    if root.exists():
        shutil.rmtree(root)
    for i in range(n_ids):
        rng = np.random.default_rng(1000 + i)
        base = synth_identity(rng, size)
        d = root / ("id%03d" % i)
        d.mkdir(parents=True, exist_ok=True)
        for j in range(per_id):
            jit = np.random.default_rng(i * 100 + j)
            img = base.astype(np.int16) + jit.integers(-25, 25, base.shape)
            img = np.clip(img, 0, 255).astype(np.uint8)
            if j % 3 == 0:
                img = cv2.GaussianBlur(img, (3, 3), 0.8)
            cv2.imwrite(str(d / ("%02d.jpg" % j)), img)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/smoke")
    ap.add_argument("--identities", type=int, default=12)
    ap.add_argument("--per-identity", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--onnx", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    data_root = out / "data"
    print("1/5 generating synthetic dataset ...")
    make_dataset(data_root, a.identities, a.per_identity)

    cfg = Config({
        "seed": 0,
        "model": {"name": "mobilefacenet", "embedding_dim": 128, "input_size": 112, "width": 0.5},
        "head": {"name": "arcface", "scale": 16.0, "margin_warmup_steps": 20},
        "data": {"root": str(data_root), "augment": "light", "min_images": 2, "num_workers": 0,
                 "holdout_frac": 0.25, "num_pairs": 200, "index_cache": str(out / "index.json")},
        "optim": {"name": "sgd", "lr": 0.05, "scale_lr": False, "warmup_steps": 10,
                  "schedule": "cosine", "weight_decay": 5e-4},
        "train": {"out_dir": str(out), "epochs": a.epochs, "batch_size": a.batch_size,
                  "amp": False, "ema": True, "log_every": 5, "eval_every_epochs": 1,
                  "eval_batch_size": 32},
    })

    print("2/5 training ...")
    from facelib.engine.train_embedding import train
    result = train(cfg)
    print("   ->", result)

    ckpt = out / "best.pt"
    if not ckpt.exists():
        ckpt = out / "last.pt"

    print("3/5 embedding a multi-face image ...")
    from facelib.pipeline.embedder import FaceEmbedder
    canvas = np.full((300, 500, 3), 220, np.uint8)
    faces_src = []
    for k, i in enumerate([0, 1, 2]):
        crop = cv2.imread(str(data_root / ("id%03d" % i) / "00.jpg"))
        canvas[40:152, 20 + k * 150:132 + k * 150] = crop
        faces_src.append("id%03d" % i)
    cv2.imwrite(str(out / "multi.jpg"), canvas)

    emb = FaceEmbedder(ckpt, detector="whole", device="cpu")
    single = emb.embed_path(data_root / "id000" / "00.jpg")
    assert len(single) == 1, "whole-image detector should return exactly one face"
    print("   embedding dim=%d, norm=%.4f" % (single[0].embedding.shape[0],
                                              float(np.linalg.norm(single[0].embedding))))

    print("4/5 gallery enroll + identify ...")
    from facelib.pipeline.gallery import Gallery
    g = Gallery(emb.embedding_dim, mode="centroid", threshold=0.3)
    for i in range(min(5, a.identities)):
        for j in range(3):
            f = emb.embed_path(data_root / ("id%03d" % i) / ("%02d.jpg" % j))
            if f:
                g.add("id%03d" % i, f[0].embedding)
    probe = emb.embed_path(data_root / "id000" / ("%02d.jpg" % (a.per_identity - 1)))
    match = g.identify(probe[0].embedding)
    print("   gallery=%s" % json.dumps(g.summary()))
    print("   probe id000 -> %s (%.3f)" % (match.label, match.score))
    g.save(out / "gallery.npz")
    assert len(Gallery.load(out / "gallery.npz")) == len(g)

    if a.onnx:
        print("5/5 exporting ONNX ...")
        from facelib.export.onnx_export import export_embedding
        print("   ->", json.dumps(export_embedding(ckpt, out / "embedding.onnx"), indent=2))
    else:
        print("5/5 skipped ONNX export (pass --onnx to include it)")

    print("\nSMOKE TEST PASSED. Synthetic data is not faces, so the accuracy number\n"
          "is only a wiring check - train on real aligned crops for real results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
