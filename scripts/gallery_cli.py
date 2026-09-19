#!/usr/bin/env python
"""Identity gallery: enroll people, then identify faces in new images.

  python scripts/gallery_cli.py enroll  --checkpoint runs/mbf/best.pt \
      --images data/enroll --out weights/gallery.npz --detector haar
  python scripts/gallery_cli.py identify --checkpoint runs/mbf/best.pt \
      --gallery weights/gallery.npz --image group.jpg --draw out/annotated.jpg

`enroll` expects data/enroll/<person-name>/*.jpg.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.align.warp import draw_faces
from facelib.data.dataset import IMG_EXTS, read_image_rgb
from facelib.pipeline.embedder import FaceEmbedder
from facelib.pipeline.gallery import Gallery


def make_embedder(a) -> FaceEmbedder:
    return FaceEmbedder(a.checkpoint, detector=a.detector, device=a.device,
                        flip_tta=a.flip_tta, min_det_score=a.min_score)


def cmd_enroll(a) -> int:
    emb = make_embedder(a)
    g = Gallery(emb.embedding_dim, mode=a.mode, threshold=a.threshold,
                metadata={"checkpoint": str(a.checkpoint)})
    root = Path(a.images)
    people = sorted(p for p in root.iterdir() if p.is_dir())
    if not people:
        print("expected %s/<person>/*.jpg" % root)
        return 2
    for person in people:
        files = [f for f in sorted(person.rglob("*")) if f.suffix.lower() in IMG_EXTS]
        added = 0
        for f in files:
            faces = emb.embed_path(f)
            if not faces:
                continue
            g.add(person.name, faces[0].embedding)  # largest face in an enrolment photo
            added += 1
        print("%-24s %d/%d enrolled" % (person.name, added, len(files)))
    g.save(a.out)
    print(json.dumps(g.summary(), indent=2))
    return 0


def cmd_identify(a) -> int:
    emb = make_embedder(a)
    g = Gallery.load(a.gallery)
    if a.threshold is not None:
        g.threshold = a.threshold
    img = read_image_rgb(a.image)
    if img is None:
        print("cannot read %s" % a.image)
        return 2
    faces = emb.embed_image(img)
    results = []
    for i, f in enumerate(faces):
        hits = g.search(f.embedding, top_k=a.top_k)
        best = hits[0] if hits else None
        label = best.label if best and best.is_known else "unknown"
        results.append({"face": i, "bbox": [round(float(v), 1) for v in f.bbox],
                        "det_score": round(float(f.det_score), 3), "label": label,
                        "matches": [h.to_dict() for h in hits]})
        print("face %d %s -> %s (%.3f)" % (i, results[-1]["bbox"], label,
                                           best.score if best else -1))
    print(json.dumps({"gallery": g.summary(), "faces": results}, indent=2))
    if a.draw:
        labels = ["%s %.2f" % (r["label"], r["matches"][0]["score"] if r["matches"] else -1)
                  for r in results]
        lmks = ([f.landmarks for f in faces]
                if all(f.landmarks is not None for f in faces) else None)
        vis = draw_faces(img, [f.bbox for f in faces], lmks, labels)
        Path(a.draw).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(a.draw, vis[:, :, ::-1])
        print("annotated -> %s" % a.draw)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("enroll", "identify"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--detector", default="haar")
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--min-score", type=float, default=0.6)
        p.add_argument("--flip-tta", action="store_true")
        if name == "enroll":
            p.add_argument("--images", required=True)
            p.add_argument("--out", default="weights/gallery.npz")
            p.add_argument("--mode", default="centroid", choices=["centroid", "nearest"])
            p.add_argument("--threshold", type=float, default=0.35)
        else:
            p.add_argument("--gallery", required=True)
            p.add_argument("--image", required=True)
            p.add_argument("--threshold", type=float, default=None)
            p.add_argument("--top-k", type=int, default=3)
            p.add_argument("--draw", default=None)
    a = ap.parse_args()
    return cmd_enroll(a) if a.cmd == "enroll" else cmd_identify(a)


if __name__ == "__main__":
    raise SystemExit(main())
