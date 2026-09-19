#!/usr/bin/env python
"""Detect + align raw photos into the training layout.

    raw/<identity>/*.jpg   ->   aligned/<identity>/*.jpg   (112x112, landmark-aligned)

Detector options: whole (images are already crops) | haar (opencv bootstrap) |
path to a trained TinyFace checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.align.warp import align_face, alignment_error, crop_bbox
from facelib.data.dataset import IMG_EXTS, read_image_rgb
from facelib.detect.bootstrap import (align_by_eyes, build_bootstrap_detector,
                                      has_eye_landmarks, has_full_landmarks)


def build_detector(spec: str, min_score: float, device: str):
    if spec.lower() in ("whole", "haar", "none"):
        return build_bootstrap_detector(spec.lower())
    from facelib.detect.infer import FaceDetector
    return FaceDetector(spec, device=device, score_threshold=min_score)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw/<identity>/*.jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detector", default="haar", help="whole | haar | path/to/detector.pt")
    ap.add_argument("--size", type=int, default=112)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-score", type=float, default=0.6)
    ap.add_argument("--max-faces-per-image", type=int, default=1,
                    help="1 keeps only the largest face (right for identity folders)")
    ap.add_argument("--min-face-px", type=int, default=32)
    ap.add_argument("--max-align-error", type=float, default=-1.0)
    ap.add_argument("--limit-per-identity", type=int, default=0)
    args = ap.parse_args()

    raw, out = Path(args.raw), Path(args.out)
    det = build_detector(args.detector, args.min_score, args.device)
    identities = sorted(p for p in raw.iterdir() if p.is_dir())
    if not identities:
        print("no identity subdirectories under %s" % raw)
        return 2

    stats = {"images": 0, "written": 0, "no_face": 0, "too_small": 0, "bad_align": 0,
             "unreadable": 0, "modes": {"landmarks": 0, "eyes": 0, "bbox": 0}}

    for ident in identities:
        files = [f for f in sorted(ident.rglob("*")) if f.suffix.lower() in IMG_EXTS]
        if args.limit_per_identity:
            files = files[:args.limit_per_identity]
        dst_dir = out / ident.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for f in files:
            stats["images"] += 1
            img = read_image_rgb(f)
            if img is None:
                stats["unreadable"] += 1
                continue
            res = det.detect(img)
            if len(res) == 0:
                stats["no_face"] += 1
                continue
            order = np.argsort(-((res.boxes[:, 2] - res.boxes[:, 0]) *
                                 (res.boxes[:, 3] - res.boxes[:, 1])))
            for n, i in enumerate(order[:max(args.max_faces_per_image, 1)]):
                box, lmk = res.boxes[i], res.landmarks[i]
                if min(box[2] - box[0], box[3] - box[1]) < args.min_face_px:
                    stats["too_small"] += 1
                    continue
                if has_full_landmarks(lmk):
                    err = alignment_error(lmk, args.size)
                    if args.max_align_error > 0 and err > args.max_align_error:
                        stats["bad_align"] += 1
                        continue
                    crop, mode = align_face(img, lmk, args.size), "landmarks"
                elif has_eye_landmarks(lmk):
                    crop, mode = align_by_eyes(img, lmk, args.size), "eyes"
                else:
                    crop, mode = crop_bbox(img, box, args.size), "bbox"
                stats["modes"][mode] += 1
                suffix = "" if n == 0 else "_%d" % n
                dst = dst_dir / ("%s%s.jpg" % (f.stem, suffix))
                cv2.imwrite(str(dst), crop[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                stats["written"] += 1
                kept += 1
        print("%-28s %4d/%4d kept" % (ident.name, kept, len(files)))
        if kept == 0:
            dst_dir.rmdir()

    out.mkdir(parents=True, exist_ok=True)
    (out / "prepare_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n%s" % json.dumps(stats, indent=2))
    print("aligned dataset -> %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
