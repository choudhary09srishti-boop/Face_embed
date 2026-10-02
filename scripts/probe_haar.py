"""Probe Haar boxes on LFW demo photos to estimate the margin LFW framing implies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.data.dataset import IMG_EXTS, read_image_rgb
from facelib.detect.bootstrap import build_bootstrap_detector

PHOTOS = Path("data/demo/photos")
det = build_bootstrap_detector("haar")

files = sorted(p for p in PHOTOS.iterdir() if p.suffix.lower() in IMG_EXTS)
ratios_w, ratios_h, cx_off, cy_off = [], [], [], []
found = 0
for p in files:
    img = read_image_rgb(p)
    h, w = img.shape[:2]
    r = det.detect(img)
    if len(r) == 0:
        continue
    found += 1
    b = r.boxes[np.argmax((r.boxes[:, 2] - r.boxes[:, 0]) * (r.boxes[:, 3] - r.boxes[:, 1]))]
    ratios_w.append((b[2] - b[0]) / w)
    ratios_h.append((b[3] - b[1]) / h)
    cx_off.append(((b[0] + b[2]) / 2 - w / 2) / w)
    cy_off.append(((b[1] + b[3]) / 2 - h / 2) / h)

print("photos:", len(files), "| haar found face:", found)
if found:
    print("box w / image w  median: %.3f" % np.median(ratios_w))
    print("box h / image h  median: %.3f" % np.median(ratios_h))
    print("center x offset  median: %+.3f" % np.median(cx_off))
    print("center y offset  median: %+.3f" % np.median(cy_off))

    