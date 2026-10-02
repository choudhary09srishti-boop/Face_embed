"""Find the crop_bbox margin on Haar boxes whose embedding best matches the 'whole' path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.align.warp import crop_bbox
from facelib.data.dataset import IMG_EXTS, read_image_rgb
from facelib.detect.bootstrap import build_bootstrap_detector
from facelib.pipeline.embedder import FaceEmbedder

PHOTOS = Path("data/demo/photos")
MARGINS = [0.25, 0.8, 1.2, 1.5, 1.7, 1.9, 2.2]

embedder = FaceEmbedder("runs/mbf2/best.pt", detector="whole", flip_tta=True)
haar = build_bootstrap_detector("haar")
size = embedder.align_size

files = sorted(p for p in PHOTOS.iterdir() if p.suffix.lower() in IMG_EXTS)
sims = {m: [] for m in MARGINS}
for p in files:
    img = read_image_rgb(p)
    ref = embedder.embed_image(img)[0].embedding
    r = haar.detect(img)
    if len(r) == 0:
        continue
    areas = (r.boxes[:, 2] - r.boxes[:, 0]) * (r.boxes[:, 3] - r.boxes[:, 1])
    box = r.boxes[int(np.argmax(areas))]
    crops = [crop_bbox(img, box, size, m) for m in MARGINS]
    embs = embedder.embed_crops(crops)
    for m, e in zip(MARGINS, embs):
        sims[m].append(float(ref @ e))

print("margin  median_cos  mean_cos  n")
for m in MARGINS:
    v = np.array(sims[m])
    print("%5.2f   %.3f       %.3f     %d" % (m, np.median(v), v.mean(), len(v)))