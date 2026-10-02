"""Re-tune the /search threshold on the Haar-crop pipeline (leave-one-out on the demo set)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.data.dataset import IMG_EXTS
from facelib.pipeline.embedder import FaceEmbedder
from facelib.pipeline.haar_crop import HaarCropEmbedder

PHOTOS = Path("data/demo/photos")
MIN_REL_AREA = 0.5
THRESHOLDS = [round(t, 2) for t in np.arange(0.20, 0.86, 0.05)]

engine = HaarCropEmbedder(FaceEmbedder("runs/mbf2/best.pt", detector="whole", flip_tta=True))


def person(p: Path) -> str:
    return p.stem.rsplit("__", 1)[0]


rows, row_photo, row_person = [], [], []
queries = {}  # photo name -> largest-face embedding
for p in sorted(q for q in PHOTOS.iterdir() if q.suffix.lower() in IMG_EXTS):
    faces = engine.embed_path(p)
    if not faces:
        continue
    biggest = max(f.area for f in faces)
    keep = [f for f in faces if f.area >= MIN_REL_AREA * biggest]
    queries[p.name] = faces[0].embedding
    for f in keep:
        rows.append(f.embedding)
        row_photo.append(p.name)
        row_person.append(person(p))

emb = np.stack(rows)
row_photo = np.array(row_photo)
row_person = np.array(row_person)
print("indexed photos:", len(queries), "| faces:", len(rows))


def matches(q_emb, mask, thr):
    sims = emb[mask] @ q_emb
    return set(row_photo[mask][sims >= thr])


print("thr   hit-rate  precision  false-alarm")
for thr in THRESHOLDS:
    hits = n_hit_q = 0
    correct = returned = 0
    fa = n_fa_q = 0
    for name, q in queries.items():
        who = name.rsplit("__", 1)[0]
        not_self = row_photo != name
        same_in_db = ((row_person == who) & not_self).any()
        if same_in_db:
            m = matches(q, not_self, thr)
            n_hit_q += 1
            hits += any(x.rsplit("__", 1)[0] == who for x in m)
            correct += sum(x.rsplit("__", 1)[0] == who for x in m)
            returned += len(m)
        # false alarm: remove every photo of this person from the db
        mask = row_person != who
        n_fa_q += 1
        fa += len(matches(q, mask, thr)) > 0
    print("%.2f   %.3f     %.3f      %.3f" % (
        thr, hits / max(n_hit_q, 1), correct / max(returned, 1), fa / max(n_fa_q, 1)))