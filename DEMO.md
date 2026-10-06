# Demo: group photos by person + search by face

FastAPI backend (`scripts/demo_app.py`) with a single-page UI (`scripts/demo_ui.html`).

- **Upload** photos: each face is detected, embedded and stored.
- **Group**: photos of the same person are clustered together (average-linkage, cosine distance 0.70).
- **Search**: upload one person's photo; only the matching stored photos are returned (no scores).

## Run

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# place the trained checkpoint at runs\mbf2\best.pt (not in the repo: *.pt is gitignored)
uvicorn scripts.demo_app:app --port 8000
```

Open http://localhost:8000. Uploaded photos and the index persist in `data/app/`; `POST /reset` clears them.

## How it works

Uncropped photos: Haar cascade detects faces, each box is cropped with margin 1.7 (matches the framing of the LFW crops the model was trained on), then embedded (MobileFaceNet, flip-TTA). Faces smaller than half the largest face in a photo are dropped as false positives. Set `DETECTOR=whole` for photos that are already tight face crops.

| Env var | Default | Meaning |
| --- | --- | --- |
| `EMBED_CHECKPOINT` | `runs/mbf2/best.pt` | embedding checkpoint |
| `DETECTOR` | `haar_crop` | `haar_crop` or `whole` |
| `MIN_REL_AREA` | `0.5` | drop faces smaller than this fraction of the largest |
| `DEMO_DATA` | `data/app` | where photos and the index are stored |

## Results on the 100-photo demo set (28 held-out LFW identities)

- Haar found a face in 97 of 100 photos.
- Grouping (distance 0.70): precision ~0.6-0.7, F1 ~0.63-0.65.
- Search threshold 0.60 (default): 0 false alarms, precision 1.0, hit-rate ~0.21. It is tuned never to claim a match that does not exist; it misses many true matches.

## Scripts

| Script | Purpose |
| --- | --- |
| `scripts/make_demo_set.py` | build the labelled demo set from held-out LFW identities |
| `scripts/eval_demo_clusters.py` | sweep clustering thresholds against ground truth |
| `scripts/eval_demo_search.py` | search threshold sweep (whole-image path) |
| `scripts/eval_demo_search_haar.py` | search threshold sweep (Haar-crop path, the one the app uses) |
| `facelib/pipeline/haar_crop.py` | Haar detect -> margin crop -> embed |

## Limitations

- Embedding model trained on LFW only (~1.5k identities): usable for a demo, not production accuracy.
- Haar is frontal-only and weak on angles and lighting.
- Thresholds were tuned on 100 photos and must be re-tuned at real scale.
- Brute-force in-memory cosine search; swap for a vector DB (e.g. FAISS) as the index grows.
