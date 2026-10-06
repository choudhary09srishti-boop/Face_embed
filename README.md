# Face Grouping and Search

Upload a pile of photos and the app groups them by person, like Google Photos "People". Upload one person's photo and it returns only the stored photos of that person, or nothing if they are not in the records.

Built for a records-style use case: given a large photo database, check whether a person is already in it. The search is tuned to avoid false matches.

## Features

- **Upload** many photos at once; faces are detected, embedded and stored.
- **Group by person** with average-linkage clustering on cosine distance (adjustable in the UI).
- **Search by face** returns matching photos only, with no scores.
- **Works on uncropped photos** via Haar detection and a margin crop matched to the training data.
- **Persistent index** in `data/app/`, reloaded on startup.

## How it works

```
photo -> Haar face detection -> crop with margin 1.7 -> MobileFaceNet (512-d, L2-normalised)
      -> cosine similarity -> group (agglomerative clustering) / search (threshold 0.60)
```

The embedding model is trained from scratch (MobileFaceNet backbone, ArcFace loss) on LFW; the full training, detector and evaluation library is documented in [docs/PIPELINE.md](docs/PIPELINE.md). The margin of 1.7 was chosen by measurement: it gave the closest match (median cosine 0.88) to the whole-image embeddings the model was trained on, which is why alignment must be the same at training and inference.

## Results (100 photos, 28 held-out identities the model never saw)

| Metric | Result |
| --- | --- |
| Verification accuracy (training holdout) | 88.3% |
| Haar found a face | 97 / 100 photos |
| Grouping (distance 0.70) | precision ~0.6-0.7, F1 ~0.63-0.65 |
| Search at threshold 0.60 | 0 false alarms, precision 1.0, hit-rate ~0.21 |

The search threshold was chosen from a sweep so that a person who is not in the database is never reported as found. The trade-off is that many true matches are missed.

## Tech stack

Python 3.10, PyTorch, OpenCV, NumPy, scikit-learn, FastAPI, vanilla HTML/CSS/JS.

## Run it

```powershell
git clone https://github.com/choudhary09srishti-boop/Face_embed.git
cd Face_embed
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# place the trained checkpoint at runs\mbf2\best.pt (model files are not stored in the repo)
uvicorn scripts.demo_app:app --port 8000
```

Open http://localhost:8000. Details and configuration are in [DEMO.md](DEMO.md).

## Project layout

| Path | What |
| --- | --- |
| `scripts/demo_app.py`, `scripts/demo_ui.html` | FastAPI backend and single-page UI |
| `facelib/pipeline/haar_crop.py` | Haar detect, margin crop, embed |
| `facelib/data/` | dataset, pairs, sampler and transforms for training |
| `scripts/eval_demo_*.py` | clustering and search threshold sweeps against ground truth |
| `scripts/make_demo_set.py` | builds the labelled demo set from held-out LFW identities |

## Limitations and next steps

- The model is trained on LFW only (about 1.5k identities): fine for a demo, not production accuracy.
- Haar is frontal-only; a trained detector would handle angles and lighting better.
- Thresholds were tuned on 100 photos and should be re-tuned at real scale.
- Search is brute-force cosine similarity in memory; swap in a vector database (for example FAISS) for large collections.
- The UI is functional but plain.

## Credits

Embedding library (`facelib`: backbones, detector, alignment, training) by Rajeev ([RAJEEV2510/face_embed](https://github.com/RAJEEV2510/face_embed)). Demo application, data module, Haar-crop pipeline and evaluation by Srishti Choudhary. Data: Labeled Faces in the Wild.
