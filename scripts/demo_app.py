"""Demo backend: upload photos -> embed -> group by person -> search by face.

Run from the repo root:  uvicorn scripts.demo_app:app --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sklearn.cluster import AgglomerativeClustering

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.data.dataset import IMG_EXTS
from facelib.pipeline.embedder import FaceEmbedder

CHECKPOINT = os.environ.get("EMBED_CHECKPOINT", "runs/mbf2/best.pt")
DETECTOR = os.environ.get("DETECTOR", "whole")  # "whole" = photos are already face crops
DATA_DIR = Path(os.environ.get("DEMO_DATA", "data/app"))
PHOTOS = DATA_DIR / "photos"
INDEX_NPZ = DATA_DIR / "index.npz"
INDEX_JSON = DATA_DIR / "index.json"
UI_FILE = Path(__file__).with_name("demo_ui.html")

PHOTOS.mkdir(parents=True, exist_ok=True)
embedder = FaceEmbedder(CHECKPOINT, detector=DETECTOR, flip_tta=True)
lock = threading.Lock()

emb = np.zeros((0, embedder.embedding_dim), dtype=np.float32)
names: list[str] = []  # names[i] = image filename that embedding row i came from


def load_index() -> None:
    global emb, names
    if INDEX_NPZ.exists() and INDEX_JSON.exists():
        emb = np.load(INDEX_NPZ)["emb"]
        names = json.loads(INDEX_JSON.read_text(encoding="utf-8"))


def save_index() -> None:
    np.savez(INDEX_NPZ, emb=emb)
    INDEX_JSON.write_text(json.dumps(names), encoding="utf-8")


def unique_name(filename: str) -> str:
    p = Path(filename or "photo.jpg")
    out, n = p.name, 0
    while (PHOTOS / out).exists():
        n += 1
        out = "%s_%d%s" % (p.stem, n, p.suffix)
    return out


load_index()
app = FastAPI(title="Face grouping + search demo")


@app.get("/")
def index():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {"ok": True, "photos": len(set(names)), "faces": len(names), "detector": DETECTOR}


@app.post("/upload")
def upload(files: list[UploadFile] = File(...)):
    global emb, names
    added, no_face, skipped = 0, [], []
    with lock:
        for f in files:
            if Path(f.filename or "").suffix.lower() not in IMG_EXTS:
                skipped.append(f.filename)
                continue
            name = unique_name(f.filename)
            dest = PHOTOS / name
            dest.write_bytes(f.file.read())
            try:
                faces = embedder.embed_path(dest)
            except Exception:
                faces = []
            if not faces:
                dest.unlink(missing_ok=True)
                no_face.append(f.filename)
                continue
            emb = np.vstack([emb, np.stack([x.embedding for x in faces]).astype(np.float32)])
            names.extend([name] * len(faces))
            added += 1
        save_index()
    return {"added": added, "no_face": no_face, "skipped": skipped,
            "total_photos": len(set(names))}


@app.get("/groups")
def groups(distance: float = 0.70):
    """Group photos of the same person. Lower distance = stricter grouping."""
    with lock:
        if len(names) == 0:
            return {"num_photos": 0, "groups": []}
        if len(names) == 1:
            labels = np.array([0])
        else:
            labels = AgglomerativeClustering(
                n_clusters=None, distance_threshold=distance,
                metric="cosine", linkage="average").fit_predict(emb)
        by_label: dict[int, list[str]] = {}
        for lab, name in zip(labels, names):
            members = by_label.setdefault(int(lab), [])
            if name not in members:
                members.append(name)
        ordered = sorted(by_label.values(), key=len, reverse=True)
        return {"num_photos": len(set(names)),
                "groups": [{"id": i, "photos": g} for i, g in enumerate(ordered)]}


@app.post("/search")
def search(file: UploadFile = File(...), threshold: float = 0.50):
    """Upload one face; returns the stored photos of the same person (no scores)."""
    data = np.frombuffer(file.file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "could not read the image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with lock:
        face = embedder.embed_largest_face(img)
        if face is None:
            raise HTTPException(422, "no face found in the query photo")
        if len(names) == 0:
            return {"found": False, "count": 0, "matches": []}
        sims = emb @ face.embedding
        best: dict[str, float] = {}
        for s, n in zip(sims, names):
            if s >= threshold and s > best.get(n, -1.0):
                best[n] = float(s)
        matches = sorted(best, key=best.get, reverse=True)
    return {"found": bool(matches), "count": len(matches), "matches": matches}


@app.post("/reset")
def reset():
    global emb, names
    with lock:
        for p in PHOTOS.iterdir():
            if p.is_file():
                p.unlink()
        emb = np.zeros((0, embedder.embedding_dim), dtype=np.float32)
        names = []
        save_index()
    return {"ok": True}


app.mount("/photos", StaticFiles(directory=str(PHOTOS)), name="photos")