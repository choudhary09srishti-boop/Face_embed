#!/usr/bin/env python
"""FastAPI service: POST an image, get one embedding per detected face.

  EMBED_CHECKPOINT=runs/mbf/best.pt DETECTOR=runs/detector/last.pt \
    uvicorn scripts.serve:app --host 0.0.0.0 --port 8000

  curl -F file=@group.jpg localhost:8000/embed
  curl -F file=@group.jpg localhost:8000/identify

Env: EMBED_CHECKPOINT (required), DETECTOR (whole|haar|path), GALLERY (npz),
DEVICE, MIN_SCORE, FLIP_TTA, MAX_FACES.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.pipeline.embedder import FaceEmbedder
from facelib.pipeline.gallery import Gallery

CKPT = os.environ.get("EMBED_CHECKPOINT", "runs/mbf/best.pt")
DETECTOR = os.environ.get("DETECTOR", "haar")
GALLERY_PATH = os.environ.get("GALLERY", "")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="face-embed", version="1.0")
_state: dict = {}


@app.on_event("startup")
def _load() -> None:
    _state["embedder"] = FaceEmbedder(
        CKPT, detector=DETECTOR, device=DEVICE,
        flip_tta=os.environ.get("FLIP_TTA", "0") == "1",
        min_det_score=float(os.environ.get("MIN_SCORE", 0.6)),
        max_faces=int(os.environ.get("MAX_FACES", 0)) or None,
    )
    _state["gallery"] = Gallery.load(GALLERY_PATH) if GALLERY_PATH and Path(GALLERY_PATH).exists() else None


def _read(data: bytes) -> np.ndarray:
    try:
        return np.array(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception as exc:
        raise HTTPException(400, "not a decodable image: %s" % exc)


@app.get("/health")
def health() -> dict:
    emb: FaceEmbedder = _state["embedder"]
    g: Gallery | None = _state.get("gallery")
    return {"status": "ok", "pipeline": emb.describe(),
            "gallery": g.summary() if g else None}


@app.post("/embed")
async def embed(file: UploadFile = File(...)) -> dict:
    emb: FaceEmbedder = _state["embedder"]
    faces = emb.embed_image(_read(await file.read()))
    return {"num_faces": len(faces), "embedding_dim": emb.embedding_dim,
            "faces": [f.to_dict() for f in faces]}


@app.post("/identify")
async def identify(file: UploadFile = File(...), top_k: int = 3) -> dict:
    emb: FaceEmbedder = _state["embedder"]
    g: Gallery | None = _state.get("gallery")
    if g is None:
        raise HTTPException(503, "no gallery loaded; set GALLERY=path/to/gallery.npz")
    faces = emb.embed_image(_read(await file.read()))
    out = []
    for i, f in enumerate(faces):
        hits = g.search(f.embedding, top_k=top_k)
        out.append({"face": i, **f.to_dict(include_embedding=False),
                    "label": hits[0].label if hits and hits[0].is_known else "unknown",
                    "matches": [h.to_dict() for h in hits]})
    return {"num_faces": len(faces), "faces": out}


@app.post("/compare")
async def compare(file_a: UploadFile = File(...), file_b: UploadFile = File(...)) -> dict:
    """Cosine similarity between the largest face in each image."""
    emb: FaceEmbedder = _state["embedder"]
    fa = emb.embed_largest_face(_read(await file_a.read()))
    fb = emb.embed_largest_face(_read(await file_b.read()))
    if fa is None or fb is None:
        raise HTTPException(422, "no face detected in one of the images")
    score = float(np.dot(fa.embedding, fb.embedding))
    return {"cosine": round(score, 4)}
