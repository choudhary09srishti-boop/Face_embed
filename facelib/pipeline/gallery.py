"""Identity gallery: enroll labelled embeddings, then recognise new faces.

The embedding model is never retrained to add a person. Enrolment is just
"store this person's normalised embeddings", and recognition is a cosine
similarity search - which is exactly why an embedding model is worth training
instead of a plain classifier: new identities cost one forward pass.

Two matching modes:
  * centroid - average the embeddings of each identity (robust, fast, one vector
    per person; best when you have several good photos each);
  * nearest  - keep every enrolled embedding and take the best single match
    (better for identities photographed under very different conditions).

The threshold is the whole ballgame for open-set recognition: below it, a face is
reported as unknown. Calibrate it on your own data with `scripts/evaluate.py`,
which reports the cosine threshold at a chosen false-accept rate.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-12)


@dataclass
class Match:
    label: str
    score: float           # cosine similarity in [-1, 1]
    is_known: bool         # score >= threshold

    def to_dict(self) -> dict:
        return {"label": self.label, "score": round(float(self.score), 4), "is_known": self.is_known}


class Gallery:
    def __init__(self, embedding_dim: int, mode: str = "centroid", threshold: float = 0.35,
                 metadata: dict | None = None):
        if mode not in ("centroid", "nearest"):
            raise ValueError("mode must be 'centroid' or 'nearest'")
        self.embedding_dim = int(embedding_dim)
        self.mode = mode
        self.threshold = float(threshold)
        self.metadata = metadata or {}
        self._vectors: list[np.ndarray] = []
        self._labels: list[str] = []
        self._matrix: np.ndarray | None = None      # (K, D) search matrix
        self._matrix_labels: list[str] = []

    # ------------------------------------------------------------------ enrol
    def add(self, label: str, embedding: np.ndarray) -> None:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if emb.shape[0] != self.embedding_dim:
            raise ValueError("expected %d-d embedding, got %d" % (self.embedding_dim, emb.shape[0]))
        self._vectors.append(l2_normalize(emb))
        self._labels.append(str(label))
        self._matrix = None  # invalidate the cached search matrix

    def add_batch(self, label: str, embeddings: np.ndarray) -> None:
        """Enroll several embeddings of the same identity."""
        for emb in np.asarray(embeddings, dtype=np.float32).reshape(-1, self.embedding_dim):
            self.add(label, emb)

    def remove(self, label: str) -> int:
        """Delete every embedding of one identity. Returns how many were removed."""
        keep = [i for i, l in enumerate(self._labels) if l != label]
        removed = len(self._labels) - len(keep)
        self._vectors = [self._vectors[i] for i in keep]
        self._labels = [self._labels[i] for i in keep]
        self._matrix = None
        return removed

    # ----------------------------------------------------------------- search
    def _build(self) -> None:
        if not self._vectors:
            self._matrix = np.zeros((0, self.embedding_dim), dtype=np.float32)
            self._matrix_labels = []
            return
        if self.mode == "nearest":
            self._matrix = np.stack(self._vectors, axis=0)
            self._matrix_labels = list(self._labels)
        else:
            groups: dict[str, list[np.ndarray]] = defaultdict(list)
            for lbl, vec in zip(self._labels, self._vectors):
                groups[lbl].append(vec)
            labels = sorted(groups)
            # Mean of normalised vectors, renormalised: the direction that maximises
            # the sum of cosine similarities to that identity's samples.
            self._matrix = np.stack([l2_normalize(np.mean(groups[l], axis=0)) for l in labels])
            self._matrix_labels = labels

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._build()
        return self._matrix

    @property
    def labels(self) -> list[str]:
        if self._matrix is None:
            self._build()
        return self._matrix_labels

    def search(self, embedding: np.ndarray, top_k: int = 5) -> list[Match]:
        """Top-k matches for one embedding, best first."""
        if self.matrix.shape[0] == 0:
            return []
        q = l2_normalize(np.asarray(embedding, dtype=np.float32).reshape(-1))
        scores = self.matrix @ q
        if self.mode == "nearest":
            # Collapse duplicate labels, keeping each identity's best score.
            best: dict[str, float] = {}
            for lbl, sc in zip(self.labels, scores):
                if sc > best.get(lbl, -2.0):
                    best[lbl] = float(sc)
            ranked = sorted(best.items(), key=lambda kv: -kv[1])[:top_k]
            return [Match(l, s, s >= self.threshold) for l, s in ranked]
        order = np.argsort(-scores)[:top_k]
        return [Match(self.labels[i], float(scores[i]), float(scores[i]) >= self.threshold)
                for i in order]

    def identify(self, embedding: np.ndarray) -> Match:
        """Best match, or the sentinel `unknown` when nothing clears the threshold."""
        hits = self.search(embedding, top_k=1)
        if not hits:
            return Match("unknown", -1.0, False)
        top = hits[0]
        return top if top.is_known else Match("unknown", top.score, False)

    def search_many(self, embeddings: np.ndarray, top_k: int = 5) -> list[list[Match]]:
        return [self.search(e, top_k) for e in np.asarray(embeddings).reshape(-1, self.embedding_dim)]

    # ------------------------------------------------------------------- stats
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for l in self._labels:
            out[l] += 1
        return dict(sorted(out.items()))

    def __len__(self) -> int:
        return len(self._labels)

    def summary(self) -> dict:
        c = self.counts()
        return {
            "identities": len(c),
            "embeddings": len(self._labels),
            "mode": self.mode,
            "threshold": self.threshold,
            "embedding_dim": self.embedding_dim,
            "min_per_identity": min(c.values()) if c else 0,
            "max_per_identity": max(c.values()) if c else 0,
        }

    # -------------------------------------------------------------------- IO
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            vectors=(np.stack(self._vectors) if self._vectors
                     else np.zeros((0, self.embedding_dim), np.float32)),
            labels=np.asarray(self._labels, dtype=object),
            config=json.dumps({
                "embedding_dim": self.embedding_dim,
                "mode": self.mode,
                "threshold": self.threshold,
                "metadata": self.metadata,
            }),
        )
        log.info("gallery saved: %s (%d identities, %d embeddings)",
                 path, len(self.counts()), len(self._labels))

    @classmethod
    def load(cls, path: str | Path) -> "Gallery":
        data = np.load(str(path), allow_pickle=True)
        cfg = json.loads(str(data["config"]))
        g = cls(cfg["embedding_dim"], cfg["mode"], cfg["threshold"], cfg.get("metadata"))
        vectors = data["vectors"]
        labels = [str(l) for l in data["labels"]]
        g._vectors = [np.asarray(v, dtype=np.float32) for v in vectors]
        g._labels = labels
        return g
