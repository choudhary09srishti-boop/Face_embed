"""Rank-aware logging plus a jsonl metric log that survives notebook restarts."""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

_FMT = "%(asctime)s | %(levelname).1s | %(name)s | %(message)s"


def setup_logging(out_dir: str | Path | None = None, rank: int = 0, level: int = logging.INFO) -> logging.Logger:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(level if rank == 0 else logging.WARNING)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(_FMT, datefmt="%H:%M:%S"))
    root.addHandler(sh)
    if out_dir is not None and rank == 0:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(out_dir / "train.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT))
        root.addHandler(fh)
    return root


class JsonlLogger:
    """Append-only metric log: one JSON object per line, easy to plot later."""

    def __init__(self, path: str | Path, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **kw: Any) -> None:
        if not self.enabled:
            return
        kw.setdefault("wall", time.time())
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(kw, default=float) + "\n")
