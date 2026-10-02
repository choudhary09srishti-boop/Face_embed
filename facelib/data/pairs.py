"""Verification pairs: (path_a, path_b, same_person) triples, and the PyTorch dataset for them."""
from __future__ import annotations

import random
from pathlib import Path

from torch.utils.data import Dataset

from .dataset import IMG_EXTS, read_image_rgb


def _list_identities(root) -> dict[str, list[str]]:
    root = Path(root)
    out: dict[str, list[str]] = {}
    for ident_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in ident_dir.rglob("*") if f.suffix.lower() in IMG_EXTS)
        if len(files) >= 2:
            out[ident_dir.name] = [str(f.relative_to(root)) for f in files]
    return out


def split_identities(root, holdout_frac: float, seed: int = 42) -> tuple[list[str], list[str]]:
    """Split identity names into (train_ids, holdout_ids), disjoint, for honest eval."""
    names = sorted(_list_identities(root).keys())
    rng = random.Random(seed)
    rng.shuffle(names)
    n_hold = max(1, int(len(names) * holdout_frac)) if holdout_frac > 0 else 0
    hold = sorted(names[:n_hold])
    train = sorted(names[n_hold:])
    return train, hold


def make_pairs(root, num_pairs: int, seed: int = 42,
               identities: list[str] | None = None) -> list[tuple[str, str, int]]:
    """-> list of (rel_path_a, rel_path_b, label) with label=1 same person, 0 different.
    Roughly half positive, half negative. Paths are relative to `root`."""
    index = _list_identities(root)
    if identities is not None:
        index = {k: v for k, v in index.items() if k in identities}
    names = sorted(index.keys())
    if len(names) < 2:
        raise ValueError("need at least 2 identities with >=2 images to build pairs")

    rng = random.Random(seed)
    pairs: list[tuple[str, str, int]] = []
    n_pos = num_pairs // 2
    n_neg = num_pairs - n_pos

    for _ in range(n_pos):
        name = rng.choice(names)
        files = index[name]
        a, b = rng.sample(files, 2) if len(files) >= 2 else (files[0], files[0])
        pairs.append((a, b, 1))

    for _ in range(n_neg):
        na, nb = rng.sample(names, 2)
        a = rng.choice(index[na])
        b = rng.choice(index[nb])
        pairs.append((a, b, 0))

    rng.shuffle(pairs)
    return pairs


def write_pairs(pairs: list[tuple[str, str, int]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a, b, y in pairs:
            f.write("%s\t%s\t%d\n" % (a, b, y))


def read_pairs(path) -> list[tuple[str, str, int]]:
    out = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            a, b, y = line.split("\t")
            out.append((a, b, int(y)))
    return out


class PairsDataset(Dataset):
    """Yields (image_a, image_b, label) for verification scoring."""

    def __init__(self, pairs: list[tuple[str, str, int]], root, transform):
        self.pairs = pairs
        self.root = Path(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        a, b, y = self.pairs[i]
        img_a = read_image_rgb(self.root / a)
        img_b = read_image_rgb(self.root / b)
        if img_a is None or img_b is None:
            return self.__getitem__((i + 1) % len(self.pairs))
        return self.transform(img_a), self.transform(img_b), y