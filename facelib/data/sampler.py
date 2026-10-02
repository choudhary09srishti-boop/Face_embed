"""P x K batch sampler: each batch has P identities x K images per identity.

Needed for margin-loss training when batches are small relative to the number
of classes - it guarantees positive pairs exist in every batch instead of
hoping random shuffling produces them.
"""
from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler


class PKSampler(Sampler):
    def __init__(self, targets: list[int], batch_size: int, images_per_identity: int,
                 world_size: int = 1, rank: int = 0, seed: int = 42):
        self.targets = targets
        self.k = max(images_per_identity, 1)
        self.p = max(batch_size // self.k, 1)
        self.batch_size = self.p * self.k
        self.world_size = world_size
        self.rank = rank
        self.seed = seed
        self.epoch = 0

        self.by_class: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(targets):
            self.by_class[label].append(idx)
        self.classes = list(self.by_class.keys())

        n_batches = len(targets) // self.batch_size
        self.num_samples = max(n_batches // world_size, 1) * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        classes = self.classes[:]
        rng.shuffle(classes)

        batches: list[list[int]] = []
        pool = {c: self.by_class[c][:] for c in classes}
        for c in pool:
            rng.shuffle(pool[c])

        available = [c for c in classes if len(pool[c]) > 0]
        while len(available) >= self.p:
            chosen = rng.sample(available, self.p)
            batch = []
            for c in chosen:
                picks = pool[c][:self.k]
                if len(picks) < self.k:  # sample with replacement to fill K
                    picks = picks + [rng.choice(self.by_class[c]) for _ in range(self.k - len(picks))]
                pool[c] = pool[c][self.k:]
                batch.extend(picks)
            batches.append(batch)
            available = [c for c in classes if len(pool[c]) > 0]

        rng.shuffle(batches)
        flat = [idx for batch in batches for idx in batch]
        flat = flat[self.rank::self.world_size] if self.world_size > 1 else flat
        return iter(flat[:self.num_samples])

    def __len__(self) -> int:
        return self.num_samples