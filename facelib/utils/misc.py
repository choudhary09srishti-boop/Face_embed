"""Seeding, meters, timing, checkpoint IO, EMA and distributed helpers."""
from __future__ import annotations

import copy
import math
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


# ---------------------------------------------------------------- determinism
def seed_everything(seed: int, rank: int = 0, deterministic: bool = False) -> None:
    """Seed every RNG. `rank` is folded in so ranks do not draw identical crops."""
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------- distributed
def init_distributed() -> tuple[int, int, int]:
    """Read torchrun's env vars and init the process group. -> (rank, world, local_rank)."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
    return rank, world, local


def is_main(rank: int = 0) -> bool:
    return rank == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


# -------------------------------------------------------------------- meters
class AverageMeter:
    """Running mean, plus a short-window mean for readable progress lines."""

    def __init__(self, window: int = 50):
        self.window = window
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0
        self.recent: deque[float] = deque(maxlen=max(self.window, 1))

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n
        self.recent.append(float(value))

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)

    @property
    def smooth(self) -> float:
        return sum(self.recent) / max(len(self.recent), 1)


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self) -> float:
        now = time.time()
        dt = now - self.t0
        self.t0 = now
        return dt

    def elapsed(self) -> float:
        return time.time() - self.t0


def fmt_eta(seconds: float) -> str:
    seconds = int(max(seconds, 0))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, s)


# --------------------------------------------------------------- checkpoints
def save_checkpoint(path: str | Path, **objects: Any) -> None:
    """Write to a temp file then rename, so an interrupted save cannot corrupt the last good one."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(objects, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Strip a DistributedDataParallel / DataParallel wrapper."""
    return getattr(model, "module", model)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ModelEMA:
    """Exponential moving average of weights; typically +0.1-0.3% verification accuracy."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999, warmup: int = 2000):
        self.module = copy.deepcopy(unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        # Ramp the decay in, otherwise the random init dominates the average for a long time.
        d = self.decay * (1.0 - math.exp(-self.updates / max(self.warmup, 1)))
        msd = unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            src = msd[k].detach()
            if not v.dtype.is_floating_point:
                v.copy_(src)
                continue
            v.mul_(d).add_(src.to(v.dtype), alpha=1.0 - d)
