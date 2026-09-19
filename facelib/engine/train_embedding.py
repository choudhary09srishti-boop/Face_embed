"""Embedding trainer: AMP + DDP + cosine LR/warmup + EMA + periodic verification eval."""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from ..data.dataset import FaceFolderDataset
from ..data.pairs import make_pairs, read_pairs, split_identities, write_pairs
from ..data.sampler import PKSampler
from ..data.transforms import build_eval_transform, build_train_transform
from ..nn.builder import build_embedding_model, save_classes_json
from ..nn.losses import build_criterion, embedding_stats, topk_accuracy
from ..nn.margins import build_head
from ..utils.logging import JsonlLogger, setup_logging
from ..utils.misc import (AverageMeter, ModelEMA, Timer, all_reduce_mean, barrier,
                          cleanup_distributed, count_params, fmt_eta, init_distributed,
                          load_checkpoint, save_checkpoint, seed_everything, unwrap)

log = logging.getLogger(__name__)


def lr_at(step: int, total: int, base_lr: float, warmup: int, min_lr_ratio: float,
          schedule: str = "cosine") -> float:
    if step < warmup:  # linear warmup: margin heads diverge without it
        return base_lr * (step + 1) / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    p = min(max(p, 0.0), 1.0)
    if schedule == "cosine":
        f = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    elif schedule == "poly":
        f = min_lr_ratio + (1 - min_lr_ratio) * (1 - p) ** 2
    elif schedule == "step":
        f = 0.1 ** sum(p >= m for m in (0.5, 0.75, 0.9))
    else:
        f = 1.0
    return base_lr * f


def build_optimizer(backbone: nn.Module, head: nn.Module, cfg) -> torch.optim.Optimizer:
    # No weight decay on norm/bias params, and never on the margin head's class centres.
    decay, no_decay = [], []
    for m in (backbone,):
        for name, p in m.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim <= 1 else decay).append(p)
    head_params = [p for p in head.parameters() if p.requires_grad]
    groups = [
        {"params": decay, "weight_decay": float(cfg.get("weight_decay", 5e-4))},
        {"params": no_decay, "weight_decay": 0.0},
        {"params": head_params, "weight_decay": float(cfg.get("head_weight_decay", 5e-4))},
    ]
    name = str(cfg.get("name", "sgd")).lower()
    lr = float(cfg.get("lr", 0.1))
    if name == "sgd":
        return torch.optim.SGD(groups, lr=lr, momentum=float(cfg.get("momentum", 0.9)),
                               nesterov=bool(cfg.get("nesterov", True)))
    if name == "adamw":
        return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999))
    raise ValueError("unknown optimizer %r" % name)


def _loaders(cfg, rank: int, world: int):
    data = cfg.data
    size = int(cfg.model.get("input_size", 112))
    root = Path(data.root)
    out_dir = Path(cfg.train.out_dir)

    # Identity-disjoint holdout so the verification score means something.
    pairs_file = data.get("pairs_file")
    holdout = float(data.get("holdout_frac", 0.1))
    train_ids = None
    if pairs_file and Path(pairs_file).exists():
        pairs = read_pairs(pairs_file)
    elif holdout > 0:
        train_ids, hold_ids = split_identities(root, holdout, int(cfg.get("seed", 42)))
        pairs = make_pairs(root, int(data.get("num_pairs", 6000)),
                           int(cfg.get("seed", 42)), identities=hold_ids)
        if rank == 0:
            write_pairs(pairs, out_dir / "val_pairs.txt")
            (out_dir / "holdout_identities.json").write_text(json.dumps(hold_ids), encoding="utf-8")
    else:
        pairs = make_pairs(root, int(data.get("num_pairs", 2000)), int(cfg.get("seed", 42)))

    train_set = FaceFolderDataset(
        root,
        transform=build_train_transform((size, size), str(data.get("augment", "medium"))),
        min_images=int(data.get("min_images", 1)),
        max_images=data.get("max_images"),
        index_cache=data.get("index_cache"),
        class_to_idx=({n: i for i, n in enumerate(sorted(train_ids))} if train_ids else None),
    )
    bs = int(cfg.train.batch_size)
    nw = int(data.get("num_workers", 8))
    if bool(data.get("pk_sampler", False)):
        sampler = PKSampler(train_set.targets, bs, int(data.get("images_per_identity", 4)),
                            world, rank, int(cfg.get("seed", 42)))
    elif world > 1:
        sampler = DistributedSampler(train_set, world, rank, shuffle=True, drop_last=True)
    else:
        sampler = None
    loader = DataLoader(train_set, batch_size=bs, sampler=sampler, shuffle=sampler is None,
                        num_workers=nw, pin_memory=True, drop_last=True,
                        persistent_workers=nw > 0, prefetch_factor=4 if nw > 0 else None)
    return train_set, loader, sampler, pairs


def train(cfg) -> dict:
    rank, world, local = init_distributed()
    device = torch.device("cuda:%d" % local if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.train.out_dir)
    setup_logging(out_dir, rank)
    seed_everything(int(cfg.get("seed", 42)), rank)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        cfg.dump(out_dir / "config.yaml")
    jsonl = JsonlLogger(out_dir / "metrics.jsonl", rank == 0)

    train_set, loader, sampler, val_pairs = _loaders(cfg, rank, world)
    if rank == 0:
        train_set.save_classes(out_dir / "classes.json")

    model_cfg = cfg.model.to_dict()
    backbone = build_embedding_model(model_cfg).to(device)
    head = build_head(str(cfg.head.name), int(model_cfg.get("embedding_dim", 512)),
                      train_set.num_classes,
                      scale=float(cfg.head.get("scale", 64.0)),
                      sub_centers=int(cfg.head.get("sub_centers", 1)),
                      interclass_filtering=float(cfg.head.get("interclass_filtering", 0.0)),
                      margin_warmup_steps=int(cfg.head.get("margin_warmup_steps", 0))).to(device)
    criterion = build_criterion(str(cfg.train.get("criterion", "ce")),
                                float(cfg.train.get("label_smoothing", 0.0)))
    optimizer = build_optimizer(backbone, head, cfg.optim)
    amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(backbone, float(cfg.train.get("ema_decay", 0.9999))) if cfg.train.get("ema", True) else None

    epochs = int(cfg.train.epochs)
    spe = len(loader)
    total_steps = epochs * spe
    warmup = int(cfg.optim.get("warmup_steps", min(1000, total_steps // 20)))
    base_lr = float(cfg.optim.get("lr", 0.1)) * (world if cfg.optim.get("scale_lr", True) else 1)

    start_epoch, gstep, best = 0, 0, -1.0
    resume = cfg.train.get("resume")
    if resume and Path(resume).exists():
        ck = load_checkpoint(resume, "cpu")
        unwrap(backbone).load_state_dict(ck["backbone"])
        head.load_state_dict(ck["head"])
        optimizer.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        if ema is not None and ck.get("ema"):
            ema.module.load_state_dict(ck["ema"])
        start_epoch, gstep, best = ck.get("epoch", 0) + 1, ck.get("step", 0), ck.get("best_metric", -1.0)
        log.info("resumed from %s at epoch %d", resume, start_epoch)

    if world > 1:
        backbone = nn.parallel.DistributedDataParallel(backbone, device_ids=[local] if device.type == "cuda" else None)
        head = nn.parallel.DistributedDataParallel(head, device_ids=[local] if device.type == "cuda" else None)

    if rank == 0:
        log.info("backbone=%s params=%.2fM | %d ids | %d imgs | %d steps/epoch | bs=%d x %d gpu",
                 model_cfg.get("name"), count_params(unwrap(backbone)) / 1e6,
                 train_set.num_classes, len(train_set), spe, int(cfg.train.batch_size), world)

    grad_clip = float(cfg.train.get("grad_clip", 5.0))
    log_every = int(cfg.train.get("log_every", 50))
    eval_every = int(cfg.train.get("eval_every_epochs", 1))
    timer = Timer()

    for epoch in range(start_epoch, epochs):
        backbone.train()
        head.train()
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        loss_m, acc_m = AverageMeter(), AverageMeter()

        for i, (images, labels) in enumerate(loader):
            lr = lr_at(gstep, total_steps, base_lr, warmup,
                       float(cfg.optim.get("min_lr_ratio", 0.001)),
                       str(cfg.optim.get("schedule", "cosine")))
            for g in optimizer.param_groups:
                g["lr"] = lr

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                emb = backbone(images)
            emb = emb.float()  # margin + loss always in fp32
            logits = head(emb, labels)
            loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap(backbone).parameters(), grad_clip)
                torch.nn.utils.clip_grad_norm_(unwrap(head).parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(backbone)

            loss_m.update(loss.item())
            acc_m.update(topk_accuracy(logits.detach(), labels)[0])
            gstep += 1

            if rank == 0 and (i % log_every == 0 or i == spe - 1):
                done = (epoch - start_epoch) * spe + i + 1
                eta = timer.elapsed() / max(done, 1) * (total_steps - (epoch * spe + i + 1))
                log.info("e%03d %5d/%d | loss %.4f | acc %5.2f%% | lr %.5f | eta %s",
                         epoch, i, spe, loss_m.smooth, acc_m.smooth, lr, fmt_eta(eta))
                jsonl.log(kind="train", epoch=epoch, step=gstep, loss=loss_m.smooth,
                          acc=acc_m.smooth, lr=lr)

        if rank == 0 and epoch % 5 == 0:
            log.info("embedding stats: %s", embedding_stats(emb.detach()[:64]))

        metric = float("nan")
        if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
            metric = _validate(cfg, ema, backbone, val_pairs, device, rank, epoch, jsonl)
        # Only rank 0 evaluates and checkpoints, so no reduction is needed here.
        metric_sync = metric if metric == metric else -1.0

        if rank == 0:
            payload = dict(
                backbone=unwrap(backbone).state_dict(), head=unwrap(head).state_dict(),
                optimizer=optimizer.state_dict(), scaler=scaler.state_dict() if amp else None,
                ema=ema.module.state_dict() if ema is not None else None,
                model_cfg=model_cfg, classes=train_set.classes, epoch=epoch, step=gstep,
                best_metric=max(best, metric_sync),
            )
            save_checkpoint(out_dir / "last.pt", **payload)
            if metric_sync > best:
                best = metric_sync
                save_checkpoint(out_dir / "best.pt", **payload)
                log.info("new best verification accuracy: %.4f", best)
        barrier()

    if rank == 0:
        save_classes_json(out_dir / "classes.json", train_set.classes)
        log.info("done in %s | best %.4f | weights: %s", fmt_eta(timer.elapsed()), best, out_dir)
    cleanup_distributed()
    return {"best_metric": best, "out_dir": str(out_dir)}


def _validate(cfg, ema, backbone, pairs, device, rank, epoch, jsonl) -> float:
    if rank != 0 or not pairs:
        return float("nan")
    from ..engine.evaluate import format_metrics, score_pairs, verification_metrics
    from ..nn.builder import NormalizedEmbedding
    net = NormalizedEmbedding(ema.module if ema is not None else unwrap(backbone)).to(device).eval()
    scores, labels = score_pairs(net, pairs, cfg.data.root, device,
                                 int(cfg.model.get("input_size", 112)),
                                 int(cfg.train.get("eval_batch_size", 64)),
                                 int(cfg.data.get("num_workers", 4)), flip_tta=True)
    m = verification_metrics(scores, labels)
    log.info("[eval e%03d] %s", epoch, format_metrics(m))
    jsonl.log(kind="eval", epoch=epoch, **{k: v for k, v in m.items() if isinstance(v, (int, float))})
    return float(m["accuracy"])
