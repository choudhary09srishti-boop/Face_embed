"""TinyFace detector trainer (single GPU or torchrun DDP)."""
from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from ..detect.anchors import (DEFAULT_MIN_SIZES, DEFAULT_STRIDES, DEFAULT_VARIANCES,
                              generate_anchors)
from ..detect.data import WiderFaceDataset, detection_collate
from ..detect.loss import MultiTaskDetectionLoss
from ..detect.model import build_detector
from ..utils.logging import JsonlLogger, setup_logging
from ..utils.misc import (AverageMeter, ModelEMA, Timer, barrier, cleanup_distributed,
                          count_params, fmt_eta, init_distributed, load_checkpoint,
                          save_checkpoint, seed_everything, unwrap)
from .train_embedding import lr_at

log = logging.getLogger(__name__)


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

    img_size = int(cfg.data.get("img_size", 640))
    ds = WiderFaceDataset(cfg.data.label_file, cfg.data.images_root, img_size, train=True)
    sampler = DistributedSampler(ds, world, rank, shuffle=True, drop_last=True) if world > 1 else None
    nw = int(cfg.data.get("num_workers", 8))
    loader = DataLoader(ds, batch_size=int(cfg.train.batch_size), sampler=sampler,
                        shuffle=sampler is None, num_workers=nw, pin_memory=True,
                        drop_last=True, collate_fn=detection_collate,
                        persistent_workers=nw > 0)

    model_cfg = cfg.model.to_dict()
    model_cfg.setdefault("strides", DEFAULT_STRIDES)
    model_cfg.setdefault("min_sizes", DEFAULT_MIN_SIZES)
    model_cfg.setdefault("variances", DEFAULT_VARIANCES)
    model = build_detector(model_cfg).to(device)

    # The input is a fixed square during training, so the anchors are built once.
    anchors = generate_anchors((img_size, img_size), model_cfg["strides"],
                               model_cfg["min_sizes"], device=device)
    criterion = MultiTaskDetectionLoss(
        iou_threshold=float(cfg.loss.get("iou_threshold", 0.35)),
        neg_pos_ratio=int(cfg.loss.get("neg_pos_ratio", 7)),
        variances=model_cfg["variances"],
        loc_weight=float(cfg.loss.get("loc_weight", 2.0)),
        landmark_weight=float(cfg.loss.get("landmark_weight", 1.0)),
        use_focal=bool(cfg.loss.get("use_focal", False)),
    )

    base_lr = float(cfg.optim.get("lr", 1e-2)) * (world if cfg.optim.get("scale_lr", True) else 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                momentum=float(cfg.optim.get("momentum", 0.9)),
                                weight_decay=float(cfg.optim.get("weight_decay", 5e-4)),
                                nesterov=True)
    amp = bool(cfg.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema = ModelEMA(model, float(cfg.train.get("ema_decay", 0.9998))) if cfg.train.get("ema", True) else None

    epochs = int(cfg.train.epochs)
    spe = len(loader)
    total = epochs * spe
    warmup = int(cfg.optim.get("warmup_steps", min(1500, total // 20)))

    start_epoch, gstep = 0, 0
    resume = cfg.train.get("resume")
    if resume and Path(resume).exists():
        ck = load_checkpoint(resume, "cpu")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if ema is not None and ck.get("ema"):
            ema.module.load_state_dict(ck["ema"])
        start_epoch, gstep = ck.get("epoch", 0) + 1, ck.get("step", 0)

    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local] if device.type == "cuda" else None)

    if rank == 0:
        log.info("detector params=%.2fM | %d images | %d anchors | %d steps/epoch",
                 count_params(unwrap(model)) / 1e6, len(ds), anchors.shape[0], spe)

    timer = Timer()
    log_every = int(cfg.train.get("log_every", 50))
    for epoch in range(start_epoch, epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        meters = {k: AverageMeter() for k in ("loss", "loss_box", "loss_cls", "loss_lmk")}

        for i, (images, targets) in enumerate(loader):
            lr = lr_at(gstep, total, base_lr, warmup,
                       float(cfg.optim.get("min_lr_ratio", 0.01)),
                       str(cfg.optim.get("schedule", "cosine")))
            for g in optimizer.param_groups:
                g["lr"] = lr
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=amp):
                preds = model(images)
            preds = tuple(p.float() for p in preds)
            loss, stats = criterion(preds, anchors, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(),
                                           float(cfg.train.get("grad_clip", 10.0)))
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            for k in meters:
                meters[k].update(stats[k])
            gstep += 1

            if rank == 0 and (i % log_every == 0 or i == spe - 1):
                done = (epoch - start_epoch) * spe + i + 1
                eta = timer.elapsed() / max(done, 1) * (total - (epoch * spe + i + 1))
                log.info("e%03d %5d/%d | loss %.3f (box %.3f cls %.3f lmk %.3f) | pos %.0f | lr %.5f | eta %s",
                         epoch, i, spe, meters["loss"].smooth, meters["loss_box"].smooth,
                         meters["loss_cls"].smooth, meters["loss_lmk"].smooth,
                         stats["num_pos"], lr, fmt_eta(eta))
                jsonl.log(kind="train", epoch=epoch, step=gstep, lr=lr,
                          **{k: m.smooth for k, m in meters.items()})

        if rank == 0:
            save_checkpoint(out_dir / "last.pt", model=unwrap(model).state_dict(),
                            optimizer=optimizer.state_dict(),
                            ema=ema.module.state_dict() if ema is not None else None,
                            model_cfg=model_cfg, epoch=epoch, step=gstep)
        barrier()

    if rank == 0:
        log.info("detector trained in %s -> %s", fmt_eta(timer.elapsed()), out_dir / "last.pt")
    cleanup_distributed()
    return {"out_dir": str(out_dir)}
