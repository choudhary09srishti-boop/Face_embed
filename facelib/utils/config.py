"""Tiny config system: YAML -> attribute-accessible dict, with CLI overrides."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access, recursively applied to nested dicts."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in list(self.items()):
            self[k] = self._wrap(v)

    @staticmethod
    def _wrap(v: Any) -> Any:
        if isinstance(v, dict):
            return Config(v)
        if isinstance(v, (list, tuple)):
            return [Config._wrap(x) for x in v]
        return v

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = self._wrap(value)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, Config) else x for x in v]
            else:
                out[k] = v
        return out

    def dump(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """Load a YAML config. A `_base_` key (path relative to this file) is merged first.

    `overrides` are `dotted.key=value` strings; the value is parsed as YAML, so
    `optim.lr=0.05`, `train.epochs=30`, `train.amp=false`, `strides=[8,16,32]`
    all behave as expected.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    base = raw.pop("_base_", None)
    if base:
        merged = load_config(path.parent / base).to_dict()
        raw = _deep_update(merged, raw)
    cfg = Config(raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError("override must look like key=value, got %r" % (item,))
        key, value = item.split("=", 1)
        _set_dotted(cfg, key.strip(), yaml.safe_load(value))
    return cfg


def _deep_update(dst: dict, src: dict) -> dict:
    dst = copy.deepcopy(dst)
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _set_dotted(cfg: Config, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for p in parts[:-1]:
        if p not in node or not isinstance(node[p], dict):
            node[p] = Config()
        node = node[p]
    node[parts[-1]] = Config._wrap(value)
