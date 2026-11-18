"""Configuraciones YAML con herencia (`extends`) y overrides por línea de comandos.

    python -m training.train --config configs/sac_curriculum.yaml --set algo_kwargs.batch_size=256 --set stages.0.timesteps=100000
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def deep_merge(base: dict, new: dict) -> dict:
    """Combina `new` sobre `base`. Los diccionarios se fusionan; listas y escalares se reemplazan."""
    out = copy.deepcopy(base)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        data = deep_merge(_load_yaml((path.parent / parent).resolve()), data)
    return data


def _parse_value(text: str):
    # PyYAML lee "3e-4" como texto; lo convertimos a número cuando se pueda.
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return yaml.safe_load(text)


def apply_override(cfg: dict, assignment: str) -> None:
    """Aplica 'a.b.0.c=valor' sobre cfg (los índices numéricos recorren listas)."""
    if "=" not in assignment:
        raise ValueError(f"Override inválido (use clave=valor): {assignment!r}")
    key, raw = assignment.split("=", 1)
    parts = key.strip().split(".")
    node = cfg
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
    last = parts[-1]
    value = _parse_value(raw.strip())
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def load_config(path: str | Path, overrides: list[str] | tuple[str, ...] = ()) -> dict:
    path = Path(path).resolve()
    cfg = _load_yaml(path)
    for item in overrides:
        apply_override(cfg, item)
    cfg.setdefault("name", path.stem)
    if cfg["name"] is None:
        cfg["name"] = path.stem
    return cfg
