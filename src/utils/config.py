from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml


def set_path(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    *parents, leaf = dotted.split(".")
    node = cfg
    for key in parents:
        node = node.setdefault(key, {})
    node[leaf] = value


def read_yaml(path: Path) -> dict[str, Any]:
    """Read one YAML file, resolving its optional base file first."""
    cfg = yaml.safe_load(path.read_text()) or {}
    base = cfg.pop("base", None)
    if base is None:
        return cfg
    merged = read_yaml(path.parent / base)
    merge(merged, cfg)
    return merged


def load_config(paths: Path | list[Path], overrides: list[str] | None = None) -> dict[str, Any]:
    """Merge YAML files left to right, then apply key.sub=value overrides."""
    cfg: dict[str, Any] = {}
    for path in [paths] if isinstance(paths, Path) else paths:
        merge(cfg, read_yaml(path))
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got {item}")
        set_path(cfg, key, yaml.safe_load(raw))
    return cfg


def merge(base: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value


def build[T](cls: type[T], section: dict[str, Any] | None) -> T:
    """Instantiate a config dataclass, rejecting unknown keys and turning lists into tuples."""
    section = section or {}
    types = {f.name: f.type for f in fields(cls)}
    unknown = section.keys() - types.keys()
    if unknown:
        raise KeyError(f"Unknown {cls.__name__} keys: {sorted(unknown)}")
    values = {}
    for key, value in section.items():
        if isinstance(value, list):
            value = tuple(value)
        elif types[key] is float and isinstance(value, str | int):
            # YAML 1.1 reads exponents without a decimal point, such as 1e-3, as strings.
            value = float(value)
        values[key] = value
    return cls(**values)
