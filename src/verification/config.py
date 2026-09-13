from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from src.world.types import Rollout


@dataclass(frozen=True)
class VerificationConfig:
    occworld_root: Path
    upstream_config: Path
    expected_revision: str
    checkpoint: Path | None
    data_root: Path | None
    infos: Path | None
    sample_index: int
    seed: int
    device: str
    output: Path
    rollout: Rollout

    def resolved(self) -> dict[str, Any]:
        return {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(self).items()
        }

    def validate_assets(self) -> None:
        for name in ("checkpoint", "data_root", "infos"):
            path = getattr(self, name)
            if path is None:
                raise ValueError(f"Provide {name} in the config or via --{name.replace('_', '-')}")
            exists = path.is_dir() if name == "data_root" else path.is_file()
            if not exists:
                raise FileNotFoundError(f"Missing {name}: {path}")
        if not (self.data_root / "gts").is_dir():
            raise FileNotFoundError(f"Occ3D gts directory is missing under {self.data_root}")


def load_config(path: Path, overrides: dict[str, Any], project_root: Path) -> VerificationConfig:
    with path.open() as stream:
        values = yaml.safe_load(stream)
    if not isinstance(values, dict):
        raise ValueError("Verification config must be a mapping")
    unknown = values.keys() - {field.name for field in fields(VerificationConfig)}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    values.update({key: value for key, value in overrides.items() if value is not None})
    rollout = dict(values.pop("rollout"))
    rollout["grid"] = tuple(rollout["grid"])
    values["rollout"] = Rollout(**rollout)
    if values["rollout"] != Rollout():
        raise ValueError("Stage 1 supports only the pinned official rollout and grid")
    for key in ("occworld_root", "upstream_config", "checkpoint", "data_root", "infos", "output"):
        if values[key] is not None:
            candidate = Path(values[key]).expanduser()
            values[key] = (project_root / candidate).resolve()
    if values["sample_index"] < 0 or values["seed"] < 0:
        raise ValueError("sample_index and seed must be nonnegative")
    return VerificationConfig(**values)
