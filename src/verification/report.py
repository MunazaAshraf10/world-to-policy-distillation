from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch

from src.world.types import WorldPrediction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_state(root: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {"revision": git("rev-parse", "HEAD"), "status": git("status", "--porcelain")}


def tensor_summary(prediction: WorldPrediction, classes: int) -> dict[str, Any]:
    summary = {}
    for name, tensor in prediction.tensors().items():
        summary[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "finite": bool(torch.isfinite(tensor).all()),
            "requires_grad": tensor.requires_grad,
            "min": tensor.min().item(),
            "max": tensor.max().item(),
        }
    summary["semantic_labels"]["label_counts_per_step"] = [
        torch.bincount(frame.flatten(), minlength=classes).cpu().tolist()
        for frame in prediction.semantic_labels[0]
    ]
    return summary


def write_report(path: Path, report: dict[str, Any], overwrite: bool = False) -> None:
    """Publish complete JSON atomically; concurrent writers cannot clobber a run."""
    content = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
