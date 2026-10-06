import hashlib
import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_revision(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def environment() -> dict[str, Any]:
    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "hostname": platform.node(),
        "pid": os.getpid(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
