from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.verification.config import load_config
from src.verification.report import write_report


def test_missing_assets_are_actionable() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/debug.yaml", {}, root)
    with pytest.raises(ValueError, match="--checkpoint"):
        config.validate_assets()


def test_config_override_and_resolution(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/debug.yaml",
        {"checkpoint": "checkpoints/full.pth", "output": tmp_path / "report.json"},
        root,
    )
    assert config.checkpoint == root / "checkpoints/full.pth"
    assert config.output == tmp_path / "report.json"
    assert config.rollout.future_steps == 6


def test_report_refuses_overwrite_and_preserves_original(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    write_report(path, {"status": "first"})
    with pytest.raises(FileExistsError):
        write_report(path, {"status": "second"})
    assert json.loads(path.read_text()) == {"status": "first"}
    write_report(path, {"status": "second"}, overwrite=True)
    assert json.loads(path.read_text()) == {"status": "second"}
    assert list(tmp_path.iterdir()) == [path]


def test_report_rejects_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    with pytest.raises(ValueError):
        write_report(path, {"metric": float("nan")})
    assert not path.exists()
