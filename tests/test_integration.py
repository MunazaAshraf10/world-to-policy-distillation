from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.verification.config import load_config
from src.verification.report import write_report
from src.verification.runner import run_verification


@pytest.mark.integration
def test_real_occworld(request: pytest.FixtureRequest, tmp_path: Path) -> None:
    config_path = request.config.getoption("--integration-config")
    if config_path is None:
        pytest.skip(
            "Provide --integration-config with real checkpoint and data paths; Stage 1 is not verified"
        )
    root = Path(__file__).resolve().parents[1]
    config = load_config(Path(config_path), {}, root)
    report = run_verification(config, root)
    path = tmp_path / "summary.json"
    write_report(path, report)
    saved = json.loads(path.read_text())
    assert saved["status"] == "stage1_passed"
    assert saved["trainable_parameter_count"] == 0
    assert saved["tensors"]["semantic_labels"]["shape"] == [1, 6, 200, 200, 16]
