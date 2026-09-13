from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch
from torch import nn

from src.verification.runner import verify_rollout
from src.world.occworld import OccWorldAdapter, normalize_prediction, upstream_imports
from src.world.types import Rollout


def test_frozen_parity_and_temporal_contract(
    world: nn.Module, rollout: Rollout, sample: Any
) -> None:
    adapter = OccWorldAdapter(world, rollout)
    prediction, checks = verify_rollout(adapter, *sample)
    assert checks["adapter_direct_parity"] == "passed"
    assert checks["future_occupancy_isolation"] == "passed"
    assert checks["future_displacement_isolation"] == "passed"
    assert prediction.semantic_labels.shape == (1, 6, 2, 3, 1)
    assert not any(value.requires_grad for value in prediction.tensors().values())
    assert all(
        not parameter.requires_grad and parameter.grad is None for parameter in world.parameters()
    )
    assert not hasattr(prediction, "target_occs")


@pytest.mark.parametrize("leak", ["future_occupancy_leak", "future_pose_leak"])
def test_isolation_check_detects_leaking_model(
    world: nn.Module, rollout: Rollout, sample: Any, leak: str
) -> None:
    setattr(world, leak, True)
    with pytest.raises(AssertionError):
        verify_rollout(OccWorldAdapter(world, rollout), *sample)


@pytest.mark.parametrize("defect", ["shape", "dtype", "label", "mode", "pose", "batch"])
def test_invalid_input(world: nn.Module, rollout: Rollout, sample: Any, defect: str) -> None:
    sequence, metadata = sample
    if defect == "shape":
        sequence = sequence[:, :-1]
    elif defect == "dtype":
        sequence = sequence.float()
    elif defect == "label":
        sequence[0, 0, 0, 0, 0] = 3
    elif defect == "mode":
        metadata[0]["gt_mode"][0] = 0
    elif defect == "pose":
        metadata[0]["rel_poses"][0, 0] = float("nan")
    else:
        metadata = []
    with pytest.raises(ValueError):
        OccWorldAdapter(world, rollout).predict(sequence, metadata)


@pytest.mark.parametrize("defect", ["shape", "nan", "labels", "grad"])
def test_invalid_outputs(world: nn.Module, rollout: Rollout, sample: Any, defect: str) -> None:
    world.requires_grad_(False)
    output = world.forward_autoreg_with_pose(*sample, 0, 5, 11)
    if defect == "shape":
        output["poses_"] = output["poses_"][:, :-1]
    elif defect == "nan":
        output["logits"] = torch.full_like(output["logits"], float("nan"))
    elif defect == "labels":
        output["sem_pred"] = (output["sem_pred"] + 1) % 3
    else:
        output["poses_"] = output["poses_"].clone().requires_grad_()
    with pytest.raises(ValueError):
        normalize_prediction(output, rollout)


@pytest.mark.parametrize("defect", ["training", "requires_grad", "stale_grad"])
def test_freeze_tampering_rejected(
    world: nn.Module, rollout: Rollout, sample: Any, defect: str
) -> None:
    adapter = OccWorldAdapter(world, rollout)
    if defect == "training":
        world.train()
    elif defect == "requires_grad":
        world.transformer.weight.requires_grad_(True)
    else:
        world.transformer.weight.grad = torch.ones_like(world.transformer.weight)
    with pytest.raises(RuntimeError):
        adapter.predict(*sample)


def test_import_conflict_is_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "model").mkdir()
    (tmp_path / "model/TransVQVAE.py").touch()
    module = ModuleType("utils")
    module.__file__ = "/elsewhere/utils/__init__.py"
    monkeypatch.setitem(sys.modules, "utils", module)
    before = sys.path.copy()
    with pytest.raises(RuntimeError, match="already imported"):
        with upstream_imports(tmp_path):
            pytest.fail("conflicting import accepted")
    assert sys.path == before
