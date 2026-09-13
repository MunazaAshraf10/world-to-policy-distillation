from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.world.occworld import load_checkpoint


@pytest.mark.parametrize("wrapped,prefixed", [(False, False), (True, False), (True, True)])
def test_full_checkpoint(world: nn.Module, tmp_path: Path, wrapped: bool, prefixed: bool) -> None:
    state = {
        ("module." if prefixed else "") + key: torch.full_like(value, 0.25)
        for key, value in world.state_dict().items()
    }
    path = tmp_path / "full.pth"
    torch.save({"state_dict": state, "epoch": 125} if wrapped else state, path)
    load_checkpoint(world, path)
    for value in world.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, 0.25))


@pytest.mark.parametrize(
    "defect", ["vae_only", "missing_transformer", "shape", "unexpected", "mixed_prefix", "nan"]
)
def test_invalid_checkpoint_fails_without_partial_load(
    world: nn.Module, tmp_path: Path, defect: str
) -> None:
    before = {key: value.clone() for key, value in world.state_dict().items()}
    state = {key: value.clone() for key, value in before.items()}
    if defect == "vae_only":
        state = dict(world.vae.state_dict())
    elif defect == "missing_transformer":
        del state["transformer.weight"]
    elif defect == "shape":
        state["transformer.weight"] = torch.zeros(3, 3)
    elif defect == "unexpected":
        state["unknown"] = torch.zeros(1)
    elif defect == "mixed_prefix":
        state["module.vae.weight"] = state.pop("vae.weight")
    else:
        state["vae.weight"].fill_(float("nan"))
    path = tmp_path / "invalid.pth"
    torch.save(state, path)
    with pytest.raises(ValueError):
        load_checkpoint(world, path)
    for key, value in world.state_dict().items():
        torch.testing.assert_close(value, before[key])
