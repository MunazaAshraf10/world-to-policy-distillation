import os
from pathlib import Path

import pytest
import torch

from src.world.occworld import OccWorldAdapter, build_world_model
from src.world.types import Rollout

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(os.environ.get("OCCWORLD_CHECKPOINT", ROOT / "checkpoints/occworld/last.pt"))

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not (CHECKPOINT.is_file() and torch.cuda.is_available()),
        reason="requires a trained OccWorld checkpoint and CUDA",
    ),
]


def test_frozen_rollout_from_history_only() -> None:
    rollout = Rollout()
    model = build_world_model(ROOT / "OccWorld", CHECKPOINT, torch.device("cuda"))
    adapter = OccWorldAdapter(model)
    adapter.assert_frozen()
    history = torch.full((2, rollout.history, *rollout.grid), 17, device="cuda")
    history[..., :2] = 11
    disp = torch.tensor([0.0, 2.0], device="cuda").expand(2, rollout.history, 2)
    modes = torch.eye(3, device="cuda")[2].expand(2, rollout.history + rollout.future, 3)
    pred = adapter.predict(history, disp, modes)
    assert pred.codes.shape == (2, rollout.future, *rollout.latent)
    assert pred.occupancy.shape == (2, rollout.future, *rollout.grid)
    assert torch.isfinite(pred.ego_disp).all()
    assert torch.equal(pred.codes, adapter.predict(history, disp, modes).codes)
    adapter.assert_frozen()
