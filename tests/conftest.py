from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from src.world.types import Rollout


class TinyWorld(nn.Module):
    """Small fixture for adapter contracts; never used by the verification CLI."""

    def __init__(self) -> None:
        super().__init__()
        self.vae = nn.Linear(1, 1)
        self.transformer = nn.Linear(1, 1)
        self.pose_encoder = nn.Linear(1, 1)
        self.pose_decoder = nn.Linear(1, 1)
        self.future_occupancy_leak = False
        self.future_pose_leak = False

    def forward_autoreg_with_pose(
        self,
        x: torch.Tensor,
        metas: list[dict[str, Any]],
        start_frame: int,
        mid_frame: int,
        end_frame: int,
    ) -> dict[str, torch.Tensor]:
        signal = x[:, start_frame:mid_frame].float().mean() + self.transformer.weight.sum()
        signal = signal + float(np.asarray(metas[0]["rel_poses"])[:mid_frame].sum())
        if self.future_occupancy_leak:
            signal = signal + x[:, mid_frame:].float().mean()
        if self.future_pose_leak:
            signal = signal + float(np.asarray(metas[0]["rel_poses"])[mid_frame:].sum())
        shape = (1, end_frame - mid_frame)
        logits = signal * torch.tensor([0.1, 0.2, 0.3])
        logits = logits.expand(*shape, *x.shape[2:], 3)
        return {
            "logits": logits,
            "sem_pred": logits.argmax(dim=-1),
            "pose_decoded": signal.expand(*shape, 3, 2),
            "poses_": signal.expand(*shape, 2),
            "target_occs": x[:, mid_frame:end_frame],
        }


@pytest.fixture
def rollout() -> Rollout:
    return Rollout(grid=(2, 3, 1), classes=3)


@pytest.fixture
def world() -> TinyWorld:
    return TinyWorld()


@pytest.fixture
def sample(rollout: Rollout) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    sequence = torch.zeros((1, rollout.sequence_length, *rollout.grid), dtype=torch.int64)
    modes = np.zeros((rollout.sequence_length, 3), dtype=np.float32)
    modes[:, 0] = 1
    return sequence, [
        {"rel_poses": np.zeros((rollout.sequence_length, 2), dtype=np.float32), "gt_mode": modes}
    ]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--integration-config", help="Debug YAML with explicit real checkpoint and data paths"
    )
