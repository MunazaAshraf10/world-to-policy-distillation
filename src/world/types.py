from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Rollout:
    start_frame: int = 0
    mid_frame: int = 5
    end_frame: int = 11
    sequence_length: int = 12
    grid: tuple[int, int, int] = (200, 200, 16)
    classes: int = 18
    modes: int = 3
    step_seconds: float = 0.5

    @property
    def future_steps(self) -> int:
        return self.end_frame - self.mid_frame


@dataclass(frozen=True)
class WorldPrediction:
    """Native grid axes; ego values are per-step displacements, not positions."""

    semantic_labels: torch.Tensor
    semantic_logits: torch.Tensor
    ego_displacement_modes: torch.Tensor
    selected_ego_displacements: torch.Tensor

    def tensors(self) -> dict[str, torch.Tensor]:
        return {
            "semantic_labels": self.semantic_labels,
            "semantic_logits": self.semantic_logits,
            "ego_displacement_modes": self.ego_displacement_modes,
            "selected_ego_displacements": self.selected_ego_displacements,
        }
