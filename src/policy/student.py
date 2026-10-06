from dataclasses import dataclass

import torch
from torch import nn

from src.policy.decoder import EgoEncoder, PlanDecoder, PlanHead
from src.policy.encoder import OccEncoder


@dataclass(slots=True)
class StudentConfig:
    dim: int = 128
    widths: tuple[int, ...] = (32, 64, 128)
    layers: int = 2
    heads: int = 4
    frames: int = 5
    horizon: int = 6


@dataclass(frozen=True, slots=True)
class StudentOutput:
    query: torch.Tensor
    traj: torch.Tensor


class Student(nn.Module):
    """Single-modal student T = P_h(P_D(Q^S, F^w)) (Eq. 3); never touches the world model."""

    def __init__(self, cfg: StudentConfig) -> None:
        super().__init__()
        self.encoder = OccEncoder(cfg.dim, cfg.frames, tuple(cfg.widths))
        self.ego = EgoEncoder(cfg.dim)
        self.plan_query = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        nn.init.trunc_normal_(self.plan_query, std=0.02)
        self.decoder = PlanDecoder(cfg.dim, cfg.layers, cfg.heads)
        self.head = PlanHead(cfg.dim, cfg.horizon)

    def forward(
        self, occ: torch.Tensor, command: torch.Tensor, ego_hist: torch.Tensor
    ) -> StudentOutput:
        """Returns the refined plan query Q^S [B, D] and trajectory [B, T, 2]."""
        feat = self.encoder(occ)
        query = self.plan_query + self.ego(command, ego_hist).unsqueeze(1)
        query = self.decoder(query, feat).squeeze(1)
        return StudentOutput(query=query, traj=self.head(query))
