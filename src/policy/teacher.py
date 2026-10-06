from dataclasses import dataclass

import torch
from torch import nn

from src.policy.decoder import EgoEncoder, PlanDecoder, PlanHead
from src.world.encoder import WorldEncoder


@dataclass(slots=True)
class TeacherConfig:
    dim: int = 256
    layers: int = 3
    heads: int = 8
    modes: int = 6
    horizon: int = 6


@dataclass(frozen=True, slots=True)
class TeacherOutput:
    query: torch.Tensor
    traj_set: torch.Tensor


class Teacher(nn.Module):
    """Multi-modal teacher planning on the predicted world, T_{t+1} = P_h(P_D(Q^T, F^w_{t+1})).

    Following Eq. 8, the planning decoder's memory is the world model's predicted future
    rather than current observation features. Each of the K mode queries yields one candidate.
    """

    def __init__(self, cfg: TeacherConfig, codebook: torch.Tensor) -> None:
        super().__init__()
        self.world = WorldEncoder(cfg.dim, codebook)
        self.ego = EgoEncoder(cfg.dim)
        self.mode_queries = nn.Parameter(torch.zeros(1, cfg.modes, cfg.dim))
        nn.init.trunc_normal_(self.mode_queries, std=0.02)
        self.decoder = PlanDecoder(cfg.dim, cfg.layers, cfg.heads)
        self.head = PlanHead(cfg.dim, cfg.horizon)

    def forward(
        self, codes: torch.Tensor, command: torch.Tensor, ego_hist: torch.Tensor
    ) -> TeacherOutput:
        """Returns refined queries Q^T [B, K, D] and candidates [B, K, T, 2]."""
        world = self.world(codes)
        queries = self.mode_queries + self.ego(command, ego_hist).unsqueeze(1)
        queries = self.decoder(queries, world)
        return TeacherOutput(query=queries, traj_set=self.head(queries))
