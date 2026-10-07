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
    anchor_scale: float = 10.0


@dataclass(frozen=True, slots=True)
class TeacherOutput:
    query: torch.Tensor
    traj_set: torch.Tensor


def trajectory_anchors(traj: torch.Tensor, k: int, iters: int = 50, seed: int = 0) -> torch.Tensor:
    """K-means centers [k, T, 2] of expert trajectories [N, T, 2], seeded with k-means++."""
    points = traj.flatten(1).double()
    gen = torch.Generator().manual_seed(seed)
    centers = points[torch.randint(len(points), (1,), generator=gen)]
    for _ in range(1, k):
        dist = torch.cdist(points, centers).min(dim=1).values ** 2
        centers = torch.cat([centers, points[torch.multinomial(dist, 1, generator=gen)]])
    for _ in range(iters):
        assign = torch.cdist(points, centers).argmin(dim=1)
        centers = torch.stack(
            [points[assign == j].mean(0) if (assign == j).any() else centers[j] for j in range(k)]
        )
    return centers.float().view(k, *traj.shape[1:])


class Teacher(nn.Module):
    """Multi-modal teacher planning on the predicted world, T_{t+1} = P_h(P_D(Q^T, F^w_{t+1})).

    Following Eq. 8, the planning decoder's memory is the world model's predicted future
    rather than current observation features. Each of the K mode queries is tied to a k-means
    trajectory anchor and predicts its candidate as a residual from that anchor, as in the
    UniAD heads that Sec. 3.1 cites for the candidate set.
    """

    def __init__(self, cfg: TeacherConfig, codebook: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        self.world = WorldEncoder(cfg.dim, codebook)
        self.ego = EgoEncoder(cfg.dim)
        self.register_buffer("anchors", torch.zeros(cfg.modes, cfg.horizon, 2))
        self.anchor_embed = nn.Sequential(
            nn.Linear(cfg.horizon * 2, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim)
        )
        self.mode_queries = nn.Parameter(torch.zeros(1, cfg.modes, cfg.dim))
        nn.init.trunc_normal_(self.mode_queries, std=0.02)
        self.decoder = PlanDecoder(cfg.dim, cfg.layers, cfg.heads)
        self.head = PlanHead(cfg.dim, cfg.horizon)

    def forward(
        self, codes: torch.Tensor, command: torch.Tensor, ego_hist: torch.Tensor
    ) -> TeacherOutput:
        """Returns refined queries Q^T [B, K, D] and candidates [B, K, T, 2]."""
        world = self.world(codes)
        anchor = self.anchor_embed(self.anchors.flatten(1) / self.cfg.anchor_scale)
        queries = self.mode_queries + anchor + self.ego(command, ego_hist).unsqueeze(1)
        queries = self.decoder(queries, world)
        return TeacherOutput(query=queries, traj_set=self.anchors + self.head(queries))
