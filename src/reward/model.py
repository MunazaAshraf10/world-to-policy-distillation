from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from src.world.encoder import WorldEncoder

SIM_TERMS = ("nc", "dac", "ep", "ttc", "comf")


@dataclass(slots=True)
class RewardConfig:
    dim: int = 256
    layers: int = 2
    heads: int = 8
    horizon: int = 6
    position_scale: float = 10.0
    alpha: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    eps: float = 1e-6
    im_temperature: float = 1.0
    im_target: str = "corrected"


@dataclass(frozen=True, slots=True)
class RewardOutput:
    im_logit: torch.Tensor
    sim_logit: torch.Tensor


class CrossBlock(nn.Module):
    """Cross-attention without query self-attention, so candidates are scored independently."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(query)
        query = query + self.attn(q, memory, memory, need_weights=False)[0]
        return query + self.ffn(self.norm_ffn(query))


class RewardModel(nn.Module):
    """Trajectory-world interaction F^{w,i} = RewardModel(F^w_{t+1}, tau_i) (Eq. 9).

    An imitation head scores each candidate against the set and a simulation head predicts
    the five rule based terms of App. 6.3.
    """

    def __init__(self, cfg: RewardConfig, codebook: torch.Tensor) -> None:
        super().__init__()
        self.cfg = cfg
        self.world = WorldEncoder(cfg.dim, codebook)
        self.traj_encoder = nn.Sequential(
            nn.Linear(cfg.horizon * 2, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim)
        )
        self.blocks = nn.ModuleList(CrossBlock(cfg.dim, cfg.heads) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(cfg.dim)
        self.im_head = nn.Linear(cfg.dim, 1)
        self.sim_head = nn.Linear(cfg.dim, len(SIM_TERMS))

    def forward(self, codes: torch.Tensor, traj_set: torch.Tensor) -> RewardOutput:
        """codes [B, F, 50, 50], traj_set [B, N, T, 2] to logits [B, N] and [B, N, 5]."""
        world = self.world(codes)
        feat = self.traj_encoder(traj_set.flatten(2) / self.cfg.position_scale)
        for block in self.blocks:
            feat = block(feat, world)
        feat = self.norm(feat)
        return RewardOutput(im_logit=self.im_head(feat).squeeze(-1), sim_logit=self.sim_head(feat))


def final_reward(output: RewardOutput, alpha: tuple[float, ...], eps: float) -> torch.Tensor:
    """Eq. 14 on predicted probabilities, [B, N].

    The weighted TTC, EP, and comfort term is divided by 12 as in the NAVSIM PDM score so
    that it lies in [0, 1]; the paper omits this normalization.
    """
    log_im = F.log_softmax(output.im_logit, dim=-1)
    log_sim = F.logsigmoid(output.sim_logit)
    prob = dict(zip(SIM_TERMS, output.sim_logit.sigmoid().unbind(-1), strict=True))
    blend = (5.0 * prob["ttc"] + 5.0 * prob["ep"] + 2.0 * prob["comf"]) / 12.0
    return (
        alpha[0] * log_im
        + alpha[1] * log_sim[..., SIM_TERMS.index("nc")]
        + alpha[2] * log_sim[..., SIM_TERMS.index("dac")]
        + alpha[3] * torch.log(blend + eps)
    )
