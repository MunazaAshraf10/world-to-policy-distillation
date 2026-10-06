import torch
from torch import nn

from src.world.types import Rollout


class WorldEncoder(nn.Module):
    """Embeds OccWorld's predicted latents F^w_{t+1} into world tokens (Eq. 8).

    Inputs are frozen codebook vectors, so the world model's exact decoder input is what
    the teacher and reward model see.
    """

    def __init__(self, dim: int, codebook: torch.Tensor, rollout: Rollout = Rollout()) -> None:
        super().__init__()
        self.register_buffer("codebook", codebook.detach().clone(), persistent=True)
        e_dim = codebook.shape[1]
        self.proj = nn.Sequential(
            nn.Conv2d(e_dim, dim, 3, stride=2, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        side = rollout.latent[0] // 2
        self.step_embed = nn.Parameter(torch.zeros(1, rollout.future, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, side * side, dim))
        nn.init.trunc_normal_(self.step_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """codes [B, F, 50, 50] to tokens [B, F * 25 * 25, D]."""
        bs, f = codes.shape[:2]
        latent = self.codebook[codes.long()].permute(0, 1, 4, 2, 3).flatten(0, 1)
        tokens = self.proj(latent).flatten(2).transpose(1, 2)
        tokens = tokens.reshape(bs, f, -1, tokens.shape[-1])
        return (tokens + self.step_embed + self.pos_embed).flatten(1, 2)
