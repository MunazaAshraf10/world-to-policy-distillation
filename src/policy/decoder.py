import torch
from torch import nn


class EgoEncoder(nn.Module):
    """Embeds the driving command and the two past ego displacements into a query offset."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(3 + 4, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, command: torch.Tensor, ego_hist: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([command, ego_hist.flatten(1)], dim=-1))


class PlanDecoder(nn.Module):
    """Planning decoder P_D: queries refined by cross-attention to world features (Eq. 1)."""

    def __init__(self, dim: int, layers: int, heads: int) -> None:
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            dim, heads, dim_feedforward=4 * dim, dropout=0.0, batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(dim))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        return self.layers(queries, memory)


class PlanHead(nn.Module):
    """MLP plan head P_h (Eq. 2): per-step displacements accumulated into positions."""

    def __init__(self, dim: int, horizon: int) -> None:
        super().__init__()
        self.horizon = horizon
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, horizon * 2))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """query [..., D] to positions [..., T, 2] in the LiDAR frame of the origin."""
        # Positions reach tens of meters, beyond bf16's useful resolution.
        with torch.autocast(query.device.type, enabled=False):
            disp = self.mlp(query.float()).unflatten(-1, (self.horizon, 2))
        return disp.cumsum(dim=-2)
