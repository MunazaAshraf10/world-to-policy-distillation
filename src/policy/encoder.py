import torch
from torch import nn


def conv_block(c_in: int, c_out: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, c_out),
        nn.SiLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.GroupNorm(8, c_out),
        nn.SiLU(inplace=True),
    )


class OccEncoder(nn.Module):
    """Encodes observed occupancy into BEV planning features F^w (Sec. 3.1).

    Classes are embedded and heights folded into channels as in OccWorld's VAE, then
    each frame is downsampled 8x and the frames are fused by a 1x1 projection.
    """

    def __init__(
        self,
        dim: int,
        frames: int,
        widths: tuple[int, ...] = (32, 64, 128),
        class_dim: int = 8,
        classes: int = 18,
        grid: tuple[int, int, int] = (200, 200, 16),
    ) -> None:
        super().__init__()
        self.class_embed = nn.Embedding(classes, class_dim)
        stages, c_in = [], grid[2] * class_dim
        for width in widths:
            stages.append(conv_block(c_in, width, stride=2))
            c_in = width
        self.stages = nn.Sequential(*stages)
        self.fuse = nn.Conv2d(frames * c_in, dim, 1)
        side = grid[0] // 2 ** len(widths), grid[1] // 2 ** len(widths)
        self.pos_embed = nn.Parameter(torch.zeros(1, side[0] * side[1], dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.frames = frames

    def forward(self, occ: torch.Tensor) -> torch.Tensor:
        """occ [B, T, X, Y, Z] labels to tokens [B, X/8 * Y/8, D]."""
        bs, t, x, y, z = occ.shape
        if t != self.frames:
            raise ValueError(f"Encoder expects {self.frames} frames, got {t}")
        feat = self.class_embed(occ.long()).view(bs * t, x, y, -1).permute(0, 3, 1, 2)
        feat = self.stages(feat)
        feat = feat.unflatten(0, (bs, t)).flatten(1, 2)
        return self.fuse(feat).flatten(2).transpose(1, 2) + self.pos_embed
