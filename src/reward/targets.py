from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.world.types import DRIVABLE, OBJECT_CLASSES, OFFROAD_CLASSES

ROAD_CLOSING = 3


@dataclass(slots=True)
class SimConfig:
    """Rule thresholds of App. 6.3; comfort bounds follow the NAVSIM standard (Eq. 23)."""

    ego_length: float = 4.084
    ego_width: float = 1.85
    ego_offset: float = 0.5
    voxel: float = 0.4
    grid_min: float = -40.0
    min_progress: float = 5.0
    ttc_distance: float = 10.0
    step_seconds: float = 0.5
    lon_acc: tuple[float, float] = (-4.05, 2.40)
    lat_acc: float = 4.89
    jerk: float = 8.37
    lon_jerk: float = 4.13
    closing: int = ROAD_CLOSING
    fit_degree: int = 3


def close_mask(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    """Morphological closing of [..., X, Y] bool masks with a square kernel."""
    flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
    pad = kernel // 2
    dilated = F.max_pool2d(flat, kernel, stride=1, padding=pad)
    closed = -F.max_pool2d(-dilated, kernel, stride=1, padding=pad)
    return closed.squeeze(1).bool().view_as(mask)


def bev_masks(occupancy: torch.Tensor, closing: int = ROAD_CLOSING) -> torch.Tensor:
    """[..., X, Y, Z] labels to obstacle and drivable columns [..., 2, X, Y].

    Occ3D labels only observed voxels, so unseen road reads as free space. A column is
    therefore off road only when it holds an explicit non road class (other flat, sidewalk,
    terrain, manmade, vegetation) and neither road nor an object; isolated off road cells
    are removed by a morphological closing of the drivable mask.
    """
    device = occupancy.device
    obstacle = torch.isin(occupancy, torch.tensor(OBJECT_CLASSES, device=device)).any(dim=-1)
    road = (occupancy == DRIVABLE).any(dim=-1)
    offroad = torch.isin(occupancy, torch.tensor(OFFROAD_CLASSES, device=device)).any(dim=-1)
    drivable = close_mask(~(offroad & ~road & ~obstacle), closing)
    return torch.stack([obstacle, drivable], dim=-3)


def headings(traj: torch.Tensor) -> torch.Tensor:
    """Yaw of each step from successive positions, holding the last heading when nearly stopped."""
    origin = torch.zeros_like(traj[..., :1, :])
    delta = torch.diff(torch.cat([origin, traj], dim=-2), dim=-2)
    yaw = torch.atan2(delta[..., 0], delta[..., 1])
    moving = delta.norm(dim=-1) > 0.05
    yaw = torch.where(moving, yaw, torch.zeros_like(yaw))
    for t in range(1, yaw.shape[-1]):
        yaw[..., t] = torch.where(moving[..., t], yaw[..., t], yaw[..., t - 1])
    return yaw


def footprint(cfg: SimConfig, device: torch.device) -> torch.Tensor:
    """Points [P, 2] covering the ego box as (lateral, longitudinal) offsets in meters."""
    lon = torch.linspace(-cfg.ego_length / 2, cfg.ego_length / 2, 5, device=device)
    lat = torch.linspace(-cfg.ego_width / 2, cfg.ego_width / 2, 3, device=device)
    grid = torch.stack(torch.meshgrid(lat, lon + cfg.ego_offset, indexing="ij"), dim=-1)
    return grid.reshape(-1, 2)


def place(points: torch.Tensor, traj: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rigidly place local points [P, 2] at every pose; returns [..., T, P, 2] LiDAR frame.

    Yaw is measured from the forward (+y) LiDAR axis toward +x.
    """
    cos, sin = yaw.cos()[..., None], yaw.sin()[..., None]
    lat, lon = points[:, 0], points[:, 1]
    x = traj[..., 0:1] + lat * cos + lon * sin
    y = traj[..., 1:2] - lat * sin + lon * cos
    return torch.stack([x, y], dim=-1)


def sample_masks(
    masks: torch.Tensor,
    points: torch.Tensor,
    ego_shift: torch.Tensor,
    lidar2ego: torch.Tensor,
    cfg: SimConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up predicted BEV masks at LiDAR frame points.

    masks: [B, T, 2, X, Y] obstacle and drivable, each step in its own predicted ego frame.
    points: [B, N, T, P, 2] in the origin LiDAR frame.
    ego_shift: [B, T, 2] cumulative predicted ego displacement; yaw is ignored as in upstream.
    lidar2ego: [B, 2, 3] planar extrinsics of the origin frame.
    Returns obstacle and drivable hits [B, N, T, P] and an in-grid flag of the same shape.
    """
    bs, n, t, p, _ = points.shape
    local = points - ego_shift[:, None, :, None, :]
    rot, trans = lidar2ego[:, None, None, None, :, :2], lidar2ego[:, None, None, None, :, 2]
    ego = (rot @ local.unsqueeze(-1)).squeeze(-1) + trans
    idx = torch.floor((ego - cfg.grid_min) / cfg.voxel).long()
    size = masks.shape[-1]
    inside = ((idx >= 0) & (idx < size)).all(dim=-1)
    idx = idx.clamp(0, size - 1)
    b = torch.arange(bs, device=points.device)[:, None, None, None].expand(bs, n, t, p)
    s = torch.arange(t, device=points.device)[None, None, :, None].expand(bs, n, t, p)
    obstacle = masks[b, s, 0, idx[..., 0], idx[..., 1]] & inside
    drivable = masks[b, s, 1, idx[..., 0], idx[..., 1]] | ~inside
    return obstacle, drivable


def comfort(traj: torch.Tensor, cfg: SimConfig) -> torch.Tensor:
    """Eq. 23 on derivatives of a least squares polynomial through the origin and positions.

    Raw third differences of 2 Hz positions amplify annotation noise past the jerk bounds;
    NAVSIM likewise filters trajectories before differentiating. Returns [B, N].
    """
    steps = traj.shape[-2]
    t = torch.arange(steps + 1, device=traj.device, dtype=traj.dtype) * cfg.step_seconds
    powers = torch.arange(cfg.fit_degree + 1, device=traj.device, dtype=traj.dtype)
    basis = t[:, None] ** powers
    points = torch.cat([torch.zeros_like(traj[..., :1, :]), traj], dim=-2)
    coef = torch.linalg.pinv(basis) @ points

    def derivative(order: int) -> torch.Tensor:
        scale = torch.ones_like(powers)
        for k in range(order):
            scale = scale * (powers - k)
        shifted = (powers - order).clamp_min(0)
        return (t[:, None] ** shifted * scale) @ coef

    vel, acc, jerk = derivative(1), derivative(2), derivative(3)
    yaw = torch.atan2(vel[..., 0], vel[..., 1])
    fwd = torch.stack([yaw.sin(), yaw.cos()], dim=-1)
    left = torch.stack([-yaw.cos(), yaw.sin()], dim=-1)
    a_lon = (acc * fwd).sum(-1)
    a_lat = (acc * left).sum(-1)
    j_lon = (jerk * fwd).sum(-1)
    ok = (
        ((a_lon >= cfg.lon_acc[0]) & (a_lon <= cfg.lon_acc[1])).all(-1)
        & (a_lat.abs() <= cfg.lat_acc).all(-1)
        & (jerk.norm(dim=-1) <= cfg.jerk).all(-1)
        & (j_lon.abs() <= cfg.lon_jerk).all(-1)
    )
    return ok.float()


def simulation_targets(
    traj_set: torch.Tensor,
    masks: torch.Tensor,
    ego_disp: torch.Tensor,
    lidar2ego: torch.Tensor,
    cfg: SimConfig,
) -> torch.Tensor:
    """Rule based targets [B, N, 5] in the order NC, DAC, EP, TTC, Comf (Eqs. 17 to 23).

    All terms are evaluated against the world model's predicted occupancy, not ground truth.
    traj_set: [B, N, T, 2] candidate positions in the origin LiDAR frame.
    """
    yaw = headings(traj_set)
    shift = ego_disp.cumsum(dim=1)
    body = place(footprint(cfg, traj_set.device), traj_set, yaw)
    obstacle, drivable = sample_masks(masks, body, shift, lidar2ego, cfg)
    nc = (~obstacle.flatten(-2).any(-1)).float()
    dac = drivable.flatten(-2).all(-1).float()

    # Longitudinal progress along the origin heading (+y), normalized within the set (Eq. 19).
    progress = traj_set[..., -1, 1]
    best = progress.max(dim=-1, keepdim=True).values
    ratio = torch.where(best > cfg.min_progress, progress / best.clamp_min(1e-6), 1.0)
    ep = torch.where(progress >= 0, ratio.clamp(0, 1), 0.0) * nc * dac

    # TTC extends the final pose forward by a fixed distance (Eq. 21).
    reach = torch.linspace(cfg.ttc_distance / 4, cfg.ttc_distance, 4, device=traj_set.device)
    offsets = torch.stack([torch.zeros_like(reach), reach], dim=-1)
    ahead = (footprint(cfg, traj_set.device)[None] + offsets[:, None]).reshape(-1, 2)
    tip = place(ahead, traj_set[..., -1:, :], yaw[..., -1:])
    tip_obstacle, _ = sample_masks(masks[:, -1:], tip, shift[:, -1:], lidar2ego, cfg)
    ttc = (~tip_obstacle.flatten(-2).any(-1)).float() * dac

    return torch.stack([nc, dac, ep, ttc, comfort(traj_set, cfg)], dim=-1)


def imitation_target(
    traj_set: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    literal: bool = False,
) -> torch.Tensor:
    """Soft imitation target over candidates (Eq. 11), [B, N].

    Eq. 11 as printed, softmax(-d_i / sum_j -d_j), cancels the signs and favors the farthest
    candidate. The corrected form softmax(-d_i / temperature) is the default.
    """
    err = (traj_set - target[:, None]).norm(dim=-1)
    dist = (err * mask[:, None]).sum(-1) / mask.sum(-1, keepdim=True).clamp_min(1.0)
    if literal:
        return torch.softmax(-dist / (-dist).sum(-1, keepdim=True).clamp_max(-1e-6), dim=-1)
    return torch.softmax(-dist / temperature, dim=-1)
