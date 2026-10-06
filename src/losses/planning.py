import torch


def masked_l1(traj: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean L1 over valid steps; traj and target [..., T, 2], mask broadcastable to [..., T]."""
    err = (traj - target).abs().sum(-1) * mask
    return err.sum(-1) / mask.sum(-1).clamp_min(1.0)


def plan_loss(traj: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Imitation of the expert positions by a single-modal policy, traj [B, T, 2]."""
    return masked_l1(traj, target, mask).mean()


def wta_loss(traj_set: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Winner-takes-all imitation for a multi-modal planner, traj_set [B, K, T, 2].

    Only the closest candidate is regressed, which keeps the remaining modes diverse.
    """
    err = masked_l1(traj_set, target[:, None], mask[:, None])
    return err.min(dim=-1).values.mean()
