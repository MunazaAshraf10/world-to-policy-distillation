import torch
import torch.nn.functional as F


def imitation_loss(im_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy between the imitation scores and the soft target over candidates (Eq. 12)."""
    return -(target * F.log_softmax(im_logit, dim=-1)).sum(-1).mean()


def simulation_loss(sim_logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross entropy on the five rule based terms (Eq. 13), both [B, N, 5]."""
    return F.binary_cross_entropy_with_logits(sim_logit, target)
