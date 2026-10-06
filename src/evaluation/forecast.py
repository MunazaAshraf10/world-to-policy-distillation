import torch

from src.world.types import FREE, Rollout


class ForecastMeter:
    """Occupancy forecasting mIoU and IoU per future step, as in upstream multi_step_MeanIou.

    Semantic mIoU averages classes 0 to 16, scoring a class absent from the targets as 1 like
    upstream. IoU treats every non free voxel as occupied.
    """

    def __init__(self, rollout: Rollout = Rollout()) -> None:
        n = FREE + 1
        self.seen = torch.zeros(rollout.future, n)
        self.correct = torch.zeros(rollout.future, n)
        self.positive = torch.zeros(rollout.future, n)
        self.occupied = torch.zeros(rollout.future, 3)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """pred and target [B, F, X, Y, Z] labels."""
        n = FREE + 1
        for t in range(pred.shape[1]):
            p, g = pred[:, t].flatten(), target[:, t].flatten()
            self.seen[t] += torch.bincount(g, minlength=n).float()
            self.positive[t] += torch.bincount(p, minlength=n).float()
            self.correct[t] += torch.bincount(g[p == g], minlength=n).float()
            occ_p, occ_g = p != FREE, g != FREE
            hits = torch.stack([(occ_p & occ_g).sum(), occ_g.sum(), occ_p.sum()])
            self.occupied[t] += hits.float()

    def summary(self) -> dict[str, float]:
        steps = int((self.seen.sum(-1) > 0).sum())
        union = self.seen + self.positive - self.correct
        iou = torch.where(self.seen > 0, self.correct / union.clamp_min(1), torch.ones_like(union))
        miou = 100 * iou[:steps, :FREE].mean(-1)
        hit, gt, pos = self.occupied[:steps].unbind(-1)
        occ_iou = 100 * hit / (gt + pos - hit).clamp_min(1)
        out = {f"miou_{t + 1}": miou[t].item() for t in range(steps)}
        out |= {f"iou_{t + 1}": occ_iou[t].item() for t in range(steps)}
        out["miou_mean"] = miou.mean().item()
        out["iou_mean"] = occ_iou.mean().item()
        return out
