from pathlib import Path

import numpy as np
import torch

from src.data.cache import PlanningCache
from src.world.occworld import InstanceBoxes, upstream_imports

HORIZONS = (1, 2, 3)


def l2_errors(traj: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-step Euclidean error [B, T] between positions [B, T, 2]."""
    return (traj - target).norm(dim=-1)


class PlanningEvaluator:
    """nuScenes open-loop L2 and collision rate through OccWorld's STP3 planning metric.

    Reports both conventions: the per-time value at each horizon (suffix _single, used by
    OccWorld and the WPT tables) and the average up to the horizon (ST-P3 and VAD).
    """

    def __init__(self, occworld_root: Path) -> None:
        with upstream_imports(occworld_root):
            from utils.metric_stp3 import PlanningMetric

        self.metric = PlanningMetric()
        self.sums: dict[str, float] = {}
        self.count = 0

    def add(
        self, traj: np.ndarray, target: np.ndarray, boxes: np.ndarray, attrs: np.ndarray
    ) -> None:
        """One sample: predicted and expert positions [T, 2] in the origin LiDAR frame."""
        seg, ped = self.metric.get_label(InstanceBoxes(boxes), torch.from_numpy(attrs)[None])
        occupancy = torch.logical_or(seg, ped)
        pred = torch.from_numpy(traj).float()[None]
        gt = torch.from_numpy(target).float()[None]
        for sec in HORIZONS:
            t = 2 * sec
            values = {
                f"l2_{sec}s": self.metric.compute_L2(pred[0, :t], gt[0, :t]),
                f"l2_{sec}s_single": self.metric.compute_L2(pred[0, t - 1 : t], gt[0, t - 1 : t]),
            }
            col, box_col = self.metric.evaluate_coll(pred[:, :t], gt[:, :t], occupancy)
            col_s, box_col_s = self.metric.evaluate_coll(
                pred[:, t - 1 : t], gt[:, t - 1 : t], occupancy[:, t - 1 : t]
            )
            values |= {
                f"col_{sec}s": col.mean().item(),
                f"box_col_{sec}s": box_col.mean().item(),
                f"col_{sec}s_single": col_s.item(),
                f"box_col_{sec}s_single": box_col_s.item(),
            }
            for key, value in values.items():
                self.sums[key] = self.sums.get(key, 0.0) + float(value)
        self.count += 1

    def summary(self) -> dict[str, float]:
        """Means over samples; L2 in meters, collision rates in percent."""
        out = {}
        for key, total in self.sums.items():
            scale = 100.0 if "col" in key else 1.0
            out[key] = scale * total / max(self.count, 1)
        for suffix in ("", "_single"):
            for name in ("l2", "col", "box_col"):
                out[f"{name}_avg{suffix}"] = float(
                    np.mean([out[f"{name}_{s}s{suffix}"] for s in HORIZONS])
                )
        out["samples"] = self.count
        return out


def evaluate_trajectories(
    dataset: PlanningCache, trajs: np.ndarray, occworld_root: Path
) -> dict[str, float]:
    """Score predicted positions [N, T, 2] aligned with the dataset's windows."""
    evaluator = PlanningEvaluator(occworld_root)
    targets = np.cumsum(dataset.windows["target"][: len(trajs)], axis=1)
    masks = dataset.windows["target_mask"][: len(trajs)]
    for i in range(len(trajs)):
        if masks[i].min() < 1:
            continue
        evaluator.add(trajs[i], targets[i], *dataset.annotations(i))
    return evaluator.summary()


def jittered_collisions(
    dataset: PlanningCache,
    trajs: np.ndarray,
    occworld_root: Path,
    sigma: float = 0.01,
    draws: int = 20,
    seed: int = 0,
) -> dict[str, float]:
    """Collision rates averaged over Gaussian position jitter of sigma meters.

    A few windows sit on the collision boundary, so centimeter changes flip the exact rate by
    about half a point; the jittered mean and its spread make runs comparable.
    """
    rng = np.random.default_rng(seed)
    runs = [
        evaluate_trajectories(
            dataset, trajs + rng.normal(0.0, sigma, trajs.shape).astype(trajs.dtype), occworld_root
        )
        for _ in range(draws)
    ]
    out = {}
    for key in ("box_col_avg_single", "box_col_avg"):
        values = np.array([run[key] for run in runs])
        out[f"{key}_jitter_mean"] = float(values.mean())
        out[f"{key}_jitter_std"] = float(values.std())
    return out
