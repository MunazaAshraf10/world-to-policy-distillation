import argparse
import logging
from pathlib import Path

import torch

from src.data.cache import PlanningCache
from src.evaluation.forecast import evaluate_rollouts
from src.utils.config import load_config
from src.utils.repro import environment, git_revision, sha256, write_json
from src.world.occworld import OccWorldAdapter, build_world_model

logger = logging.getLogger("evaluate_world")


def main() -> None:
    parser = argparse.ArgumentParser(description="OccWorld forecast mIoU/IoU on the full val split")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/occworld_forecast.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    paths = load_config(args.config)["paths"]
    model = build_world_model(Path(paths["occworld_root"]), args.checkpoint, torch.device("cuda"))
    data = PlanningCache(Path(paths["cache"]), "val")
    metrics = evaluate_rollouts(OccWorldAdapter(model), data, args.batch_size)
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "windows": len(data),
        "batch_size": args.batch_size,
        "metrics": metrics,
        "git": git_revision(Path(__file__).resolve().parents[1]),
        "environment": environment(),
    }
    write_json(args.output, report)
    logger.info(
        "mIoU %s | IoU %s -> %s",
        " ".join(f"{metrics[f'miou_{t}']:.2f}" for t in (2, 4, 6)),
        " ".join(f"{metrics[f'iou_{t}']:.2f}" for t in (2, 4, 6)),
        args.output,
    )


if __name__ == "__main__":
    main()
