import argparse
import logging
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.cache import PlanningCache
from src.evaluation.planning import evaluate_trajectories, jittered_collisions
from src.training.trainer import build_models, collect_predictions, load_codebook
from src.utils.repro import environment, git_revision, sha256, write_json

logger = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open-loop nuScenes planning evaluation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--jitter", type=float, default=0.01, help="position noise in meters")
    parser.add_argument("--draws", type=int, default=20)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load((args.checkpoint.parent / "config.yaml").read_text())
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    cache = Path(cfg["paths"]["cache"])
    # A distilled student deploys alone; only the teacher reference needs the world cache.
    stage = "student" if "student" in state["models"] else "teacher"
    world = stage == "teacher"
    codebook = load_codebook(cache) if world else torch.empty(0)
    models = build_models(cfg, stage, codebook)
    for name, model in models.items():
        model.load_state_dict(state["models"][name], strict=True)
        model.cuda()

    data = PlanningCache(cache, "val", world=world)
    loader = DataLoader(data, batch_size=args.batch_size, num_workers=8, pin_memory=True)
    trajs = collect_predictions(models, loader, torch.device("cuda"), bf16=True)
    # The planning metric runs on the CPU, so release the GPU for concurrent evaluations.
    for model in models.values():
        model.cpu()
    torch.cuda.empty_cache()
    root = Path(cfg["paths"]["occworld_root"])
    metrics = evaluate_trajectories(data, trajs, root)
    metrics |= jittered_collisions(data, trajs, root, args.jitter, args.draws)
    report = {
        "experiment": cfg["experiment"],
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "epoch": state["epoch"],
        "deployed": sorted(models),
        "metrics": metrics,
        "config": cfg,
        "git": git_revision(Path(__file__).resolve().parents[1]),
        "environment": environment(),
    }
    output = args.output or Path("results") / f"{cfg['experiment']}.json"
    write_json(output, report)
    logger.info(
        "%s: L2 %.3f / col %.3f (per-time) | L2 %.3f / col %.3f (cumulative) | "
        "jittered col %.3f +- %.3f (per-time) -> %s",
        cfg["experiment"],
        metrics["l2_avg_single"],
        metrics["box_col_avg_single"],
        metrics["l2_avg"],
        metrics["box_col_avg"],
        metrics["box_col_avg_single_jitter_mean"],
        metrics["box_col_avg_single_jitter_std"],
        output,
    )


if __name__ == "__main__":
    main()
