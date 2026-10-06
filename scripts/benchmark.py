import argparse
import logging
from pathlib import Path

import torch
import yaml

from src.data.cache import PlanningCache
from src.evaluation.latency import measure, parameter_count
from src.training.steps import select_best
from src.training.trainer import build_models, load_codebook
from src.utils.repro import environment, write_json
from src.world.occworld import OccWorldAdapter, build_world_model

logger = logging.getLogger("benchmark")


def load(checkpoint: Path, codebook: torch.Tensor) -> dict[str, torch.nn.Module]:
    cfg = yaml.safe_load((checkpoint.parent / "config.yaml").read_text())
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["models"]
    models = build_models(cfg, "student" if "student" in state else "teacher", codebook)
    for name, model in models.items():
        model.load_state_dict(state[name], strict=True)
        model.cuda().eval()
    return models


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deployment latency: student vs world model pipeline"
    )
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--world", type=Path, required=True, help="trained OccWorld checkpoint")
    parser.add_argument("--cache", type=Path, default=Path("cache"))
    parser.add_argument("--occworld-root", type=Path, default=Path("OccWorld"))
    parser.add_argument("--output", type=Path, default=Path("results/latency.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.backends.cudnn.benchmark = True

    codebook = load_codebook(args.cache)
    student = load(args.student, codebook)["student"]
    teacher = load(args.teacher, codebook)
    world = OccWorldAdapter(build_world_model(args.occworld_root, args.world, torch.device("cuda")))
    sample = PlanningCache(args.cache, "val", limit=1)[0]
    batch = {k: v[None].cuda() for k, v in sample.items()}
    occ = batch["occ"].long()

    def student_plan() -> torch.Tensor:
        return student(batch["occ"], batch["command"], batch["ego_hist"]).traj

    def teacher_plan() -> torch.Tensor:
        pred = world.predict(occ, batch["rel_poses"], batch["modes"])
        plan, best = select_best(
            teacher["teacher"], teacher["reward"], batch | {"codes": pred.codes}
        )
        return plan.traj_set[0, best[0]]

    report = {
        "batch_size": 1,
        "student": measure(student_plan) | {"parameters": parameter_count(student)},
        "teacher_pipeline": measure(teacher_plan, warmup=3, iters=20)
        | {"parameters": parameter_count(world.model, *teacher.values())},
        "environment": environment(),
    }
    report["speedup"] = report["teacher_pipeline"]["mean_ms"] / report["student"]["mean_ms"]
    write_json(args.output, report)
    logger.info(
        "student %.2f ms, OccWorld + teacher + reward %.2f ms, %.1fx",
        report["student"]["mean_ms"],
        report["teacher_pipeline"]["mean_ms"],
        report["speedup"],
    )


if __name__ == "__main__":
    main()
