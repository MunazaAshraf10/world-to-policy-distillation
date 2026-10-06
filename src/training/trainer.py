import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from src.data.cache import PlanningCache
from src.evaluation.planning import evaluate_trajectories
from src.losses.total import total_loss
from src.policy.student import Student, StudentConfig
from src.policy.teacher import Teacher, TeacherConfig
from src.reward.model import RewardConfig, RewardModel
from src.reward.targets import SimConfig
from src.training.steps import Batch, select_best, student_step, teacher_step, wpt_step
from src.utils.config import build
from src.utils.repro import environment, git_revision, seed_everything, write_json

logger = logging.getLogger(__name__)

STAGES = ("student", "teacher", "wpt")


@dataclass(slots=True)
class OptimConfig:
    lr: float = 2e-4
    weight_decay: float = 0.01
    epochs: int = 12
    batch_size: int = 64
    warmup_steps: int = 200
    grad_clip: float = 5.0
    workers: int = 8
    bf16: bool = True
    limit: int | None = None
    log_every: int = 50


def load_codebook(cache: Path) -> torch.Tensor:
    return torch.load(cache / "codebook.pt", weights_only=True)


def load_frozen(models: dict[str, nn.Module], path: Path) -> None:
    """Restore teacher and reward model weights and freeze them for distillation."""
    state = torch.load(path, map_location="cpu", weights_only=True)["models"]
    for name in ("teacher", "reward"):
        models[name].load_state_dict(state[name], strict=True)
        models[name].requires_grad_(False).eval()


def build_models(cfg: dict[str, Any], stage: str, codebook: torch.Tensor) -> dict[str, nn.Module]:
    models: dict[str, nn.Module] = {}
    if stage in ("student", "wpt"):
        models["student"] = Student(build(StudentConfig, cfg.get("student")))
    if stage in ("teacher", "wpt"):
        models["teacher"] = Teacher(build(TeacherConfig, cfg.get("teacher")), codebook)
        models["reward"] = RewardModel(build(RewardConfig, cfg.get("reward")), codebook)
    if stage == "wpt":
        # Training-only map from the student query space into the teacher's (Eq. 15).
        models["projector"] = nn.Linear(
            models["student"].plan_query.shape[-1], models["teacher"].mode_queries.shape[-1]
        )
        load_frozen(models, Path(cfg["paths"]["teacher_checkpoint"]))
    return models


def cosine_schedule(
    optimizer: torch.optim.Optimizer, warmup: int, total: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def to_device(batch: Batch, device: torch.device) -> Batch:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def predict(models: dict[str, nn.Module], batch: Batch) -> torch.Tensor:
    """Deployment trajectory [B, T, 2]: the student alone, or the reward selected teacher plan."""
    if "student" in models:
        return models["student"](batch["occ"], batch["command"], batch["ego_hist"]).traj
    plan, best = select_best(models["teacher"], models["reward"], batch)
    return plan.traj_set[torch.arange(best.shape[0], device=best.device), best]


def collect_predictions(
    models: dict[str, nn.Module], loader: DataLoader, device: torch.device, bf16: bool
) -> np.ndarray:
    for model in models.values():
        model.eval()
    trajs = []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            trajs.append(predict(models, to_device(batch, device)).float().cpu())
    return torch.cat(trajs).numpy()


class Trainer:
    def __init__(self, cfg: dict[str, Any], root: Path) -> None:
        self.cfg = cfg
        self.stage = cfg["stage"]
        if self.stage not in STAGES:
            raise ValueError(f"Unknown stage {self.stage}; expected one of {STAGES}")
        self.root = root
        self.optim = build(OptimConfig, cfg.get("optim"))
        self.loss_weights = {k: float(v) for k, v in cfg["loss"].items()}
        self.sim = build(SimConfig, cfg.get("sim"))
        self.device = torch.device("cuda")
        self.out = Path(cfg["paths"]["checkpoints"]) / cfg["experiment"]
        seed_everything(int(cfg["seed"]))

        cache = Path(cfg["paths"]["cache"])
        world = self.stage != "student"
        self.train_set = PlanningCache(cache, "train", world=world, limit=self.optim.limit)
        self.val_set = PlanningCache(cache, "val", world=world, limit=cfg.get("val_limit"))
        codebook = load_codebook(cache) if world else torch.empty(0)
        self.models = {
            k: m.to(self.device) for k, m in build_models(cfg, self.stage, codebook).items()
        }
        self.params = [p for m in self.models.values() for p in m.parameters() if p.requires_grad]

    def loader(self, dataset: PlanningCache, train: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.optim.batch_size,
            shuffle=train,
            drop_last=train,
            num_workers=self.optim.workers,
            pin_memory=True,
            persistent_workers=self.optim.workers > 0,
        )

    def step_terms(self, batch: Batch) -> dict[str, torch.Tensor]:
        m = self.models
        match self.stage:
            case "student":
                return student_step(batch, m["student"])
            case "teacher":
                return teacher_step(batch, m["teacher"], m["reward"], self.sim)
            case "wpt":
                return wpt_step(batch, m["student"], m["teacher"], m["reward"], m["projector"])
        raise AssertionError(self.stage)

    def set_train_mode(self) -> None:
        for name, model in self.models.items():
            frozen = self.stage == "wpt" and name in ("teacher", "reward")
            model.train(not frozen)

    def save(self, name: str, epoch: int, metrics: dict[str, Any]) -> Path:
        deploy = ("student",) if self.stage == "wpt" else tuple(self.models)
        path = self.out / f"{name}.pt"
        torch.save(
            {
                "stage": self.stage,
                "epoch": epoch,
                "models": {k: self.models[k].state_dict() for k in deploy},
                "metrics": metrics,
            },
            path,
        )
        return path

    def validate(self) -> dict[str, float]:
        loader = self.loader(self.val_set, train=False)
        trajs = collect_predictions(self.models, loader, self.device, self.optim.bf16)
        return evaluate_trajectories(self.val_set, trajs, Path(self.cfg["paths"]["occworld_root"]))

    def run(self) -> dict[str, Any]:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        loader = self.loader(self.train_set, train=True)
        total_steps = self.optim.epochs * len(loader)
        optimizer = torch.optim.AdamW(
            self.params, lr=self.optim.lr, weight_decay=self.optim.weight_decay
        )
        schedule = cosine_schedule(optimizer, self.optim.warmup_steps, total_steps)
        logger.info(
            "%s: %d train windows, %d steps, %.2fM trainable parameters",
            self.cfg["experiment"],
            len(self.train_set),
            total_steps,
            sum(p.numel() for p in self.params) / 1e6,
        )
        history, best, step, start = [], math.inf, 0, time.time()
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1, self.optim.epochs + 1):
            self.set_train_mode()
            for batch in loader:
                batch = to_device(batch, self.device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.optim.bf16):
                    terms = self.step_terms(batch)
                loss, logs = total_loss({k: v.float() for k, v in terms.items()}, self.loss_weights)
                if not math.isfinite(logs["total"]):
                    raise FloatingPointError(f"Nonfinite loss at step {step}: {logs}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, self.optim.grad_clip)
                optimizer.step()
                schedule.step()
                step += 1
                if step % self.optim.log_every == 0:
                    logger.info(
                        "epoch %d step %d/%d lr %.2e %s",
                        epoch,
                        step,
                        total_steps,
                        schedule.get_last_lr()[0],
                        " ".join(f"{k} {v:.4f}" for k, v in logs.items()),
                    )
            metrics = self.validate() | {"epoch": epoch, "step": step}
            history.append(metrics)
            logger.info(
                "epoch %d val L2 %.3f col %.3f (single) | L2 %.3f col %.3f (cumulative)",
                epoch,
                metrics["l2_avg_single"],
                metrics["box_col_avg_single"],
                metrics["l2_avg"],
                metrics["box_col_avg"],
            )
            self.save("last", epoch, metrics)
            if metrics["l2_avg_single"] < best:
                best = metrics["l2_avg_single"]
                self.save("best", epoch, metrics)
        summary = {
            "experiment": self.cfg["experiment"],
            "stage": self.stage,
            "best_val": min(history, key=lambda m: m["l2_avg_single"]),
            "final_val": history[-1],
            "history": history,
            "train_hours": (time.time() - start) / 3600,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "optim": asdict(self.optim),
            "git": git_revision(self.root),
            "environment": environment(),
        }
        write_json(self.out / "summary.json", summary)
        return summary
