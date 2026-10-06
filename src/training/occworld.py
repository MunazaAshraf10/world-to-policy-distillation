import importlib
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
from torch.utils.data import DataLoader, Dataset

from src.data.cache import PlanningCache
from src.evaluation.forecast import ForecastMeter
from src.training.trainer import cosine_schedule
from src.utils.repro import environment, git_revision, seed_everything, write_json
from src.world.occworld import (
    OccWorldAdapter,
    build_upstream,
    load_strict,
    read_state,
    upstream_config,
    upstream_imports,
)
from src.world.types import Rollout

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WorldTrainConfig:
    """Upstream optimizer settings with a step budget sized for one GPU."""

    stage: str = "vqvae"
    steps: int = 40000
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 200
    grad_clip: float = 35.0
    bf16: bool = True
    compile: bool = True
    workers: int = 8
    log_every: int = 100
    eval_every: int = 5000
    eval_windows: int = 500
    vqvae_checkpoint: str | None = None


class FrameSet(Dataset):
    """Single Occ3D frames for the VQVAE, shaped [1, X, Y, Z] as one-frame sequences."""

    def __init__(self, cache: Path, split: str) -> None:
        self.occ = np.load(cache / f"{split}_occupancy.npy", mmap_mode="r")

    def __len__(self) -> int:
        return len(self.occ)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"occ": torch.from_numpy(np.array(self.occ[index : index + 1]))}


class SequenceSet(Dataset):
    """Every contiguous window of length num_frames + offset inside a training scene."""

    def __init__(self, cache: Path, split: str, length: int) -> None:
        self.occ = np.load(cache / f"{split}_occupancy.npy", mmap_mode="r")
        with np.load(cache / f"{split}_frames.npz") as data:
            self.rel_pose, self.mode = data["rel_pose"], data["mode"]
            bounds = data["bounds"]
        self.length = length
        self.starts = np.concatenate([np.arange(lo, hi - length + 1) for lo, hi in bounds])

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rows = slice(self.starts[index], self.starts[index] + self.length)
        return {
            "occ": torch.from_numpy(np.array(self.occ[rows])),
            "rel_poses": torch.from_numpy(self.rel_pose[rows]),
            "gt_mode": torch.from_numpy(self.mode[rows]),
        }


def build_loss(root: Path, loss_cfg: dict[str, Any]) -> nn.Module:
    with upstream_imports(root):
        loss = importlib.import_module("loss")
        return loss.OPENOCC_LOSS.build(loss_cfg)


class WorldTrainer:
    """Retrains OccWorld in its two upstream stages: the VQVAE, then the world transformer."""

    def __init__(self, cfg: dict[str, Any], root: Path) -> None:
        self.cfg = cfg
        self.root = root
        self.train_cfg = WorldTrainConfig(**cfg["world"])
        self.occworld_root = Path(cfg["paths"]["occworld_root"])
        self.cache = Path(cfg["paths"]["cache"])
        self.out = Path(cfg["paths"]["checkpoints"]) / cfg["experiment"]
        self.device = torch.device("cuda")
        seed_everything(int(cfg["seed"]))

        stage = self.train_cfg.stage
        name = {"vqvae": "train_vqvae.py", "transformer": "train_occworld.py"}[stage]
        self.upstream = upstream_config(self.occworld_root, name)
        self.model = build_upstream(self.occworld_root, self.upstream.model).to(self.device)
        self.loss = build_loss(self.occworld_root, self.upstream.loss).to(self.device)
        if stage == "vqvae":
            self.data = FrameSet(self.cache, "train")
        else:
            # Upstream trains with return_len_train + 1 frames and freezes the VQVAE.
            length = self.upstream.return_len_train + 1
            self.data = SequenceSet(self.cache, "train", length)
            vae = {
                k.removeprefix("vae."): v
                for k, v in read_state(Path(self.train_cfg.vqvae_checkpoint)).items()
            }
            load_strict(self.model.vae, vae)
            self.model.vae.requires_grad_(False)
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        # The compiled module shares parameters with self.model, used for eval and saving.
        self.train_model = torch.compile(self.model) if self.train_cfg.compile else self.model

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        occ = batch["occ"].to(self.device, non_blocking=True).long()
        inputs = {"inputs": occ}
        if self.train_cfg.stage == "vqvae":
            out = self.train_model(occ)
        else:
            metas = [
                {"rel_poses": r.numpy().astype(np.float64), "gt_mode": m.numpy().astype(np.float64)}
                for r, m in zip(batch["rel_poses"], batch["gt_mode"], strict=True)
            ]
            out = self.train_model(occ, metas=metas)
        for key, source in self.upstream.loss_input_convertion.items():
            inputs[key] = out[source]
        return self.loss(inputs)

    def state(self) -> dict[str, torch.Tensor]:
        if self.train_cfg.stage == "vqvae":
            return {f"vae.{k}": v for k, v in self.model.state_dict().items()}
        return self.model.state_dict()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """VQVAE reconstruction mIoU, or rollout forecast mIoU/IoU under the official protocol."""
        self.model.eval()
        for p in self.params:
            p.grad = None
        meter = ForecastMeter()
        val = PlanningCache(self.cache, "val", limit=self.train_cfg.eval_windows)
        loader = DataLoader(val, batch_size=4, num_workers=4)
        rollout = Rollout()
        adapter = OccWorldAdapter(self.model) if self.train_cfg.stage != "vqvae" else None
        for batch in loader:
            rows = val.windows["frames"][batch["index"].numpy()]
            future = torch.from_numpy(
                val.occ[rows[:, rollout.history : rollout.history + rollout.future]]
            )
            if adapter is None:
                occ = future.to(self.device).long()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.train_cfg.bf16):
                    pred = self.model(occ)["logits"].argmax(-1)
            else:
                pred = adapter.predict(
                    batch["occ"].to(self.device).long(),
                    batch["rel_poses"].to(self.device),
                    batch["modes"].to(self.device),
                ).occupancy
            meter.update(pred.cpu(), future.long())
        if adapter is not None:
            self.model.requires_grad_(False)
            for p in self.params:
                p.requires_grad_(True)
        self.model.train()
        return meter.summary()

    def run(self) -> dict[str, Any]:
        tc = self.train_cfg
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "config.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))
        loader = DataLoader(
            self.data,
            batch_size=tc.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=tc.workers,
            pin_memory=True,
            persistent_workers=True,
        )
        optimizer = torch.optim.AdamW(self.params, lr=tc.lr, weight_decay=tc.weight_decay)
        schedule = cosine_schedule(optimizer, tc.warmup_steps, tc.steps)
        logger.info(
            "OccWorld %s: %d samples, %d steps of batch %d, %.2fM trainable parameters",
            tc.stage,
            len(self.data),
            tc.steps,
            tc.batch_size,
            sum(p.numel() for p in self.params) / 1e6,
        )
        self.model.train()
        torch.cuda.reset_peak_memory_stats()
        step, start, history = 0, time.time(), []
        while step < tc.steps:
            for batch in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=tc.bf16):
                    loss, parts = self.forward(batch)
                if not math.isfinite(loss.item()):
                    raise FloatingPointError(f"Nonfinite loss at step {step}: {parts}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.params, tc.grad_clip)
                optimizer.step()
                schedule.step()
                step += 1
                if step % tc.log_every == 0:
                    rate = step * tc.batch_size / (time.time() - start)
                    logger.info(
                        "step %d/%d lr %.2e loss %.4f %s | %.1f samples/s",
                        step,
                        tc.steps,
                        schedule.get_last_lr()[0],
                        loss.item(),
                        " ".join(f"{k} {v:.4f}" for k, v in parts.items()),
                        rate,
                    )
                if step % tc.eval_every == 0 or step == tc.steps:
                    metrics = self.evaluate() | {"step": step}
                    history.append(metrics)
                    logger.info("eval %s", metrics)
                    torch.save({"state_dict": self.state(), "step": step}, self.out / "last.pt")
                if step == tc.steps:
                    break
        summary = {
            "experiment": self.cfg["experiment"],
            "world": asdict(tc),
            "history": history,
            "train_hours": (time.time() - start) / 3600,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
            "git": git_revision(self.root),
            "environment": environment(),
        }
        write_json(self.out / "summary.json", summary)
        return summary
