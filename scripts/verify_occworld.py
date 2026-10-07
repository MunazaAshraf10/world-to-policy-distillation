import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.cache import PlanningCache, open_world, pack_bev
from src.reward.targets import bev_masks
from src.utils.config import load_config
from src.utils.repro import environment, git_revision, sha256, write_json
from src.world.occworld import OccWorldAdapter, build_world_model
from src.world.types import Rollout

logger = logging.getLogger("verify_occworld")


@torch.no_grad()
def verify(paths: dict[str, str], checkpoint: Path, windows: int) -> dict[str, Any]:
    """Stage 1: strict load, frozen parameters, finite outputs, determinism, cache parity."""
    device = torch.device("cuda")
    cache = Path(paths["cache"])
    adapter = OccWorldAdapter(build_world_model(Path(paths["occworld_root"]), checkpoint, device))
    adapter.assert_frozen()
    data = PlanningCache(cache, "val", limit=windows)
    world = open_world(cache, "val") if (cache / "val_world_codes.npy").is_file() else None
    batch = {k: torch.stack([data[i][k] for i in range(windows)]) for k in data[0]}
    inputs = (
        batch["occ"].to(device).long(),
        batch["rel_poses"].to(device),
        batch["modes"].to(device),
    )
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    first = adapter.predict(*inputs)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    second = adapter.predict(*inputs)
    checks = {
        "frozen": not any(p.requires_grad for p in adapter.model.parameters()),
        "eval_mode": not any(m.training for m in adapter.model.modules()),
        "finite": bool(torch.isfinite(first.ego_disp).all()),
        "deterministic": torch.equal(first.codes, second.codes)
        and torch.equal(first.occupancy, second.occupancy),
    }
    if world is not None:
        index = batch["index"].numpy()
        bev = pack_bev(bev_masks(first.occupancy)).cpu().numpy()
        checks["cache_matches_live"] = bool(
            np.array_equal(world["codes"][index], first.codes.cpu().numpy())
            and np.array_equal(world["bev"][index], bev)
        )
    rollout = Rollout()
    report = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "parameters": sum(p.numel() for p in adapter.model.parameters()),
        "windows": windows,
        "shapes": {
            "codes": list(first.codes.shape),
            "occupancy": list(first.occupancy.shape),
            "ego_disp": list(first.ego_disp.shape),
        },
        "rollout": {"history": rollout.history, "future": rollout.future},
        "seconds_per_window": elapsed / windows,
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "occupied_fraction": float((first.occupancy != 17).float().mean()),
        "git": git_revision(Path(__file__).resolve().parents[1]),
        "environment": environment(),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the frozen OccWorld integration")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    # Rollouts match the cache only under its batch composition; build_cache batches 16.
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--output", type=Path, default=Path("results/occworld_verify.json"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = verify(load_config(args.config)["paths"], args.checkpoint, args.windows)
    write_json(args.output, report)
    logger.info("%s %s -> %s", report["status"], report["checks"], args.output)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
