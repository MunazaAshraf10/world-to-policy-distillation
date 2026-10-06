import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.cache import (
    PlanningCache,
    build_frames,
    build_occupancy,
    build_windows,
    open_world,
    pack_bev,
)
from src.data.occ3d import load_scenes
from src.reward.targets import bev_masks
from src.utils.config import load_config
from src.utils.repro import sha256, write_json
from src.world.occworld import OccWorldAdapter, build_world_model

logger = logging.getLogger("build_cache")


def build_inputs(paths: dict[str, str], workers: int) -> None:
    cache = Path(paths["cache"])
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for split in ("train", "val"):
        infos = Path(paths[f"infos_{split}"])
        scenes = load_scenes(infos)
        build_occupancy(scenes, Path(paths["data_root"]), cache / f"{split}_occupancy.npy", workers)
        build_frames(scenes, cache / f"{split}_frames.npz")
        windows = build_windows(scenes, cache / f"{split}_windows.npz")
        manifest[split] = {"infos": str(infos), "infos_sha256": sha256(infos), "windows": windows}
    write_json(cache / "inputs.json", manifest)


@torch.no_grad()
def build_world(paths: dict[str, str], checkpoint: Path, batch_size: int) -> None:
    """Roll out the frozen world model once per window; argmax decoding makes this exact."""
    cache = Path(paths["cache"])
    device = torch.device("cuda")
    adapter = OccWorldAdapter(build_world_model(Path(paths["occworld_root"]), checkpoint, device))
    torch.save(adapter.codebook.detach().cpu().clone(), cache / "codebook.pt")
    for split in ("train", "val"):
        data = PlanningCache(cache, split)
        world = open_world(cache, split, count=len(data))
        loader = DataLoader(data, batch_size=batch_size, num_workers=8, pin_memory=True)
        for batch in tqdm(loader, desc=f"world {split}"):
            index = batch["index"].numpy()
            pred = adapter.predict(
                batch["occ"].to(device).long(),
                batch["rel_poses"].to(device),
                batch["modes"].to(device),
            )
            world["codes"][index] = pred.codes.cpu().numpy().astype(np.uint16)
            world["bev"][index] = pack_bev(bev_masks(pred.occupancy)).cpu().numpy()
            world["ego_disp"][index] = pred.ego_disp.cpu().numpy()
        for array in world.values():
            array.flush()
    write_json(cache / "world.json", {"checkpoint": str(checkpoint), "sha256": sha256(checkpoint)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute input and frozen world caches")
    parser.add_argument("target", choices=("inputs", "world"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--checkpoint", type=Path, help="trained OccWorld state dict")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    paths = load_config(args.config)["paths"]
    if args.target == "inputs":
        build_inputs(paths, args.workers)
    else:
        if args.checkpoint is None:
            parser.error("world caching requires --checkpoint")
        build_world(paths, args.checkpoint, args.batch_size)


if __name__ == "__main__":
    main()
