import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.occ3d import (
    Scenes,
    agent_annotations,
    list_windows,
    occupancy_path,
    window_record,
)
from src.world.types import Rollout

logger = logging.getLogger(__name__)

WORLD_FIELDS = ("codes", "bev", "ego_disp")


def frame_rows(scenes: Scenes) -> dict[str, int]:
    tokens = [frame["token"] for frames in scenes.values() for frame in frames]
    return {token: row for row, token in enumerate(tokens)}


def write_occupancy_rows(
    out: Path, data_root: Path, jobs: list[tuple[int, str, str]], shape: tuple[int, ...]
) -> list[int]:
    """Write rows, then read each back and return those that differ from the source."""
    occ = np.lib.format.open_memmap(out, mode="r+")
    if occ.shape[1:] != shape:
        raise ValueError(f"Memmap {out} has shape {occ.shape}, expected [N, *{shape}]")
    sources = {}
    for row, scene, token in jobs:
        sources[row] = np.load(occupancy_path(data_root, scene, token))["semantics"]
        occ[row] = sources[row].astype(np.uint8)
    occ.flush()
    return [row for row, source in sources.items() if not np.array_equal(occ[row], source)]


def build_occupancy(
    scenes: Scenes, data_root: Path, out: Path, workers: int, rollout: Rollout = Rollout()
) -> None:
    """Decode every Occ3D frame of a split once into a uint8 memmap [N, 200, 200, 16].

    Every row is verified against its source after writing and rewritten once on mismatch.
    """
    rows = frame_rows(scenes)
    jobs = [
        (rows[frame["token"]], scene, frame["token"])
        for scene, frames in scenes.items()
        for frame in frames
    ]
    missing = [p for _, s, t in jobs if not (p := occupancy_path(data_root, s, t)).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} Occ3D frames missing, first: {missing[0]}")
    if not out.is_file():
        np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(jobs), *rollout.grid))
    total = len(jobs)
    for attempt in range(2):
        chunks = [jobs[i::workers] for i in range(workers)]
        with ProcessPoolExecutor(workers) as pool:
            futures = [
                pool.submit(write_occupancy_rows, out, data_root, chunk, rollout.grid)
                for chunk in chunks
            ]
            bad = {row for future in futures for row in future.result()}
        if not bad:
            logger.info("Wrote and verified %d occupancy frames in %s", total, out)
            return
        logger.warning("%d rows differ from their source on attempt %d", len(bad), attempt + 1)
        jobs = [job for job in jobs if job[0] in bad]
    raise OSError(f"Occupancy rows {sorted(bad)} could not be written faithfully to {out}")


def build_frames(scenes: Scenes, out: Path) -> None:
    """Per-frame conditioning in upstream form, plus scene row bounds for window sampling."""
    rel_pose, mode, bounds = [], [], []
    for frames in scenes.values():
        bounds.append((len(rel_pose), len(rel_pose) + len(frames)))
        rel_pose.extend(f["gt_ego_fut_trajs"][0] for f in frames)
        mode.extend(f["pose_mode"] for f in frames)
    np.savez(
        out,
        rel_pose=np.asarray(rel_pose, dtype=np.float32),
        mode=np.asarray(mode, dtype=np.float32),
        bounds=np.asarray(bounds, dtype=np.int64),
    )


def build_windows(scenes: Scenes, out: Path, rollout: Rollout = Rollout()) -> int:
    """Stack per-window planning inputs, targets, and evaluation annotations."""
    rows = frame_rows(scenes)
    records, boxes, attrs = [], [], []
    for window in list_windows(scenes, rollout):
        frames = scenes[window.scene][window.start : window.start + rollout.window]
        record = window_record(frames, rollout)
        record["frames"] = np.array([rows[f["token"]] for f in frames], dtype=np.int64)
        records.append(record)
        box, attr = agent_annotations(frames[rollout.history - 1])
        boxes.append(box)
        attrs.append(attr)
    arrays = {key: np.stack([r[key] for r in records]) for key in records[0]}
    counts = np.array([len(b) for b in boxes], dtype=np.int64)
    np.savez(
        out,
        **arrays,
        box_offsets=np.concatenate([[0], np.cumsum(counts)]),
        boxes=np.concatenate(boxes),
        attrs=np.concatenate(attrs),
    )
    logger.info("Wrote %d windows to %s", len(records), out)
    return len(records)


def open_world(
    root: Path, split: str, count: int | None = None, rollout: Rollout = Rollout()
) -> dict[str, np.ndarray]:
    """World cache memmaps; pass count to create them for writing."""
    specs = {
        "codes": (np.uint16, (rollout.future, *rollout.latent)),
        # Obstacle and drivable masks per predicted step, bit-packed along the last grid axis.
        "bev": (np.uint8, (rollout.future, 2, rollout.grid[0], rollout.grid[1] // 8)),
        "ego_disp": (np.float32, (rollout.future, 2)),
    }
    arrays = {}
    for name, (dtype, shape) in specs.items():
        path = root / f"{split}_world_{name}.npy"
        if count is None:
            arrays[name] = np.load(path, mmap_mode="r")
        else:
            arrays[name] = np.lib.format.open_memmap(
                path, mode="w+", dtype=dtype, shape=(count, *shape)
            )
    return arrays


class PlanningCache(Dataset):
    """Training and evaluation samples served from the precomputed split caches."""

    def __init__(
        self,
        root: Path,
        split: str,
        world: bool = False,
        limit: int | None = None,
        rollout: Rollout = Rollout(),
    ) -> None:
        self.rollout = rollout
        self.occ = np.load(root / f"{split}_occupancy.npy", mmap_mode="r")
        with np.load(root / f"{split}_windows.npz") as data:
            self.windows = {key: data[key] for key in data.files}
        self.world = open_world(root, split) if world else None
        self.size = len(self.windows["frames"]) if limit is None else limit

    def __len__(self) -> int:
        return self.size

    def annotations(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        lo, hi = self.windows["box_offsets"][index : index + 2]
        return self.windows["boxes"][lo:hi], self.windows["attrs"][lo:hi]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        w = self.windows
        h = self.rollout.history
        rows = w["frames"][index, :h]
        item = {
            "occ": torch.from_numpy(np.array(self.occ[rows])),
            "rel_poses": torch.from_numpy(w["rel_poses"][index, :h]),
            "modes": torch.from_numpy(w["modes"][index, : h + self.rollout.future]),
            "target": torch.from_numpy(w["target"][index]),
            "target_mask": torch.from_numpy(w["target_mask"][index]),
            "command": torch.from_numpy(w["command"][index]),
            "ego_hist": torch.from_numpy(w["ego_hist"][index]),
            "lidar2ego": torch.from_numpy(w["lidar2ego"][index]),
            "index": torch.tensor(index),
        }
        if self.world is not None:
            for name in WORLD_FIELDS:
                item[name] = torch.from_numpy(np.array(self.world[name][index]))
        return item


def pack_bev(masks: torch.Tensor) -> torch.Tensor:
    """[..., H, W] bool to [..., H, W / 8] uint8, bit compatible with np.packbits."""
    bits = 1 << torch.arange(7, -1, -1, device=masks.device, dtype=torch.uint8)
    grouped = masks.unflatten(-1, (-1, 8)).to(torch.uint8)
    return (grouped * bits).sum(-1, dtype=torch.uint8)


def unpack_bev(packed: torch.Tensor) -> torch.Tensor:
    """[..., H, W / 8] uint8 to [..., H, W] bool, inverting np.packbits along the last axis."""
    bits = 1 << torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    unpacked = (packed.unsqueeze(-1) & bits) > 0
    return unpacked.flatten(-2)
