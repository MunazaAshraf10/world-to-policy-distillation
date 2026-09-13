from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.world.occworld import upstream_imports
from src.world.types import Rollout


@dataclass(frozen=True)
class OccupancySample:
    sequence: torch.Tensor
    metadata: list[dict[str, Any]]
    identity: dict[str, Any]


def select_window(
    scenes: dict[str, list[dict[str, Any]]],
    index: int,
    sequence_length: int,
) -> tuple[str, int, list[dict[str, Any]]]:
    """Preserve the official traversal count, including its omitted final window."""
    lengths = {scene: len(frames) - sequence_length for scene, frames in scenes.items()}
    if any(length < 0 for length in lengths.values()):
        raise ValueError("Temporal metadata contains a scene shorter than the official window")
    if index < 0 or index >= sum(lengths.values()):
        raise IndexError(f"Sample index {index} is outside the official traversal")
    offset = index
    for scene, length in lengths.items():
        if offset < length:
            return scene, offset, scenes[scene][offset : offset + sequence_length]
        offset -= length
    raise RuntimeError("Inconsistent temporal scene lengths")


def load_sample(
    root: Path,
    data_root: Path,
    infos: Path,
    index: int,
    rollout: Rollout,
) -> OccupancySample:
    """Use upstream traversal, metadata construction, wrapper, and collation."""
    with upstream_imports(root):
        from dataset import OPENOCC_DATASET, OPENOCC_DATAWRAPPER
        from dataset.dataset_wrapper import custom_collate_fn_temporal

        dataset = OPENOCC_DATASET.build(
            dict(
                type="nuScenesSceneDatasetLidarTraverse",
                data_path=str(data_root),
                return_len=rollout.sequence_length,
                offset=0,
                imageset=str(infos),
                test_mode=True,
                input_dataset="gts",
                output_dataset="gts",
            )
        )
        scene, offset, frames = select_window(dataset.nusc_infos, index, rollout.sequence_length)
        for frame in frames:
            path = data_root / "gts" / scene / frame["token"] / "labels.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing Occ3D occupancy: {path}")
        wrapper = OPENOCC_DATAWRAPPER.build(
            dict(type="tpvformer_dataset_nuscenes", phase="val"),
            default_args={"in_dataset": dataset},
        )
        sequence, target, metadata = custom_collate_fn_temporal([wrapper[index]])
    identity = {
        "dataset_version": "v1.0-trainval",
        "split": "val",
        "dataset_index": index,
        "scene_name": scene,
        "window_start_index": offset,
        "window_tokens": [frame["token"] for frame in frames],
        "history_last_token": frames[rollout.mid_frame - 1]["token"],
        "future_tokens": [
            frame["token"] for frame in frames[rollout.mid_frame : rollout.end_frame]
        ],
        "upstream_sample_idx": metadata[0]["sample_idx"],
    }
    return OccupancySample(sequence, metadata, identity)
