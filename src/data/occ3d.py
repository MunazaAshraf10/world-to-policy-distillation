import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pyquaternion import Quaternion

from src.world.types import Rollout

type Scenes = dict[str, list[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class Window:
    """A 12-frame OccWorld window; the planning origin is the last observed frame."""

    scene: str
    start: int

    def frame(self, offset: int) -> int:
        return self.start + offset


def load_scenes(path: Path) -> Scenes:
    with path.open("rb") as f:
        return pickle.load(f)["infos"]


def list_windows(scenes: Scenes, rollout: Rollout = Rollout()) -> list[Window]:
    """Matches upstream's traversal count, which omits each scene's final window."""
    return [
        Window(scene, start)
        for scene, frames in scenes.items()
        for start in range(len(frames) - rollout.window)
    ]


def occupancy_path(data_root: Path, scene: str, token: str) -> Path:
    return data_root / "gts" / scene / token / "labels.npz"


def lidar_to_ego(frame: dict[str, Any]) -> np.ndarray:
    """Planar [2, 3] transform from the LiDAR frame of the trajectories to the Occ3D ego frame."""
    rot = Quaternion(frame["lidar2ego_rotation"]).rotation_matrix[:2, :2]
    trans = np.asarray(frame["lidar2ego_translation"])[:2]
    return np.concatenate([rot, trans[:, None]], axis=1).astype(np.float32)


def window_record(frames: list[dict[str, Any]], rollout: Rollout) -> dict[str, np.ndarray]:
    """Per-window planning inputs and targets, keyed for array stacking."""
    origin = frames[rollout.history - 1]
    return {
        # Upstream conditioning: each frame's next-step displacement and command.
        "rel_poses": np.stack([f["gt_ego_fut_trajs"][0] for f in frames]).astype(np.float32),
        "modes": np.stack([f["pose_mode"] for f in frames]).astype(np.float32),
        "target": np.asarray(origin["gt_ego_fut_trajs"], dtype=np.float32),
        "target_mask": np.asarray(origin["gt_ego_fut_masks"], dtype=np.float32),
        "command": np.asarray(origin["gt_ego_fut_cmd"], dtype=np.float32),
        "ego_hist": np.asarray(origin["gt_ego_his_trajs"], dtype=np.float32),
        "lidar2ego": lidar_to_ego(origin),
    }


def agent_annotations(frame: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Boxes [N, 9] and attributes [N, 34] laid out as upstream's planning metric expects."""
    mask = frame["valid_flag"]
    velocity = np.nan_to_num(frame["gt_velocity"][mask], nan=0.0)
    boxes = np.concatenate([frame["gt_boxes"][mask], velocity], axis=-1).astype(np.float32)
    attrs = np.concatenate(
        [
            frame["gt_agent_fut_trajs"][mask],
            frame["gt_agent_fut_masks"][mask],
            frame["gt_agent_fut_goal"][mask][..., None],
            frame["gt_agent_lcf_feat"][mask],
            frame["gt_agent_fut_yaw"][mask],
        ],
        axis=-1,
    ).astype(np.float32)
    return boxes, attrs
