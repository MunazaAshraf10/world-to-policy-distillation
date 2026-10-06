import importlib
import io
import sys
import types
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Self

import numpy as np
import torch
from mmengine.config import Config
from torch import nn

from src.world.types import Rollout, WorldPrediction


class InstanceBoxes:
    """Stands in for mmdet3d LiDARInstance3DBoxes; upstream only reads the raw tensor."""

    def __init__(self, tensor: np.ndarray | torch.Tensor, box_dim: int = 9, **kwargs: Any) -> None:
        self.tensor = torch.as_tensor(np.asarray(tensor, dtype=np.float32))

    def convert_to(self, mode: Any) -> Self:
        return self


def stub_module(name: str, **attrs: Any) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def install_shims() -> None:
    """Replace imports that upstream uses only for typing or unused helpers.

    The pinned OccWorld imports mmdet3d boxes in its dataset and a nuScenes Box in its
    planning metric. Neither affects world model inference, and both pull in compiled
    operators that have no Blackwell builds.
    """
    shims = {
        "mmdet3d": {},
        "mmdet3d.structures": {},
        "mmdet3d.structures.bbox_3d": {
            "LiDARInstance3DBoxes": InstanceBoxes,
            "Box3DMode": types.SimpleNamespace(LIDAR=0),
        },
        "nuscenes": {},
        "nuscenes.utils": {},
        "nuscenes.utils.data_classes": {"Box": object},
    }
    for name, attrs in shims.items():
        sys.modules.setdefault(name, stub_module(name, **attrs))


@contextmanager
def upstream_imports(root: Path) -> Iterator[None]:
    """Scope upstream's absolute imports (model, dataset, loss, utils) to its checkout."""
    root = root.resolve()
    if not (root / "model/TransVQVAE.py").is_file():
        raise FileNotFoundError("Initialize the pinned dependency: git submodule update --init")
    for name in ("model", "dataset", "loss", "utils"):
        module = sys.modules.get(name)
        if module is not None and root not in Path(module.__file__ or "").resolve().parents:
            raise RuntimeError(f"Package {name} is already imported from {module.__file__}")
    install_shims()
    previous = sys.path.copy()
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = previous


def upstream_config(root: Path, name: str) -> Config:
    return Config.fromfile(str(root / "config" / name))


def build_upstream(root: Path, model_cfg: Mapping[str, Any]) -> nn.Module:
    with upstream_imports(root):
        importlib.import_module("model")
        from mmengine.registry import MODELS

        # Upstream prints its attention layout while constructing the VAE.
        with redirect_stdout(io.StringIO()):
            model = MODELS.build(dict(model_cfg))
    return model


def read_state(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{path} does not contain a state dictionary")
    prefixed = [key.startswith("module.") for key in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError("Checkpoint has mixed module. prefixes")
    if all(prefixed):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return dict(state)


def load_strict(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Require every parameter and buffer, with matching shapes and finite values."""
    expected = model.state_dict()
    missing = sorted(expected.keys() - state.keys())
    unexpected = sorted(state.keys() - expected.keys())
    mismatched = sorted(
        key for key in expected.keys() & state.keys() if state[key].shape != expected[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f"Incompatible checkpoint: missing={missing}, unexpected={unexpected}, "
            f"mismatched={mismatched}"
        )
    nonfinite = [key for key, value in state.items() if not torch.isfinite(value).all()]
    if nonfinite:
        raise ValueError(f"Checkpoint tensors contain nonfinite values: {nonfinite}")
    model.load_state_dict(state, strict=True)


def build_world_model(root: Path, checkpoint: Path, device: torch.device) -> nn.Module:
    model = build_upstream(root, upstream_config(root, "occworld.py").model)
    load_strict(model, read_state(checkpoint))
    return model.to(device)


class OccWorldAdapter:
    """Frozen OccWorld rollout with history-only inputs and project level outputs."""

    def __init__(self, model: nn.Module, rollout: Rollout = Rollout()) -> None:
        self.model = model.requires_grad_(False).eval()
        self.rollout = rollout

    def assert_frozen(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("OccWorld must remain in evaluation mode")
        if any(p.requires_grad or p.grad is not None for p in self.model.parameters()):
            raise RuntimeError("OccWorld parameters must be frozen with no gradients")

    @property
    def codebook(self) -> torch.Tensor:
        """Frozen VQ embeddings [512, 128]; predicted codes index into these."""
        return self.model.vae.vqvae.embedding.weight

    @torch.no_grad()
    def predict(
        self, history: torch.Tensor, history_disp: torch.Tensor, modes: torch.Tensor
    ) -> WorldPrediction:
        """Roll out the future from observed frames only.

        history: [B, H, 200, 200, 16] int64 labels for the H observed frames.
        history_disp: [B, H, 2] per-step ego displacement of each observed frame.
        modes: [B, H + F, 3] one-hot driving command, the conditioning upstream uses.
        """
        r = self.rollout
        bs = history.shape[0]
        if history.shape[1:] != (r.history, *r.grid) or history.dtype != torch.int64:
            raise ValueError(f"Unexpected history {history.dtype} {tuple(history.shape)}")
        if modes.shape[1:] != (r.history + r.future, 3):
            raise ValueError(f"Unexpected modes {tuple(modes.shape)}")
        self.assert_frozen()

        # Upstream encodes a full window but seeds autoregression only from the history
        # codes, so padding with the last observation keeps future labels out entirely.
        pad = history[:, -1:].expand(-1, r.window - r.history, -1, -1, -1)
        window = torch.cat([history, pad], dim=1)
        rel_poses = torch.zeros(bs, r.window, 2, dtype=torch.float64)
        rel_poses[:, : r.history] = history_disp.double().cpu()
        gt_mode = torch.zeros(bs, r.window, 3, dtype=torch.float64)
        gt_mode[:, : r.history + r.future] = modes.double().cpu()
        metas = [
            {"rel_poses": rel_poses[i].numpy(), "gt_mode": gt_mode[i].numpy()} for i in range(bs)
        ]

        with torch.cuda.device(history.device):
            output = self.model.forward_autoreg_with_pose(
                window, metas, start_frame=0, mid_frame=r.history, end_frame=r.history + r.future
            )
        codes = output["ce_inputs"].argmax(dim=1).view(bs, r.future, *r.latent)
        prediction = WorldPrediction(
            codes=codes,
            occupancy=output["sem_pred"],
            ego_disp=output["poses_"].float(),
            ego_disp_modes=output["pose_decoded"].float(),
        )
        prediction.validate(r)
        self.assert_frozen()
        return prediction
