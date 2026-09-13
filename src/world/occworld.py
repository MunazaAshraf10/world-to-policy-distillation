from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn

from src.world.types import Rollout, WorldPrediction


@contextmanager
def upstream_imports(root: Path) -> Iterator[None]:
    """Scope upstream's absolute imports and reject conflicting cached packages."""
    root = root.resolve()
    if not (root / "model/TransVQVAE.py").is_file():
        raise FileNotFoundError(
            "Initialize the pinned dependency: git submodule update --init OccWorld"
        )
    for name in ("model", "dataset", "utils"):
        module = sys.modules.get(name)
        if module is not None:
            location = Path(module.__file__ or "").resolve()
            if root not in location.parents:
                raise RuntimeError(
                    f"Package {name} is already imported from {location}; use a fresh process"
                )
    previous = sys.path.copy()
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = previous


def load_checkpoint(model: nn.Module, path: Path) -> None:
    """Require all parameters and buffers, including transformer and pose modules."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Checkpoint must be a state dictionary or contain state_dict")
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Checkpoint state_dict must be a nonempty mapping")
    if not all(isinstance(key, str) for key in state):
        raise ValueError("Checkpoint keys must be strings")
    prefixes = [key.startswith("module.") for key in state]
    if any(prefixes) and not all(prefixes):
        raise ValueError("Checkpoint has mixed module. prefixes")
    if all(prefixes):
        state = {key[7:]: value for key, value in state.items()}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    invalid = [
        key
        for key in expected.keys() & state.keys()
        if not isinstance(state[key], torch.Tensor) or state[key].shape != expected[key].shape
    ]
    if missing or unexpected or invalid:
        raise ValueError(
            f"Full OccWorld checkpoint required; missing={missing}, "
            f"unexpected={unexpected}, incompatible_shapes={sorted(invalid)}"
        )
    for key, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Checkpoint tensor {key} contains nonfinite values")
    model.load_state_dict(state, strict=True)


def build_model(
    root: Path, model_config: dict[str, Any], checkpoint: Path, device: torch.device
) -> nn.Module:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Pinned OccWorld requires CUDA; CPU inference is not supported")
    with upstream_imports(root), torch.cuda.device(device):
        importlib.import_module("model")
        from mmengine.registry import MODELS

        model = MODELS.build(model_config)
        model.init_weights()
        load_checkpoint(model, checkpoint)
        return model.to(device)


def validate_sequence(
    sequence: torch.Tensor, metadata: list[dict[str, Any]], rollout: Rollout
) -> None:
    expected = (1, rollout.sequence_length, *rollout.grid)
    if tuple(sequence.shape) != expected or sequence.dtype != torch.int64:
        raise ValueError(
            f"Occupancy must be int64 with shape {expected}, got {sequence.dtype} {tuple(sequence.shape)}"
        )
    if sequence.min() < 0 or sequence.max() >= rollout.classes:
        raise ValueError(f"Occupancy labels must be in [0, {rollout.classes - 1}]")
    if len(metadata) != 1:
        raise ValueError("Stage 1 requires exactly one metadata record")
    for key, width in (("rel_poses", 2), ("gt_mode", rollout.modes)):
        values = np.asarray(metadata[0][key])
        if values.shape != (rollout.sequence_length, width) or not np.isfinite(values).all():
            raise ValueError(f"Invalid {key}: expected finite [{rollout.sequence_length}, {width}]")
    modes = np.asarray(metadata[0]["gt_mode"])
    if not np.isin(modes, [0, 1]).all() or not (modes.sum(axis=-1) == 1).all():
        raise ValueError("gt_mode must be one-hot at every temporal step")


def normalize_prediction(output: dict[str, Any], rollout: Rollout) -> WorldPrediction:
    prediction = WorldPrediction(
        semantic_labels=output["sem_pred"],
        semantic_logits=output["logits"],
        ego_displacement_modes=output["pose_decoded"],
        selected_ego_displacements=output["poses_"],
    )
    shape = (1, rollout.future_steps)
    expected = {
        "semantic_labels": (*shape, *rollout.grid),
        "semantic_logits": (*shape, *rollout.grid, rollout.classes),
        "ego_displacement_modes": (*shape, rollout.modes, 2),
        "selected_ego_displacements": (*shape, 2),
    }
    for name, tensor in prediction.tensors().items():
        if tuple(tensor.shape) != expected[name]:
            raise ValueError(f"{name}: expected {expected[name]}, got {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains nonfinite values")
        if tensor.requires_grad:
            raise ValueError(f"Frozen prediction {name} unexpectedly requires gradients")
        if name != "semantic_labels" and not tensor.is_floating_point():
            raise ValueError(f"{name} must be floating point")
    labels = prediction.semantic_labels
    if labels.dtype != torch.int64 or labels.min() < 0 or labels.max() >= rollout.classes:
        raise ValueError("Semantic predictions must be int64 class indices")
    if not torch.equal(labels, prediction.semantic_logits.argmax(dim=-1)):
        raise ValueError("Semantic labels disagree with logits")
    return prediction


class OccWorldAdapter:
    def __init__(self, model: nn.Module, rollout: Rollout = Rollout()) -> None:
        self.model = model
        self.rollout = rollout
        self.model.requires_grad_(False)
        self.model.zero_grad(set_to_none=True)
        self.model.eval()

    def assert_frozen(self) -> None:
        if any(module.training for module in self.model.modules()):
            raise RuntimeError("OccWorld must remain in evaluation mode")
        if any(
            parameter.requires_grad or parameter.grad is not None
            for parameter in self.model.parameters()
        ):
            raise RuntimeError("OccWorld parameters must be frozen with absent gradients")

    def predict(self, sequence: torch.Tensor, metadata: list[dict[str, Any]]) -> WorldPrediction:
        validate_sequence(sequence, metadata, self.rollout)
        self.assert_frozen()
        context = torch.cuda.device(sequence.device) if sequence.is_cuda else nullcontext()
        with context, torch.no_grad():
            output = self.model.forward_autoreg_with_pose(
                sequence,
                metadata,
                start_frame=self.rollout.start_frame,
                mid_frame=self.rollout.mid_frame,
                end_frame=self.rollout.end_frame,
            )
            prediction = normalize_prediction(output, self.rollout)
        self.assert_frozen()
        return prediction
