from __future__ import annotations

import copy
import importlib
import importlib.metadata
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.nuscenes import load_sample
from src.verification.config import VerificationConfig
from src.verification.report import git_state, sha256, tensor_summary
from src.world.occworld import OccWorldAdapter, build_model, normalize_prediction, upstream_imports
from src.world.types import WorldPrediction


def seed_run(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def preflight(config: VerificationConfig) -> dict[str, Any]:
    """Verify the pinned source, native CUDA execution, and upstream imports."""
    source = git_state(config.occworld_root)
    if source["revision"] != config.expected_revision or source["status"]:
        raise RuntimeError(f"OccWorld must be clean at {config.expected_revision}; found {source}")
    device = torch.device(config.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("OccWorld verification requires a CUDA GPU")
    torch.cuda.set_device(device)
    with upstream_imports(config.occworld_root):
        for name in ("mmcv", "mmengine", "mmdet", "mmdet3d", "model", "dataset"):
            importlib.import_module(name)
        from mmcv.ops import box_iou_rotated

        boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0, 0.0]])
        torch.testing.assert_close(box_iou_rotated(boxes, boxes), torch.ones((1, 1)))
        matrix = torch.eye(4, device=device)
        torch.testing.assert_close(matrix @ matrix, matrix)
    torch.cuda.synchronize(device)
    packages = (
        "torch",
        "torchvision",
        "numpy",
        "mmcv",
        "mmengine",
        "mmdet",
        "mmdet3d",
        "nuscenes-devkit",
        "einops",
        "scipy",
        "numba",
        "pyyaml",
    )
    return {
        "status": "preflight_passed",
        "occworld_source": source,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "gpu_capability": list(torch.cuda.get_device_capability(device)),
        "torch_cuda_execution": "passed",
        "openmmlab_imports_and_cpu_ops": "passed",
        "mmcv_cuda_ops_required_by_rollout": False,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def assert_predictions_close(actual: WorldPrediction, expected: WorldPrediction) -> None:
    for name, tensor in actual.tensors().items():
        reference = expected.tensors()[name]
        torch.testing.assert_close(tensor, reference, rtol=1e-5, atol=1e-6, msg=name)


def verify_rollout(
    adapter: OccWorldAdapter, sequence: torch.Tensor, metadata: list[dict[str, Any]]
) -> tuple[WorldPrediction, dict[str, str]]:
    """Compare direct execution and independently perturb future annotations."""
    rollout = adapter.rollout
    prediction = adapter.predict(sequence, metadata)
    with torch.no_grad():
        direct = adapter.model.forward_autoreg_with_pose(
            sequence,
            metadata,
            start_frame=rollout.start_frame,
            mid_frame=rollout.mid_frame,
            end_frame=rollout.end_frame,
        )
        reference = normalize_prediction(direct, rollout)
    assert_predictions_close(prediction, reference)
    del direct, reference

    changed_sequence = sequence.clone()
    changed_sequence[:, rollout.mid_frame :] = (
        changed_sequence[:, rollout.mid_frame :] + 1
    ) % rollout.classes
    perturbed = adapter.predict(changed_sequence, metadata)
    assert_predictions_close(perturbed, prediction)
    del perturbed, changed_sequence

    changed_metadata = copy.deepcopy(metadata)
    changed_metadata[0]["rel_poses"] = np.asarray(changed_metadata[0]["rel_poses"]).copy()
    changed_metadata[0]["rel_poses"][rollout.mid_frame :] += 1
    perturbed = adapter.predict(sequence, changed_metadata)
    assert_predictions_close(perturbed, prediction)
    adapter.assert_frozen()
    return prediction, {
        "adapter_direct_parity": "passed",
        "future_occupancy_isolation": "passed",
        "future_displacement_isolation": "passed",
        "frozen_parameters": "passed",
        "shape_and_finite_outputs": "passed",
        "comparison_tolerance": "rtol=1e-5, atol=1e-6; class indices exact",
    }


def run_verification(config: VerificationConfig, project_root: Path) -> dict[str, Any]:
    config.validate_assets()
    seed_run(config.seed)
    environment = preflight(config)
    from mmengine import Config

    upstream_config = Config.fromfile(str(config.upstream_config))
    for name in ("start_frame", "mid_frame", "end_frame"):
        if upstream_config[name] != getattr(config.rollout, name):
            raise ValueError(f"Upstream {name} disagrees with the requested official protocol")
    device = torch.device(config.device)
    sample = load_sample(
        config.occworld_root,
        config.data_root,
        config.infos,
        config.sample_index,
        config.rollout,
    )
    model = build_model(config.occworld_root, upstream_config.model, config.checkpoint, device)
    adapter = OccWorldAdapter(model, config.rollout)
    sequence = sample.sequence.to(device)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    prediction, checks = verify_rollout(adapter, sequence, sample.metadata)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device)
    code_paths = sorted(project_root.glob("src/**/*.py")) + sorted(
        project_root.glob("scripts/*.py")
    )
    return {
        "status": "stage1_passed",
        "config": config.resolved(),
        "upstream_model_config": upstream_config.model.to_dict(),
        "upstream_config_sha256": sha256(config.upstream_config),
        "source": git_state(project_root),
        "environment": environment,
        "code_sha256": {str(path.relative_to(project_root)): sha256(path) for path in code_paths},
        "checkpoint_sha256": sha256(config.checkpoint),
        "metadata_sha256": sha256(config.infos),
        "sample": sample.identity,
        "conditioning_protocol": {
            "name": "official_occupancy_with_future_gt_mode",
            "history_indices": list(range(config.rollout.start_frame, config.rollout.mid_frame)),
            "future_indices": list(range(config.rollout.mid_frame, config.rollout.end_frame)),
            "future_gt_mode_used": True,
            "pose_condition_source": "Per-frame gt_ego_fut_trajs[0], including the last history frame",
            "observation_only_forecasting": False,
            "future_annotations": "Occupancy and displacement targets are carried by the upstream API; isolation checked separately",
        },
        "tensors": tensor_summary(prediction, config.rollout.classes),
        "checks": checks,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": 0,
        "verification_elapsed_seconds": elapsed,
        "verification_forward_calls": 4,
        "peak_allocated_vram_bytes": peak,
        "timing_scope": "Four verification forwards including assertions; excludes loading; not a latency benchmark",
    }
