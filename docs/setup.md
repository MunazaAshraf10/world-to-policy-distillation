# Setup

Everything runs from the repository root through uv. The environment targets Python 3.12 and PyTorch built for CUDA 12.8. This covers Blackwell GPUs such as the RTX 5090 as well as Ada and Hopper.

```bash
git submodule update --init OccWorld
uv sync
```

OccWorld stays pinned at 1ee7f77ecc4c984a4f7f6411d95c2e6e73806b6e and is never modified. Its mmdet3d and nuScenes devkit imports are replaced at import time by small stand-ins (src/world/occworld.py). Neither mmcv nor mmdet3d needs to be installed.

## Data

```bash
scripts/download_data.sh data
```

This fetches and verifies three files. Camera images and LiDAR sweeps are not needed.

| File | SHA256 | Source |
| --- | --- | --- |
| gts.tar.gz (Occ3D-nuScenes labels, 2.74 GB) | 0635d138...abba53 | Hugging Face mirror of the Occ3D release |
| nuscenes_infos_train_temporal_v3_scene.pkl | fcacfe8f...a111c87 | OccWorld metadata as redistributed by COME |
| nuscenes_infos_val_temporal_v3_scene.pkl | 7e354777...cab8186 | same |

The original OccWorld metadata links are no longer reachable. The redistributed files load with the pinned dataset code and give upstream's 4,219 validation windows. Occ3D data remains subject to the Occ3D and nuScenes terms of use.

## Caches

```bash
uv run python -m scripts.build_cache inputs
```

This decodes all 34,149 occupancy frames into uint8 memmaps (about 22 GB under cache/) and stacks per-window planning inputs, targets, and agent annotations. The train split yields 19,730 windows and the val split 4,219. The step takes about one minute with 12 workers.

## Pipeline

```bash
# 1. Retrain OccWorld: VQVAE, then the world transformer with the VQVAE frozen.
uv run python -m scripts.train --config configs/occworld_vqvae.yaml
uv run python -m scripts.train --config configs/occworld.yaml
cp checkpoints/occworld/last.pt checkpoints/occworld/world.pt

# 2. Verify the frozen integration and cache its rollouts.
uv run python -m scripts.build_cache world --checkpoint checkpoints/occworld/world.pt
uv run python -m scripts.verify_occworld --checkpoint checkpoints/occworld/world.pt
uv run python -m scripts.evaluate_world --checkpoint checkpoints/occworld/world.pt

# 3. Policies.
uv run python -m scripts.train --config configs/student.yaml
uv run python -m scripts.train --config configs/teacher.yaml
uv run python -m scripts.train --config configs/wpt.yaml
uv run python -m scripts.train --config configs/wpt.yaml --set experiment=wpt_policy loss.reward=0
uv run python -m scripts.train --config configs/wpt.yaml --set experiment=wpt_reward loss.policy=0

# 4. Evaluation and latency.
uv run python -m scripts.evaluate --checkpoint checkpoints/wpt_full/best.pt
uv run python -m scripts.benchmark --student checkpoints/wpt_full/best.pt \
  --teacher checkpoints/teacher/best.pt --world checkpoints/occworld/world.pt
```

Any config value can be overridden with --set key.sub=value, and configs can be stacked. For example, adding configs/debug.yaml to any stage runs a 16 window overfit check.

## Tests

```bash
uv run pytest -q
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

tests/test_wpt.py runs on CPU with tiny models. tests/test_occworld.py needs the trained checkpoint (OCCWORLD_CHECKPOINT, default checkpoints/occworld/last.pt) and a GPU, and is skipped otherwise.
