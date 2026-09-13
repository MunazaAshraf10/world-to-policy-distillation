# OccWorld setup and verification

Run commands from the repository root. The isolated environment leaves the system PyTorch installation unchanged.

```bash
git submodule update --init OccWorld
uv venv --python 3.8.20 .venv-occworld
uv pip install --python .venv-occworld/bin/python \
  'torch==2.0.1+cu118' 'torchvision==0.15.2+cu118' \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  --index-strategy unsafe-best-match
uv pip install --python .venv-occworld/bin/python \
  -r requirements-occworld.txt \
  --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html \
  --only-binary mmcv
.venv-occworld/bin/python scripts/verify_occworld.py \
  --preflight-only --output results/occworld_stage1/preflight.json
```

The requirements retain upstream's principal versions. Python 3.8.20 replaces the original 3.8.0 patch release. MMDetection 3.0.0 and MMDetection3D 1.1.1 are explicit additions because upstream imports MMDetection3D but omits its version in the exported environment. Their declared compatibility includes MMCV 2.0.1 and MMEngine 0.8.4. See the [MMDetection3D version checks](https://github.com/open-mmlab/mmdetection3d/blob/v1.1.1/mmdet3d/__init__.py).

The complete installed dependency snapshot is requirements-occworld.lock.txt. To reproduce its resolved transitive versions after installing the CUDA PyTorch wheels, substitute this lock file for requirements-occworld.txt in the second installation command.

## Required assets

Stage 1 needs a full pretrained OccWorld checkpoint, original Occ3D labels for the selected window, and the temporal validation metadata. Camera JPEGs, lidar sweeps, and the full nuScenes download are not read by this occupancy-only verification. The temporal pickle provides calibration and annotation fields used by the upstream dataset.

Keep assets outside version control:

```text
checkpoints/occworld/full.pth
data/nuscenes_infos_val_temporal_v3_scene.pkl
data/nuscenes/gts/scene-0003/<sample-token>/labels.npz
```

Both original Tsinghua cloud shares returned “Link does not exist” during setup. Upstream issues [39](https://github.com/wzzheng/OccWorld/issues/39) and [37](https://github.com/wzzheng/OccWorld/issues/37) report the missing model and metadata links. No verified replacement full OccWorld checkpoint was found. A DOME checkpoint is a different model and cannot substitute for it.

A reply in [OccWorld issue 35](https://github.com/wzzheng/OccWorld/issues/35) points to [DOME's data preparation links](https://github.com/gusongen/DOME#data-preparation). Its validation metadata copy was downloaded and inspected: 150 scenes, with the fields expected by the pinned dataset. Its identity against the unavailable original cannot be independently established. This provenance distinction must remain in experiment reporting.

```bash
mkdir -p data
uv tool run --from gdown==5.2.0 gdown \
  1VzINYKgQAnw_xERZ4Foqz2nD-SQpMhV0 \
  -O data/nuscenes_infos_val_temporal_v3_scene.pkl
uv tool run --from gdown==5.2.0 gdown \
  1kiXVNSEi3UrNERPMz_CfiJXKkgts_5dY -O data/gts.tar.gz
```

Extract exactly one official validation window and record its file hashes:

```bash
.venv-occworld/bin/python scripts/extract_occworld_sample.py \
  --archive data/gts.tar.gz \
  --infos data/nuscenes_infos_val_temporal_v3_scene.pkl \
  --data-root data/nuscenes \
  --manifest data/occworld_sample_manifest.json
```

This setup downloaded the original archive and extracted 12 frames from scene-0003. The archive SHA256 is 0635d1383d8b99cce26a0344ec3ca3c4346f53907a06e82ea525acf7b1abba53. Existing identical occupancy files are reused; differing files and existing manifests are preserved and cause an error. The source archive remains available for additional windows.

The second URL is the occupancy archive from the [official challenge trainval folder](https://github.com/CVPR2023-3D-Occupancy-Prediction/CVPR2023-3D-Occupancy-Prediction#download). Fetching image archives is unnecessary. Dataset use remains subject to the terms identified by [Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D#license).

The metadata SHA256 is d174b9563adcfc2b0450d164dce108b11742197208436f70781d4187251da566. The selected sample is dataset index 0, scene-0003, with 12 frames starting at fd8420396768425eabec9bdddf7e64b6. The current observation is f56a544064a548a39a81f18cc8f633c5 at window index 4.

## Run the real verification

Once the full checkpoint is available:

```bash
.venv-occworld/bin/python scripts/verify_occworld.py \
  --checkpoint checkpoints/occworld/full.pth \
  --data-root data/nuscenes \
  --infos data/nuscenes_infos_val_temporal_v3_scene.pkl
```

The checkpoint path above is a user-supplied asset location, not a file created by setup. This command must not be reported as successful until the real checkpoint-backed rollout passes.

The CLI validates inputs, loads the entire checkpoint, runs adapter/direct parity and two future-annotation isolation checks, and writes results/occworld_stage1/summary.json. Existing reports require a different output path or --overwrite. Configuration paths resolve relative to the project root; CLI paths use the same rule.

## Tests

```bash
.venv-occworld/bin/python -m pytest -q
.venv-occworld/bin/ruff check src scripts tests
.venv-occworld/bin/ruff format --check src scripts tests
```

The integration test is explicitly skipped without asset configuration. To enable it, copy configs/debug.yaml, fill checkpoint, data_root, and infos, and run:

```bash
.venv-occworld/bin/python -m pytest -q -m integration \
  --integration-config /absolute/path/to/local_debug.yaml
```

An explicitly configured integration test fails for missing assets rather than skipping. Unit fixtures never enter the production verification path. A preflight report and passing unit tests do not mean Stage 1 has passed.
