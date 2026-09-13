# WPT with frozen OccWorld

This repository implements a staged adaptation of **World-to-Policy Transfer via Online World Model Distillation (WPT)** using pretrained **OccWorld** in place of Drive-OccWorld. The goal is to transfer world-informed planning knowledge into a student policy that runs independently at deployment.

**Current scope: Stage 1, frozen OccWorld verification.** The adapter, sample loader, verification CLI, tests, and reproducibility reports are implemented. No teacher, student, reward model, distillation loss, or training run is implemented yet. Stage 1 is not accepted until a real pretrained checkpoint-backed rollout passes.

Validation completed: **35 tests passed**, one real integration test skipped, and the downloaded occupancy window passed the upstream data loader. See [verification status](docs/verification_status.md) for evidence and the remaining checkpoint blocker.

The local [WPT paper](WPT.pdf) is arXiv:2511.20095v2, March 18, 2026. The [OccWorld submodule](OccWorld) is pinned to `1ee7f77ecc4c984a4f7f6411d95c2e6e73806b6e` and is kept unchanged.

## Intended training architecture

```text
nuScenes history
    |
    +-- historical occupancy + motion metadata --> frozen OccWorld
    |                                                |
    |                                         predicted future world
    |                                                |
    |                              +-----------------+----------------+
    |                              |                                  |
    |                       teacher plan decoder                 reward model
    |                              |                                  ^
    |                   planning representations                      |
    |                   + candidate trajectories ---------------------+
    |                              |                                  |
    |                              |                           best teacher reward
    |                              v                                  |
    |                     policy distillation                 reward distillation
    |                              ^                                  ^
    |                              |                                  |
    +--> student encoder + decoder +--> student trajectory --> same reward model
```

The teacher's planning decoder uses predicted world features (WPT Eq. 8). Policy distillation matches planning representations (Eq. 15). World reward distillation compares the student's trajectory reward with the best teacher reward (Eq. 16). Selection of a teacher trajectory is not itself the policy distillation loss.

For the initial official OccWorld protocol, future `gt_mode` conditioning also enters the world model. The full [paper mapping and diagram](docs/paper_mapping.md) makes that dependency explicit.

At deployment, the intended path is simply observations → student → trajectory. The student must not initialize or call OccWorld, the teacher, or the reward model.

## Stage 1 setup

See [setup and asset instructions](docs/setup.md) for the isolated Python 3.8 environment, exact package versions, downloads, and tests. The existing system PyTorch environment remains unchanged.

```bash
.venv-occworld/bin/python scripts/verify_occworld.py \
  --preflight-only --output results/occworld_stage1/preflight.json
```

The real rollout requires explicit asset paths:

```bash
.venv-occworld/bin/python scripts/verify_occworld.py \
  --checkpoint checkpoints/occworld/full.pth \
  --data-root data/nuscenes \
  --infos data/nuscenes_infos_val_temporal_v3_scene.pkl
```

The checkpoint filename above is a required input, not a bundled asset. The original checkpoint share currently reports “Link does not exist”; see [upstream issue 39](https://github.com/wzzheng/OccWorld/issues/39). A metadata copy is available through DOME; its distinct provenance and checksum are documented in the setup guide. A DOME model checkpoint cannot substitute for pretrained OccWorld.

The verifier uses one official validation window: 12 occupancy grids, five history frames, and six predictions at 0.5-second intervals. It loads every checkpoint parameter and buffer, freezes the entire model, and checks shape, finite values, adapter/direct parity, and isolation from future occupancy and displacement targets.

| Output | Shape |
| --- | --- |
| Semantic labels | `[1,6,200,200,16]` |
| Semantic logits | `[1,6,200,200,16,18]` |
| Ego displacement modes | `[1,6,3,2]` |
| Selected ego displacements | `[1,6,2]` |

Spatial axes and per-step displacement semantics remain unchanged. Future `gt_mode` values are retained exactly as in upstream inference; this is an official-protocol verification, not observation-only forecasting. Future occupancy and displacement annotations are independently perturbed to check that predictions do not depend on them.

Results are written to `results/occworld_stage1/summary.json`, with configuration, source and asset hashes, sample identity, conditioning, tensor statistics, checks, software versions, GPU, elapsed time, and peak allocated VRAM. An existing report is preserved unless `--overwrite` is specified. Only a report with `status: stage1_passed` establishes acceptance.

## Tests

```bash
.venv-occworld/bin/python -m pytest -q
.venv-occworld/bin/ruff check src scripts tests
.venv-occworld/bin/ruff format --check src scripts tests
```

The real integration test is skipped unless `--integration-config` identifies a YAML file with actual asset paths. A skipped integration test is not a successful Stage 1 run. Small test fixtures verify adapter contracts only and cannot be selected by the production CLI.

## Research roadmap

1. Complete frozen OccWorld verification with a full pretrained checkpoint.
2. Establish a lightweight student baseline without distillation.
3. Implement a teacher decoder conditioned on predicted future world features.
4. Implement separate imitation and simulation reward supervision.
5. Implement policy distillation and evaluate its ablation.
6. Implement world reward distillation and evaluate its ablation.
7. Compose full WPT adaptation training and compare all controlled variants.

Later evaluation will report trajectory L2 and collision rates at 1, 2, and 3 seconds, plus latency, parameter count, VRAM, and GPU hours. No reproduction metrics or speedup claims are currently made.

Before reward implementation, resolve Eq. 11's apparent sign inconsistency, query alignment, norm reductions, shared reward normalization, loss weights, update schedules, and gradient boundaries. These are documented in [implementation notes](docs/implementation_notes.md), not filled in with arbitrary approximations.

## References

- [WPT paper](https://arxiv.org/abs/2511.20095)
- [OccWorld source and paper](https://github.com/wzzheng/OccWorld)
- [Occ3D dataset](https://github.com/Tsinghua-MARS-Lab/Occ3D)
- [nuScenes](https://www.nuscenes.org/)
