# WPT on a frozen OccWorld

This repository adapts World-to-Policy Transfer (WPT, [arXiv:2511.20095](https://arxiv.org/abs/2511.20095)) to OccWorld on nuScenes. A frozen OccWorld rolls out future occupancy. A multi-modal teacher plans on that predicted world and a learned reward model scores its candidates. Policy query distillation and world reward distillation then compress this guidance into a compact student, which plans without the world model, the teacher, or the reward model.

It is an adaptation, not a reproduction. WPT uses Drive-OccWorld and camera based policies. Here the world model is OccWorld (pinned at [1ee7f77](https://github.com/wzzheng/OccWorld/tree/1ee7f77ecc4c984a4f7f6411d95c2e6e73806b6e), unmodified, retrained because no pretrained checkpoint is reachable), and the policies observe Occ3D occupancy. None of the paper's reported numbers are claimed.

## Method

```
observed occupancy, command, ego history
  ├─► frozen OccWorld ─► predicted latents F^w_{t+1}, predicted occupancy
  │                        ├─► teacher P_D(Q^T, F^w_{t+1}) ─► K candidates, queries Q^T
  │                        └─► reward model (Eq. 9) ─► imitation + 5 simulation scores ─► r_final (Eq. 14)
  │                                                     └─► tau*_T = argmax r_final
  └─► student P_D(Q^S, F^w) ─► trajectory, query Q^S
          ├─ policy distillation    ||P(Q^S) - Q^T[tau*]||_2         (Eq. 15)
          └─ reward distillation    |r_final(tau^S) - r_final(tau*_T)| (Eq. 16)
```

Training runs in stages:
1. Retrain OccWorld: VQVAE, then the world transformer.
2. Cache its rollouts once. The model is frozen and decodes by argmax, so the cache is exact.
3. Train a student baseline.
4. Train the teacher and the reward model jointly.
5. Train the student with the teacher and reward model frozen and running online.

The teacher's simulation targets (NC, DAC, EP, TTC, comfort; App. 6.3) are computed on the world model's predicted occupancy.

[docs/paper_mapping.md](docs/paper_mapping.md) maps every component to its equation and source file. [docs/implementation_notes.md](docs/implementation_notes.md) records every choice the paper leaves open, including:
- the sign error in Eq. 11
- the singleton softmax problem in Eq. 16
- the privileged conditioning OccWorld gives the teacher
- the occlusion-aware drivable area used for DAC

## Repository

```
configs/   stage configs; any value can be overridden with --set key=value
scripts/   download_data.sh, build_cache, train, verify_occworld, evaluate, benchmark
src/world      OccWorld adapter, import shims, world feature encoder
src/data       Occ3D windows and memmap caches
src/policy     occupancy encoder, plan decoder and head, student, teacher
src/reward     reward model and rule based simulation targets
src/losses     planning, reward, policy distillation, reward distillation, total
src/training   policy trainer, per-stage steps, OccWorld retraining
src/evaluation STP3 planning metric, forecast IoU, latency
tests/     CPU unit tests for the WPT components; GPU integration test for OccWorld
```

## Usage

Setup, data provenance with checksums, and the full command sequence are in [docs/setup.md](docs/setup.md). In short:

```bash
git submodule update --init OccWorld && uv sync
scripts/download_data.sh data
uv run python -m scripts.build_cache inputs
uv run python -m scripts.train --config configs/occworld_vqvae.yaml
uv run python -m scripts.train --config configs/occworld.yaml
uv run python -m scripts.build_cache world --checkpoint checkpoints/occworld/last.pt
uv run python -m scripts.train --config configs/student.yaml
uv run python -m scripts.train --config configs/teacher.yaml
uv run python -m scripts.train --config configs/wpt.yaml
uv run python -m scripts.evaluate --checkpoint checkpoints/wpt_full/best.pt
```

Everything targets a single 32 GB GPU. Retraining OccWorld dominates the cost: about 8 hours for the VQVAE and 12 for the transformer on an RTX 5090. Each policy stage takes one to a few hours.

## Evaluation

Open-loop planning on the 4,219 nuScenes validation windows of the OccWorld protocol. Metrics:
- L2 error and collision rate at 1, 2, and 3 s, through OccWorld's STP3 planning metric. Both the per-time convention (used in the OccWorld and WPT tables) and the cumulative ST-P3 convention are reported.
- Student and teacher pipeline latency at batch size 1, and parameter counts.

Each result file in results/ stores its full configuration, checkpoint hash, git revision, and environment.

## Results

No results have been produced yet. This section will list the student baseline, policy-only and reward-only distillation, full WPT, and the teacher reference, together with the retrained OccWorld's forecast quality next to the published checkpoint's.

## Limitations

- OccWorld is retrained on a shortened schedule, so its forecasts are weaker than the original checkpoint's.
- The policies observe ground truth Occ3D occupancy rather than camera images, so the results are not comparable to camera based planners.
- The teacher and reward model are privileged: OccWorld's rollout is conditioned on future driving commands and on the first future ego displacement. This affects only training signals, never the deployed student.
- Loss weights, the number of candidates, and feature sizes are a priori choices; the paper does not report them.
- nuScenes open-loop metrics are known to reward ego-status shortcuts. The student sees only the command and two past displacements.

## References

- Jiang et al., WPT: World-to-Policy Transfer via Online World Model Distillation, arXiv:2511.20095.
- Zheng et al., OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving, ECCV 2024.
- Tian et al., Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving, NeurIPS 2023.
- Caesar et al., nuScenes: A Multimodal Dataset for Autonomous Driving, CVPR 2020.
