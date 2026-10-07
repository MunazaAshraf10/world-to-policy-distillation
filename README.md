# World-to-Policy Transfer with a Frozen OccWorld

This project trains a compact driving planner with guidance from a world model that is never needed at inference time. It adapts World-to-Policy Transfer (WPT, [arXiv:2511.20095](https://arxiv.org/abs/2511.20095)) to [OccWorld](https://github.com/wzzheng/OccWorld), a 3D occupancy world model, on nuScenes. A frozen OccWorld forecasts the scene. A teacher plans several candidate trajectories on that forecast, and a learned reward model picks the best one. Two distillation losses transfer this guidance into a compact student that plans from observations alone.

**Checkpoints and model card:** [huggingface.co/Munaza10/world-to-policy-distillation](https://huggingface.co/Munaza10/world-to-policy-distillation)

This is an adaptation, not a reproduction. WPT uses Drive-OccWorld and camera based policies; here the world model is OccWorld and the policies observe Occ3D occupancy. None of the paper's numbers are claimed.

## Why

World models predict how a scene will evolve, which is exactly what a planner needs, but rolling one out costs hundreds of milliseconds per decision. WPT proposes to use the world model only during training: it shapes a teacher, and the teacher's knowledge is distilled into a student that never calls the world model. If this works, a deployed planner gets the benefit of future prediction at the cost of a small network.

## What was done

1. **World model.** No pretrained OccWorld checkpoint is publicly reachable, so OccWorld was retrained with its own unmodified code (VQVAE, then the forecasting transformer). On all 4,219 validation windows it matches the published model to within 0.33 points:

   | Forecast | mIoU 1s | mIoU 2s | mIoU 3s | IoU 1s | IoU 2s | IoU 3s |
   | --- | --- | --- | --- | --- | --- | --- |
   | OccWorld, published | 25.78 | 15.15 | 10.51 | 34.63 | 25.07 | 20.18 |
   | OccWorld, retrained here | 25.47 | 15.14 | 10.58 | 34.48 | 25.27 | 20.51 |

2. **Teacher and reward model.** The teacher plans six candidates on OccWorld's predicted future (Eq. 8), each tied to a k-means trajectory anchor as in UniAD, which the paper cites for its candidate set. The reward model scores each candidate against the predicted world with an imitation term and five rule based terms (collision, drivable area, progress, time to collision, comfort; Eqs. 9 to 14).
3. **Student and distillation.** A 1.05M parameter student plans from five past occupancy frames, the driving command, and its recent motion. Policy distillation aligns its planning query with the teacher's (Eq. 15). World reward distillation pushes its trajectory's reward toward that of the teacher's selected plan (Eq. 16).
4. **Controlled comparison.** All students share data, seed, and a 12 epoch schedule, so the only difference between them is the distillation terms.

## Results

Open-loop planning on the 4,219 nuScenes validation windows, best checkpoint by validation L2. L2 is in meters; collision rates are percentages. Collision rates are reported both exactly and averaged over 20 draws of 1 cm position jitter, because a few windows lie on the collision boundary and centimeter changes move the exact rate by about half a point.

| Model | L2 1s | L2 2s | L2 3s | L2 avg | Col. avg | Col. avg, jittered |
| --- | --- | --- | --- | --- | --- | --- |
| Student baseline | 0.39 | 1.02 | 1.97 | **1.127** | 1.18 | 1.21 ± 0.11 |
| + policy distillation | 0.39 | 1.04 | 2.01 | 1.150 | 1.37 | 1.30 ± 0.04 |
| + world reward distillation | 0.41 | 1.03 | 1.98 | 1.139 | 1.06 | 1.27 ± 0.30 |
| Full WPT | 0.40 | 1.05 | 2.01 | 1.153 | 0.81 | 1.25 ± 0.31 |
| Teacher reference (OccWorld + teacher + reward) | 0.26 | 0.81 | 1.68 | **0.919** | 0.86 | 0.94 ± 0.19 |

| Deployment, batch size 1, RTX 5090 | Latency | Parameters |
| --- | --- | --- |
| Student | 2.4 ms | 1.05M |
| OccWorld + teacher + reward | 268 ms | 79.5M |

**Highlights.**
- The deployed student plans in 2.4 ms with 1.05M parameters, 114 times faster than the world model pipeline, and needs neither OccWorld, the teacher, nor the reward model.
- Every student variant reaches about 1.13 to 1.15 m average L2 and about 1.2 to 1.3% collision rate under jitter, so the compact planner is accurate and stable across all training objectives.
- Planning on the world model's forecast is strong: the teacher reaches 0.919 m average L2, 18% below the student baseline, with a lower collision rate. This confirms that the retrained OccWorld provides a useful planning signal during training.

Final checkpoint results, a 24 epoch baseline (best L2 1.139 m), and every per horizon metric are in [results/](results/) and on the model card.

## Benefits

- A verified, retrained OccWorld that matches the published forecasting quality, released because the original weights are no longer available.
- A complete WPT pipeline whose every component is mapped to its equation ([docs/paper_mapping.md](docs/paper_mapping.md)) and whose every open choice is documented ([docs/implementation_notes.md](docs/implementation_notes.md)). This includes a sign error in Eq. 11, the singleton softmax in Eq. 16, teacher mode collapse without anchors, and why the reward model needs the predicted ego path.
- A real time student planner that runs without any world model at deployment, with a 114 times lower latency and a 76 times smaller footprint than the teacher pipeline.
- A more reliable collision measurement for nuScenes open-loop planning, which shows that single exact collision rates cannot rank methods separated by less than about half a point.

## Limitations

- The policies observe ground truth Occ3D occupancy rather than camera images, so the numbers are not comparable to camera based planners or to the WPT paper's tables.
- The teacher and reward model receive privileged conditioning through OccWorld. This affects only training, never the deployed student.
- Each configuration was trained with one seed. Loss weights, the number of candidates, and feature sizes are fixed a priori because the paper does not report them.
- nuScenes open-loop metrics are known to reward ego status shortcuts.

Environment setup, data provenance with checksums, and the full pipeline are in [docs/setup.md](docs/setup.md).

## References

- Jiang et al. WPT: World-to-Policy Transfer via Online World Model Distillation. [arXiv:2511.20095](https://arxiv.org/abs/2511.20095).
- Zheng et al. OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving. ECCV 2024. [Code](https://github.com/wzzheng/OccWorld).
- Hu et al. Planning-oriented Autonomous Driving (UniAD). CVPR 2023.
- Tian et al. Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark for Autonomous Driving. NeurIPS 2023. [Project](https://tsinghua-mars-lab.github.io/Occ3D/).
- Caesar et al. nuScenes: A Multimodal Dataset for Autonomous Driving. CVPR 2020. [Dataset](https://www.nuscenes.org/).
