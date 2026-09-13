# WPT-OccWorld: World-to-Policy Distillation for Lightweight Autonomous Driving

> **Research reproduction + extension project** inspired by  
> **WPT: World-to-Policy Transfer via Online World Model Distillation** (2025/2026)  
> with a **pretrained OccWorld world model** to avoid training a world model from scratch.

This repository explores whether knowledge from a pretrained autonomous-driving world model can be transferred into a lightweight driving policy through **policy distillation** and **world-reward distillation**.

The project is designed as an independent research implementation: reproduce the core WPT methodology, adapt it to a publicly available pretrained occupancy world model, and evaluate whether the distilled student retains planning quality while reducing inference cost.

---

## Research Question

**Can a computationally expensive world model be used only during training to improve a lightweight policy that runs independently at inference time?**

The central idea is:

```text
                         TRAINING
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  nuScenes observations                                       │
│          │                                                   │
│          ├──────────────► Pretrained OccWorld                │
│          │                    (frozen)                        │
│          │                       │                            │
│          │                 future world states               │
│          │                       │                            │
│          ▼                       ▼                            │
│    Teacher Policy ───────► Reward Model                      │
│          │                       │                            │
│          │                 best trajectory                   │
│          │                       │                            │
│          ├──── Policy Distillation ───────────┐              │
│          │                                    │              │
│          └──── World-Reward Distillation ─────┤              │
│                                               ▼              │
│                                      Student Policy          │
│                                                              │
└──────────────────────────────────────────────────────────────┘

                         INFERENCE

                 camera / BEV observations
                           │
                           ▼
                    Student Policy
                           │
                           ▼
                   predicted trajectory

         OccWorld + teacher + reward model are discarded.
```

---

## Why This Project?

World models can provide rich information about how a scene may evolve, but they can be too expensive for real-time deployment.

WPT proposes transferring that knowledge into a lightweight policy so that the world model is required **during training only**.

This project adapts that idea using **OccWorld**, which has publicly released pretrained weights. This avoids the high cost of training a complete world model from scratch and makes the experiment practical on a single H100-class GPU.

> **Important:** this is not intended to be an exact reproduction of the original WPT architecture.  
> The original methodology is being independently implemented while **OccWorld is substituted as the pretrained world-model component**.

---

## Papers

### WPT

**WPT: World-to-Policy Transfer via Online World Model Distillation**

- arXiv: https://arxiv.org/abs/2511.20095
- Core ideas:
  - online world-model guidance
  - teacher policy refinement
  - policy distillation
  - world-reward distillation
  - lightweight student deployment

### OccWorld

**OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving — ECCV 2024**

- GitHub: https://github.com/wzzheng/OccWorld
- Pretrained weights: https://cloud.tsinghua.edu.cn/d/ff4612b2453841fba7a5/
- Dataset: nuScenes
- Predicts future 3D occupancy and ego motion.

---

## Planned Architecture

### World Model

**OccWorld**

- pretrained
- frozen during distillation
- provides future occupancy/world-state predictions
- no backpropagation through the world model during the initial experiments

### Teacher Policy

A larger trajectory-planning policy.

Initial implementation:

```text
BEV / scene representation
        ↓
planning decoder
        ↓
K candidate trajectories
```

The teacher proposes multiple candidate trajectories.

The reward module uses predicted future world states to rank them.

### Student Policy

A smaller policy that produces a trajectory without running the world model.

```text
observation
    ↓
lightweight encoder
    ↓
planning decoder
    ↓
trajectory
```

Possible student backbones:

- BEVFormer-Tiny-style encoder
- lightweight ViT/BEV encoder
- compact custom transformer policy

The exact backbone will be kept modular so different students can be evaluated.

---

## Distillation Objectives

### 1. Policy Distillation

Transfer the teacher's internal planning representation to the student.

Conceptually:

```math
L_policy = D(z_student, z_teacher)
```

where `z_teacher` and `z_student` are planning features/query representations.

Initial implementation candidates:

- MSE
- Smooth L1
- cosine-distance loss

The final form will follow the WPT paper as closely as possible.

### 2. World-Reward Distillation

The teacher generates multiple trajectories:

```text
τ1, τ2, ..., τK
```

The world-model-guided reward module selects:

```text
τ* = argmax R(τk)
```

The student is then encouraged to generate a trajectory whose world reward approaches the teacher's best trajectory.

Conceptually:

```math
L_reward = |R(τ_student) - R(τ_teacher*)|
```

### Total Objective

```math
L_total =
    L_planning
  + λ_policy L_policy
  + λ_reward L_reward
```

---

## Dataset

### nuScenes

This project uses the **nuScenes** autonomous-driving dataset.

Expected inputs include:

- multi-camera observations
- ego pose / ego motion
- map information when required
- trajectory annotations
- occupancy annotations required by OccWorld

Dataset website:

https://www.nuscenes.org/

OccWorld additionally expects the occupancy data/preprocessing described in its repository.

---

## Compute

Primary development hardware:

```text
GPU: 1 × NVIDIA H100 PCIe
VRAM: 80 GB
```

The PCIe H100 is sufficient for initial development because the pretrained world model is frozen.

Development strategy:

1. debug on a very small nuScenes subset
2. verify forward passes and loss values
3. overfit a tiny dataset
4. run reduced experiments
5. launch full training only after the pipeline is stable

This avoids wasting expensive GPU hours on implementation bugs.

---

## Experimental Plan

### Experiment 0 — Pipeline Sanity Check

Goal:

- load nuScenes
- load OccWorld checkpoint
- run frozen world-model inference
- visualize predicted future occupancy

Success criterion:

```text
observation → OccWorld → future occupancy
```

works end-to-end.

---

### Experiment 1 — Student Baseline

Train the lightweight student **without distillation**.

This establishes the baseline:

```text
Student + imitation/planning loss
```

Record:

- trajectory L2
- collision rate
- inference latency
- parameter count

---

### Experiment 2 — Teacher Policy

Train the larger teacher policy.

Teacher outputs multiple candidate trajectories.

Expected behavior:

```text
observation
   ↓
teacher
   ↓
{τ1, τ2, ..., τK}
```

---

### Experiment 3 — World-Model Reward

Freeze OccWorld.

For each candidate trajectory:

```text
candidate trajectory
        +
predicted future world
        ↓
    reward score
```

Select the teacher's best candidate.

---

### Experiment 4 — Policy Distillation

Train the student to match the teacher's planning representation.

Compare:

```text
Student baseline
vs.
Student + policy distillation
```

---

### Experiment 5 — Full WPT-Style Distillation

Train using:

```text
planning loss
+
policy distillation
+
world-reward distillation
```

Compare:

| Model | Policy KD | World-Reward KD | Expected role |
|---|---:|---:|---|
| Student baseline | ✗ | ✗ | baseline |
| Student + Policy KD | ✓ | ✗ | ablation |
| Student + Reward KD | ✗ | ✓ | ablation |
| Full student | ✓ | ✓ | main method |
| Teacher | — | — | upper reference |

---

## Evaluation Metrics

The primary metrics are related to **planning quality and deployment efficiency**, rather than classification accuracy.

### Planning

- **Average trajectory L2 error ↓**
- **Collision rate ↓**
- planning success / driving score where supported by the evaluation benchmark

### Efficiency

- **Inference latency ↓**
- **FPS ↑**
- parameter count ↓
- peak VRAM ↓
- FLOPs / compute when practical

### Distillation

Also report:

```text
Δ performance over student baseline
Δ performance relative to teacher
student / teacher latency ratio
```

The ideal result is:

```text
Student + WPT
    ≫ baseline student performance
    ≈ teacher planning quality
    ≪ teacher/world-model inference cost
```

---

## Ablations

Planned ablation experiments:

1. remove policy distillation
2. remove world-reward distillation
3. vary reward-loss weight
4. vary policy-distillation weight
5. change number of teacher candidate trajectories
6. compare different student capacities
7. compare frozen vs partially trainable reward module

---

## Repository Structure

Planned layout:

```text
wpt-occworld/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── baseline.yaml
│   ├── teacher.yaml
│   ├── policy_kd.yaml
│   └── full_wpt.yaml
│
├── data/
│   └── README.md
│
├── models/
│   ├── world_model.py
│   ├── teacher_policy.py
│   ├── student_policy.py
│   └── reward_model.py
│
├── distillation/
│   ├── policy_distillation.py
│   ├── world_reward_distillation.py
│   └── losses.py
│
├── datasets/
│   └── nuscenes_dataset.py
│
├── evaluation/
│   ├── planning_metrics.py
│   ├── collision_metrics.py
│   └── latency.py
│
├── scripts/
│   ├── download_checkpoints.sh
│   ├── train_student_baseline.sh
│   ├── train_teacher.sh
│   └── train_wpt.sh
│
├── train.py
├── evaluate.py
│
├── tests/
│   ├── test_world_model.py
│   ├── test_reward_model.py
│   └── test_distillation.py
│
└── results/
    ├── tables/
    ├── plots/
    └── visualizations/
```

---

## Implementation Roadmap

### Phase 1 — Environment

- [ ] clone OccWorld
- [ ] create compatible environment
- [ ] download pretrained weights
- [ ] download / prepare nuScenes
- [ ] run official OccWorld evaluation
- [ ] reproduce one provided visualization

### Phase 2 — WPT Components

- [ ] implement teacher policy
- [ ] implement lightweight student
- [ ] implement candidate trajectory generation
- [ ] implement reward module
- [ ] implement policy distillation
- [ ] implement world-reward distillation

### Phase 3 — Verification

- [ ] unit-test loss functions
- [ ] check all tensor dimensions
- [ ] verify frozen OccWorld gradients
- [ ] overfit 32–128 samples
- [ ] visualize teacher/student trajectories

### Phase 4 — Experiments

- [ ] student baseline
- [ ] teacher baseline
- [ ] policy-KD ablation
- [ ] reward-KD ablation
- [ ] full WPT-style training
- [ ] latency benchmark
- [ ] final evaluation

### Phase 5 — Research Report

- [ ] results table
- [ ] training curves
- [ ] qualitative trajectory examples
- [ ] ablation analysis
- [ ] failure cases
- [ ] limitations
- [ ] reproducibility instructions

---

## Results

Results will be added after experiments.

| Method | Trajectory L2 ↓ | Collision ↓ | Latency ↓ | Params |
|---|---:|---:|---:|---:|
| Student baseline | TBD | TBD | TBD | TBD |
| Student + Policy KD | TBD | TBD | TBD | TBD |
| Student + Reward KD | TBD | TBD | TBD | TBD |
| **Full WPT-style student** | **TBD** | **TBD** | **TBD** | **TBD** |
| Teacher | TBD | TBD | TBD | TBD |

---

## Reproducibility Principles

This repository aims to document:

- all hyperparameters
- random seeds
- dataset splits
- checkpoint versions
- GPU type
- training duration
- dependency versions
- failed experiments and implementation decisions

The objective is not only to report a final number, but to make the independent implementation auditable and reproducible.

---

## Project Status

**Stage:** setup / implementation

Current priority:

```text
nuScenes
   ↓
pretrained OccWorld
   ↓
verify world-model inference
   ↓
build student baseline
   ↓
implement WPT losses
```

---

## References

```bibtex
@article{jiang2025wpt,
  title   = {WPT: World-to-Policy Transfer via Online World Model Distillation},
  author  = {Jiang, Guangfeng and Luo, Yueru and Liu, Jun and Huang, Yi and
             Zhu, Yiyao and Qu, Zhan and Chen, Dave Zhenyu and Liu, Bingbing
             and Yan, Xu},
  journal = {arXiv preprint arXiv:2511.20095},
  year    = {2025}
}
```

```bibtex
@inproceedings{zheng2024occworld,
  title     = {OccWorld: Learning a 3D Occupancy World Model for Autonomous Driving},
  author    = {Zheng, Wenzhao and Chen, Weiliang and Huang, Yuanhui and
               Zhang, Borui and Duan, Yueqi and Lu, Jiwen},
  booktitle = {European Conference on Computer Vision},
  year      = {2024}
}
```

---

## Acknowledgements

This project builds on the ideas introduced in **WPT** and uses the publicly released **OccWorld** model and pretrained checkpoint.

The WPT methodology is independently implemented for research and educational purposes.
