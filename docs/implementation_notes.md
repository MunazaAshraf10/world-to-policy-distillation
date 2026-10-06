# Implementation notes

This file records every decision that the WPT paper (arXiv:2511.20095v2) leaves open or that follows from replacing Drive-OccWorld with OccWorld. Each entry states the choice and its consequence.

## World model

**Retrained OccWorld.** No pretrained OccWorld checkpoint is publicly reachable. The official Tsinghua cloud links return "Link does not exist" (upstream issues 37 and 39), and the VQVAE was never released (issue 26). We retrain OccWorld at commit 1ee7f77 with its own model, loss, and configuration code, unmodified, in two upstream stages: the VQVAE (config/train_vqvae.py), then the world transformer with the VQVAE frozen (config/train_occworld.py). Our single-GPU loop in src/training/occworld.py replaces upstream train.py, which requires a distributed launcher, compiled mmcv, and pre-2.6 torch.load defaults. We keep its optimizer (AdamW, lr 1e-3, weight decay 0.01), cosine schedule with warmup, and gradient clipping at 35. Batches are larger and the step budget is shorter than the paper's 200 epochs. Forecast quality is therefore expected to fall below the published OccWorld numbers, and results/ reports ours next to theirs. The schedule used here is 150k VQVAE steps of 16 frames (about 34% of upstream's frame passes) and 100k transformer steps of 4 windows (about 57%). Both use bf16 autocast; the VQVAE forward is compiled with torch.compile, while the transformer runs eagerly because compilation produces a misshaped gradient in its upstream code.

**Import shims.** The pinned dataset module imports mmdet3d boxes, and the planning metric imports a nuScenes Box it never uses. Neither participates in world model inference, and mmdet3d drags in compiled mmcv operators without Blackwell builds. src/world/occworld.py registers minimal stand-ins in sys.modules within the upstream import scope. The box stand-in exposes the raw tensor, which is the only attribute upstream reads.

**History-only rollout.** forward_autoreg_with_pose encodes a full 12-frame window but seeds autoregression only from the five observed codes. The adapter pads the window with copies of the last observation, so future occupancy never enters the model, and leaves future displacement entries zero.

**Privileged conditioning.** Upstream conditions the rollout on future driving commands (gt_mode, the nuScenes planning command), and its rel_poses input includes the last observed frame's next-step ego displacement. Both are kept because OccWorld was trained with them. The teacher and reward model, which consume the rollout, are therefore privileged, and this is acceptable only because neither is deployed. The student receives only the current command and the two past displacements, the standard nuScenes open-loop inputs.

**World cache.** OccWorld is frozen and its rollout decodes by argmax, so a single offline pass gives exactly the outputs that online calls would give; scripts/verify_occworld.py checks this equality. Per window we store the predicted codebook indices [6, 50, 50], BEV obstacle and drivable masks [6, 2, 200, 200], and predicted per-step ego displacements [6, 2]. Teacher, reward, and distillation training read this cache instead of calling OccWorld. WPT's online distillation is preserved in the sense the paper uses: the frozen teacher and reward model run in the loop on every student step.

**World features.** WPT's F^w_{t+1} is Drive-OccWorld's predicted BEV feature. The OccWorld counterpart is the predicted quantized latent at 50 x 50 x 128, the exact tensor its VAE decoder consumes. We obtain it by looking up the cached indices in the frozen codebook, which src/world/encoder.py stores as a buffer.

**Train split optimism.** The world model is trained on the nuScenes train split, so the rollouts used to train the teacher and reward model come from scenes it has seen. Validation rollouts are not affected.

## Data and coordinates

**Windows.** We follow upstream's validation traversal: 12-frame windows, len(scene) - 12 per scene. The planning origin is window index 4, the last observed frame. Expert targets are that frame's gt_ego_fut_trajs (six 0.5 s steps in its LiDAR frame) with gt_ego_fut_masks. Upstream's own planning evaluation instead treats index 5 as the origin and chains each frame's first-step displacement; we do not reproduce that convention.

**Frames.** Trajectories live in the origin LiDAR frame (x right, y forward). Occ3D grids are ego frame, 200 x 200 x 16 at 0.4 m over [-40, 40] m. Points move between the two through the origin's lidar2ego extrinsics. Each predicted grid is centered on the ego at its own time step. A candidate point at step k is therefore shifted by the cumulative predicted ego displacement before lookup, ignoring yaw as upstream does.

## Policies

**Observations.** The paper's policies consume multi-view camera images through a BEVFormer encoder. Here the student consumes the five observed Occ3D occupancy grids plus the command and ego history, the same observation modality OccWorld uses. This keeps training within a single GPU budget. The resulting student is not camera based, and its numbers are not comparable to camera based rows in WPT Table 1.

**Teacher.** Following Eq. 8, the teacher's planning decoder attends to the predicted world tokens (six steps of 25 x 25 tokens with step embeddings), not to current features. It has K = 6 mode queries, trained by winner-takes-all L1, the usual multi-modal planning objective. The paper names neither K nor the teacher's imitation loss.

**Student.** One plan query refined by two decoder layers over 25 x 25 occupancy tokens; dimension 128.

## Reward model

**Architecture.** Per candidate, a trajectory MLP embedding cross-attends to the reward model's own world tokens (Eq. 9), followed by an imitation logit and five simulation logits (Fig. 3). Candidates do not attend to each other. This keeps each candidate's network output independent of the rest of the set, which matters when the student trajectory is appended in Eq. 16.

**Eq. 11.** As printed, softmax(-d_i / sum_j -d_j) equals softmax(d_i / sum_j d_j). The signs cancel, the target favors the farthest candidate, and the scale keeps it nearly uniform. We use softmax(-d_i / tau) with tau = 1 m, where d_i is the mean L2 distance over valid steps. The literal form remains available as reward.im_target: literal.

**Eq. 14.** Evaluated on predicted probabilities, with logsigmoid for NC and DAC. The TTC, EP, and comfort blend is divided by 12 as in the NAVSIM PDM score, so its logarithm is at most zero. alpha = (1, 1, 1, 1) because the paper gives no values.

**Selection.** Sec. 3.4 selects tau*_T by the final reward. We use Eq. 14 for both selection and distillation rather than the linear form of Eq. 10.

**Simulation targets (App. 6.3).**
- Rules are evaluated on the world model's predicted occupancy. Obstacles are columns containing classes 0 to 10.
- Occ3D labels only observed voxels, so unseen road reads as free space. A column is therefore off road only when it holds an explicit non road class (other flat, sidewalk, terrain, manmade, vegetation) and neither road nor an object. A 3 x 3 closing removes isolated off road cells.
- The ego footprint is 4.084 x 1.85 m, offset 0.5 m forward like upstream's collision metric, and sampled at 15 points. Points beyond the 40 m grid count as drivable and collision free.
- EP is the final forward displacement, normalized by the set maximum when that exceeds 5 m (Eq. 19). "Batch" is read as the candidate set of one scene, matching NAVSIM's per-scene normalization.
- TTC extends the final pose 10 m forward (Eq. 21).
- Comfort applies the NAVSIM thresholds to derivatives of a least squares cubic through the origin and the six positions. Raw third differences at 2 Hz amplify annotation noise beyond the jerk bounds, and NAVSIM likewise filters before differentiating.

**Target validation.** Before any training, the rules were checked on ground truth occupancy for 1,500 validation windows, with the expert trajectory scored against the ground truth future frames. The expert scores NC 0.968, DAC 0.991, and Comf 0.950. A version drifting laterally by 3 m falls to DAC 0.68 (0.24 at 6 m). Swapped or unrotated frame conventions reduce expert DAC on the current frame map, which fixes the LiDAR to Occ3D transform. The first versions of the rules gave the expert DAC 0.645 and Comf 0.642, which motivated both changes above.

**Training.** The teacher and reward model train jointly for 12 epochs. Candidates are detached before scoring, so the reward losses update only the reward model and the teacher is shaped by imitation alone, as in Baseline-T.

## Distillation

**Eq. 15.** The student query (dimension 128) passes through a linear projector to the teacher dimension (256). It is matched by an unsquared L2 norm to the teacher's refined query for the reward selected mode. The projector exists only during training.

**Eq. 16.** A single student trajectory makes the imitation softmax of Eq. 14 identically one. We therefore append the student trajectory to the teacher's candidate set and score all K + 1 jointly in the same predicted world. The loss is the absolute difference between the student's reward and that of tau*_T in this shared set, with the teacher reward detached. The reward model is frozen; gradients reach the student through its trajectory input. The literal objective also pulls a student whose reward exceeds the teacher's back down; we keep it as printed.

**Schedule.** Table 8 lists separate teacher and student runs, which we read as two stages: teacher and reward first, then the student with both frozen. Loss weights are plan 1.0, policy 1.0, reward 0.5, fixed a priori. Ablations zero one weight at a time.

## Evaluation

L2 and collision rate come from upstream's PlanningMetric, fed with the origin frame's agent boxes and futures. We report the per-time value at each horizon (OccWorld and WPT tables) and the average up to the horizon (ST-P3 and VAD). Windows whose expert future is incomplete are skipped. Latency is measured at batch size 1 on the same GPU. The student path is measured alone; the teacher reference includes the OccWorld rollout.
