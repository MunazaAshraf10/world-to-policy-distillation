# Implementation notes

## Stage 1 contract

OccWorld is pinned to commit 1ee7f77ecc4c984a4f7f6411d95c2e6e73806b6e. The adapter uses its TransVQVAE.forward_autoreg_with_pose method, with the official evaluation configuration's start_frame=0, mid_frame=5, end_frame=11. The model architecture retains num_frames=15. The dataset returns 12 occupancy grids, while the rollout uses five history frames and predicts six future frames; the extra last grid is retained for upstream API compatibility.

The actual prediction origin is window index 4. The upstream dataset's sample_idx uses index 5 for this configuration, and its scene_token is the scene's fixed fourth-index token rather than a general window identifier. Reports therefore preserve all window tokens and explicitly record history_last_token separately. No temporal shift is silently introduced.

Inputs retain the native spatial ordering [200,200,16], semantic labels 0–17, and class 17 as free space. Geometry is not transformed in Stage 1. Before reward integration, coordinate frames, voxel centers, drivable classes, ego footprints, and time alignment must be established explicitly.

The exposed motion values are per-step displacements. They are not cumulative positions and are not the WPT teacher's candidate trajectories. OccWorld's three maneuver modes are distinct from a later teacher's configurable candidate count K.

## Conditioning and isolation

Upstream autoregression uses future gt_mode entries to select pose modes and feed them into subsequent world predictions. Stage 1 preserves this behavior. The upstream rel_poses field is built from each frame’s gt_ego_fut_trajs[0], including the last history frame’s next-step displacement. This annotated motion condition is also retained. Passing the verification does not establish observation-only forecasting or camera-only planning.

Upstream also accepts and encodes a sequence containing future occupancy targets. Only history codes seed its autoregressive predictor. The verifier independently perturbs all occupancy values and displacement annotations from mid_frame onward and checks that every returned prediction remains unchanged, with maneuver modes fixed. This is an empirical guard against future-target leakage on the tested sample, not a general proof.

Preflight tests PyTorch CUDA matrix multiplication, OpenMMLab imports, and the MMCV rotated-box operator on CPU. The prebuilt MMCV 2.0.1 CUDA operator failed on this H100 with “no kernel image is available.” OccWorld's inspected rollout uses PyTorch operations rather than MMCV CUDA operators. Any later policy that requires those operators needs an H100-compatible build and a separate CUDA operator test.

## Checkpoints and freezing

The upstream evaluation configuration uses revise_ckpt=3, which routes loading to the VAE only. The adapter deliberately does not use that setting: it requires a full matching state dictionary for the VAE, transformer, pose encoder, pose decoder, and buffers. Only a uniform module. prefix is normalized. Missing, unexpected, malformed, nonfinite, or incompatible tensors are rejected before model state changes.

All OccWorld parameters are frozen, stale gradients are cleared, and every module remains in evaluation mode. Prediction executes under torch.no_grad(). The upstream implementation's own detaches are retained. No optimizer, training loop, synthetic fallback, or partial checkpoint fallback exists.

## Reporting

The JSON report uses a deterministic filename and sorted keys. Runtime measurements naturally vary. It records full configuration, model configuration, code hashes, repository revisions and dirty state, checkpoint and metadata hashes, sample identity, conditioning, output statistics, and checks. Existing reports are preserved unless overwrite is explicitly requested.

Reported time covers four forward calls plus assertions, excluding checkpoint/data loading. Peak allocated VRAM covers the verification sequence and retained reference tensors. Neither measurement is a deployment latency benchmark.

## Deferred WPT decisions

- Eq. 11 divides negative distances by their negative sum, cancelling the signs and apparently increasing target preference with error. The all-zero-distance case is also undefined. Preserve this discrepancy in review before choosing a corrected or literal experimental variant.
- Eq. 15 specifies an L2 norm, not an interchangeable choice among MSE, cosine distance, and Smooth L1. Refined query identity, alignment, and reduction dimensions need specification.
- Eq. 16 requires comparable rewards for a teacher candidate and a single student trajectory. A softmax over the student's singleton candidate set would always produce one; reward normalization must be resolved.
- Teacher/reward update schedules, target detachment, gradients through the learned reward function, and loss weights remain unspecified here.
- Predicted feature extraction and adaptation into teacher/reward dimensions require a later interface decision; semantic logits are not declared equivalent to WPT's latent world features.

Later work proceeds through a student baseline, multimodal teacher, reward supervision, each distillation loss independently, joint composition, and ablations. No stage beyond frozen verification is implemented or automatically launched.
