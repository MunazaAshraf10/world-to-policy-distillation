# Stage 1 verification status

Stage 1 implementation is present. **Pretrained inference acceptance remains blocked by the unavailable full OccWorld checkpoint.** No training was launched and no planning quality or model prediction results are reported.

| Check | Observed result |
| --- | --- |
| Pinned OccWorld source | Initialized at 1ee7f77ecc4c984a4f7f6411d95c2e6e73806b6e; upstream working tree clean |
| Isolated environment | Python 3.8.20, PyTorch 2.0.1+cu118, CUDA 11.8; complete dependency check passed |
| GPU and required imports | PyTorch CUDA execution and OpenMMLab imports/CPU operator passed on NVIDIA H100 PCIe |
| Upstream architecture construction | TransVQVAE built from the pinned evaluation configuration; 72,385,578 parameters; no untrained prediction used as verification |
| Validation metadata | Downloaded the DOME-hosted copy linked in an OccWorld issue; schema matches required fields for 150 scenes |
| Occupancy data | Downloaded the original 2,737,816,708-byte archive; extracted 12 frames from scene-0003 |
| Real upstream data loader | Passed; occupancy int64 [1,12,200,200,16], rel_poses [12,2], gt_mode [12,3] |
| Unit tests | 35 passed |
| Real integration test | 1 skipped because no pretrained checkpoint is available |
| Lint and format | Ruff check and format check passed |
| Missing checkpoint behavior | Verification raises an explicit missing-checkpoint error and produces no success report |
| Pretrained rollout, real parity and temporal isolation | Not run; checkpoint required |

The prebuilt MMCV CUDA rotated-box operator is incompatible with this H100. It is not used by the inspected OccWorld rollout; this limitation is detailed in implementation_notes.md. The preflight result does not certify other MMCV CUDA consumers.

Local evidence is stored in:

- results/occworld_stage1/preflight.json
- results/occworld_stage1/data_check.json
- data/occworld_sample_manifest.json
- requirements-occworld.lock.txt

The data manifest records the archive, metadata, and each extracted occupancy file's SHA256. Setup commands and provenance are in setup.md.

Both original cloud shares reported “Link does not exist.” The repository's model link remains unchanged, and [upstream issue 39](https://github.com/wzzheng/OccWorld/issues/39) has no replacement checkpoint link in its inspected discussion. Searches of upstream issues and public model listings did not establish a verified replacement. The available DOME checkpoint uses a different world model and was not downloaded as a substitute.

To finish Stage 1, supply the actual pretrained OccWorld state dictionary and run the verification command in setup.md. Only a resulting stage1_passed report establishes that the full checkpoint loaded and all real rollout checks succeeded.
