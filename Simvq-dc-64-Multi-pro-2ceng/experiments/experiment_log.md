# Experiment Log

## Target

- Dataset for final quality: Kodak (24 images).
- Transmission test: LDPC rate 0.5, BPSK, SNR values `0, 3, 6, 9, 12` dB.
- BPG reference currently supplied by `../BPG/bpg_test_output.log`:
  `PSNR=32.30 dB`, `MS-SSIM=0.9777` for
  `kodak-decompress-0.5-bpsk-ratio-0.28`.
- A fair claim of outperforming BPG additionally requires confirming which
  SNR produced that BPG folder and matching effective BPP/channel usage.

## Runs

| Run | Started | Source | Configuration or change | Status | Result |
| --- | --- | --- | --- | --- | --- |
| observed-001 | 2026-05-26 | already running when tracking began | Two-layer SimVQ checkpoint with `K=[64,64]`; active process reported 500 planned epochs, while on-disk config had changed to `K=[128,128]` and 400 epochs | stopped after exporting 90 completed epochs because its no-channel upper bound is below BPG; stdout recovered from deleted descriptor; best epoch is 79 (`Val Recon Loss=0.0097922534`) | BPP `0.4688`; no-channel PSNR/MS-SSIM `26.9998 / 0.9417`; real-chain at SNR 0 dB `26.9464 / 0.9404` |
| quality-v1-k64 | 2026-05-26 | new quality improvement trial | `K=[64,64]`, lighter VQ weights `[0.25,0.5]` to `[0.25,0.25]`, skip dropout `0.1` to `0.0` by 40% of training, LR `5e-5`, micro-batch `24`, isolated checkpoint directory | running; epoch 2 complete, epoch 3 in progress | best Val Recon Loss improved from epoch 1 `0.03333480` to epoch 2 `0.02344946`; scheduled quality screening starts after epoch 5 |

## Changes

| Change | Date | Description | Reason |
| --- | --- | --- | --- |
| tracking-001 | 2026-05-26 | Preserve timestamped training logs and resumed checkpoints in `run_train.sh`; append per-epoch training metrics in `train.py`; allow checkpoint/JSON and fast no-channel upper-bound selection in evaluation scripts; save resume state after updating the best validation threshold. | Make repeated trials measurable without destroying previous evidence, screen poor candidates cheaply, and avoid stale best-model state after resume. |
| model-001 | 2026-05-26 | Add `quality-v1-k64`: preserve baseline `K=[64,64]`/BPP while reducing early skip dropout and VQ-loss dominance, increasing generator LR, and separating checkpoints. | Baseline cannot reach BPG even before channel corruption, so its reconstruction optimization must improve before real-chain evaluation is worthwhile. |

## Execution Attempts

| Invocation | Scheme | Completed epochs attributable to invocation | Outcome |
| --- | --- | --- | --- |
| observed-001 | baseline configuration found running | 90 | stopped and archived after poor no-channel ceiling |
| quality-v1-k64-001 | model-001 | 1 | produced the initial checkpoint; interrupted during epoch 2 to detach it cleanly |
| quality-v1-k64-002-resume | model-001 | 0 | background launch did not remain alive after returning from its parent command |
| quality-v1-k64-003-resume | model-001 | in progress from epoch 2 | running in an independent session; monitored automatically |

Counts as of this record: `4` training process invocations observed/started,
`2` model configurations measured or running, and `1` performance-oriented
model/configuration modification introduced in this project.

## Artifacts

- `epoch_metrics.csv`: appended automatically on epochs run after
  `tracking-001` is loaded by a new/resumed training process.
- `logs/train_<run_id>.log`: stdout/stderr from future training launches.
- `logs/baseline_epoch088_*.log`: evaluation begun from the best checkpoint
  observed while the active run had reached epoch 88.
- `variants/preexisting_k128_config.py`: preserved copy of the incompatible
  on-disk `K=[128,128]` configuration found before starting `quality-v1-k64`.
- `monitor_quality.py`: automatically evaluates the best `quality-v1-k64`
  checkpoint at epochs 5, 10, and every 10 epochs thereafter; only runs the
  expensive real-chain test once the no-channel quality ceiling meets BPG.
- `quality_v1_k64_screening.csv`: monitor output table once the first
  screening milestone completes.
