# Two-scale Seq2Seq RAQ: source K256, target K4..4096

Independent experiment; no existing scripts or nets files are edited.

- Original `RAQVAE_TWO`: bottom H/4 x W/4, top H/8 x W/8; 5120 indices per 256x256 image.
- Source codebook: 256 entries. Embedding dimension 64, n_hid 64, two ResBlocks.
- Target sizes: {4,8,16,32,64,128,256,512,1024,2048,4096}.
- Both training and validation sample one of these 11 sizes uniformly per batch,
  using the same size/codebook for top and bottom. This is the requested sampler
  change from the original training sampler over all integers.
- Original forward, losses, EMA, Seq2Seq LSTM, gradient flow and AdamW remain intact.
- From scratch, Cars196, 256x256; 200 epochs, train batch 24, val batch 64,
  FP32, AdamW lr=5e-4, seed 42, eight loader workers, validation every two epochs.
- Top five validation checkpoints plus last; monitor MSE_src + MSE_trg.
- No channel during training. Test defaults: Kodak, K4, LDPC 3/4 (192,256), QPSK,
  AWGN, SNR 6 dB, channel/model seeds 42, CSCR off.

Launch on physical GPU 4 (UUID binding prevents CUDA device-order ambiguity):

```bash
cd /workspace/yi/work/RAQ-VAE-main
/home/yi/.conda/envs/work/bin/python -B scripts/launch_seq2seq_cars_src256_k4_4096.py --gpu 4
```

The launcher detaches a supervisor with its own session and log; the supervisor
detaches its training worker. Terminal closure does not terminate either process.
A file lock prevents duplicate supervisors for this experiment on the same GPU.
Progress is in `train.log`, `status.json`, `csv/`, and `tensorboard/`.
Supervisor heartbeat, PID, retry count, exit code and crash log tail are in
`supervisor.json` and `supervisor.log`.

Full recovery checkpoints are saved before the first batch, every 50 optimizer
steps and at epoch end. Two recovery slots and all best/last checkpoints are
written with fsync and atomic replacement. The supervisor automatically retries
unexpected exits, with backoff capped at 300 seconds. Training validates available
checkpoints and restores the greatest saved global step, falling back to an older
complete checkpoint if necessary. Model, optimizer, loop and main-process RNG
states are restored; unsaved steps can be replayed and DataLoader worker RNG is
not restored exactly. Server shutdown/hardware failure cannot be prevented by a
local process; after reboot, explicitly restart the supervisor:

```bash
/home/yi/.conda/envs/work/bin/python -B scripts/launch_seq2seq_cars_src256_k4_4096.py \
  --gpu 4 --resume-run /absolute/path/to/run
```

To stop without automatic restart, create a file named `STOP` in that run directory.
The supervisor requests a graceful stop after the current batch. Remove that marker
before explicitly resuming. Do not use a worker PID alone to intentionally stop:
the supervisor interprets that exit as a recovery request.

Test the latest formal run's best validation checkpoint (available after validation):

```bash
bash scripts/test_seq2seq_src256_k4_4096_channel.sh --target-k 4096
```

Use `RAQ_RUN_DIR=/absolute/path/to/run` to select a different run, `GPU_ID` to
change the test GPU, or `--checkpoint /absolute/path/to/checkpoint` for an explicit
snapshot. Extra CLI arguments override script defaults. `--dry-run` validates
the rate plan without creating test artifacts. Avoid competing with training on
GPU 4. Original experiment scripts continue using their own run directories.

`--smoke` on the training entry point checks K4 and K4096 at the original batch
sizes in one short epoch. To check a mid-epoch recovery explicitly, first run
`--smoke --max-steps 1`, then use `--smoke --resume
/absolute/path/to/run/checkpoints/recovery_1.ckpt` with the same output directory.
The recovery slot is used because the artificial max-steps stop also produces an
end-of-epoch last checkpoint. Normal supervisor recovery chooses the most advanced
complete checkpoint automatically.

## Started run and verification

Formal run: `runs/seq2seq_src256_k4_4096_20261020_000421_431635`.
Physical GPU 4 UUID: `GPU-5f174e61-459c-e68c-b9fd-afb7d3d3b9ad`.
Initial supervisor PID: 2542014; initial training worker PID: 2542015.
Worker PID can change after recovery; read `supervisor.json` for the current one.

Preflight passed four new CPU tests and 18 existing PHY/rate tests, plus K4/K4096
evaluation dry runs. On GPU 4, K4096 trained with batch 24 and validated with batch
64 without OOM (peak allocated memory 8687 MiB). A separate supervised smoke run
had its worker forcibly killed after step 1 (exit -9); the supervisor restarted it,
restored the complete checkpoint and finished at step 3. Evidence is stored in
the formal run's `preflight_verification.json`; the smoke run and detailed logs
are retained under `runs/smoke_seq2seq_src256_k4_4096_20261020_000300_301398`.
All 51 pre-existing source/document/script files checked before construction
were unchanged by this work.
