# Cars196 Seq2Seq RAQ training

The training adapter calls the original `RAQVAE_TWO.training_step`,
`validation_step`, `configure_optimizers`, and target-K samplers directly.
No changes have been made to `nets/model.py`, `nets/quantizer.py`,
`nets/cbk2cbk.py`, or `nets/enc_dec.py`.

## Configuration

- Two spatial scales: bottom H/4 x W/4 and top H/8 x W/8.
- One shared source codebook: K=64, embedding dimension=64.
- One shared two-layer LSTM Seq2Seq generator, hidden=64, dropout=0.5.
- Original n_hid=64, two residual blocks per encoder/decoder.
- Train: one uniformly sampled integer K in [2,64] per batch, shared by both scales.
- Validation: one random K from [2,4,8,16,32,64] per batch, original exponent sampler.
- Original objective: MSE_src + MSE_trg + latent_src + latent_trg. Each latent term
  sums both scales, with the original 0.25 commitment coefficient and codebook loss.
- Original codebook EMA updates, straight-through estimator and optimizer behavior.
- AdamW lr=0.0005, default betas=(0.9,0.999), eps=1e-8, weight_decay=0.01.
- 200 epochs, batch=24, validation batch=64, FP32, no channel training or scheduler.
- Validation every two epochs, as in the original DD training entry point.
- Best five checkpoints ranked by `val_recon_loss` (MSE_src + MSE_trg); also `last.ckpt`.
- Seed 42, eight loader workers, one GPU.
- `Cars196/train_data` and `Cars196/val_data`: reference `shiyan` loader, including
  Resize(256), random/center crop, training flip, normalization to [-1,1].
- No Kodak data is used for training, validation or checkpoint selection.

Source K, K range, batch size, epochs and data source are experiment hyperparameters;
the computational method and losses are unchanged. Output batch tensors are wrapped
as `(batch,)` solely to meet the original training step's `(images, labels)` interface.
Extra epoch-aggregate logging and finite-gradient/loss checks do not change gradients.
There is no initial sanity validation, so the first optimizer step follows seed setup
and initialization without consuming extra random validation K draws.

## Environment and launch

All output, dependency, cache and temporary directories are inside this project.
`shiyan/data/datasets.py` and `shiyan/utils/reproducibility.py` are imported read-only;
Python bytecode writes are disabled. Source hashes and image counts are saved per run.

The existing `work` environment supplies torch 2.7.1+cu118 and torchvision 0.22.1.
Project-local `.runtime/deps` supplies repository-pinned pytorch-lightning 1.9.5,
pytorch-msssim 1.0.0, torchmetrics 1.3.0 and lightning-utilities 0.10.0.
A process-local `np.Inf = np.inf` compatibility alias accommodates NumPy 2.x.

```bash
cd /workspace/yi/work/RAQ-VAE-main
/home/yi/.conda/envs/work/bin/python -B scripts/launch_seq2seq_cars.py --gpu 0 --smoke
/home/yi/.conda/envs/work/bin/python -B scripts/launch_seq2seq_cars.py --gpu 0
```

The launcher prints the PID and a fresh run directory. `train.log`, `status.json`,
`config.json`, `first_step_gradients.json`, CSV/TensorBoard metrics and checkpoints
are stored there. `status.json` reports running, completed or failed after trainer
startup; failures before trainer startup appear in `train.log`.

Resume uses the same run's full checkpoint and the original hyperparameters:

```bash
CUDA_VISIBLE_DEVICES=0 /home/yi/.conda/envs/work/bin/python -B train_seq2seq_cars.py \
  --output /workspace/yi/work/RAQ-VAE-main/runs/YOUR_RUN \
  --resume /workspace/yi/work/RAQ-VAE-main/runs/YOUR_RUN/checkpoints/last.ckpt
```

Resume restores model, optimizer, epoch, global step and main-process RNG state.
Persistent loader workers' augmentation RNG is not serialized, so resumed training
is not guaranteed bit-for-bit identical to uninterrupted training. Only locally
produced, trusted checkpoints should be passed to `--resume`.

## Transmission budgets for later testing

The reference ratio is complex channel symbols / (3*H*W), with LDPC k=192,n=256,
rate 3/4 and QPSK (two coded bits per symbol). For 256x256 RGB images:

- Shared K=2: 5120 source indices * 1 bit = 5120 payload bits.
- LDPC requires ceil(5120/192)=27 blocks: 64 padding bits, 6912 coded bits.
- QPSK requires 3456 complex symbols: ratio 3456/196608 = 0.017578125 = 1/56.8889.
- A 1/48 budget allows 4096 symbols / 32 LDPC blocks / 6144 input bits. K=2 fits
  within that budget. Exact utilization requires 1024 non-information padding bits
  in addition to the 5120 source bits; padding must be disclosed separately.
- A 1/64 budget allows 3072 symbols / 24 LDPC blocks / 4608 input bits. K=2 does not fit.
  Lossless entropy coding would need at least 10% savings, plus any metadata overhead.
  Whether that is possible depends on the trained indices and must be measured.

No rate-padding or entropy-coding extension is implemented as part of training.
Tests must reuse the reference Kodak loader and exact PSNR/MS-SSIM preprocessing
and aggregation, and report actual payload, padding, coded bits and channel symbols.
