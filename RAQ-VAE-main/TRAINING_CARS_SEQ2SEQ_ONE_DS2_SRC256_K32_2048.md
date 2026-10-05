# RAQVAE_ONE ds2 / source-K256 / target-K32..2048 baseline

This experiment is the LSTM-Seq2Seq RAQ baseline for:

`/workspace/yi/work/Ours-RAQ/scripts/eval/test_src256_raq32_2048_curriculum_ch256_res6_unet1_ds2.sh`

## Comparison definition

- original method family: data-driven `RAQVAE_ONE`;
- one spatial scale, x2 downsampling (`256x256 -> 128x128`);
- 16,384 indices per 256x256 image, exactly matching the Ours-RAQ ds2 rate grid;
- source codebook `K_src=256`;
- training target K: every integer from `32` through `2048`, sampled uniformly
  once per batch by the original `RAQVAE_ONE.sample_train_trg`;
- validation target K: one of `32, 64, 128, 256, 512, 1024, 2048`, sampled
  using the original exponent sampler;
- original baseline capacity is retained by default: `n_hid=64`, `D=64`, two ResBlocks;
- the original step-by-step `CdBk2CdBk` LSTM and original non-chunked
  `Quantizer` are retained;
- no channel is used during training, matching the original RAQVAE method;
- physical AWGN/Rician + LDPC testing uses the same exact-LLR evaluator as the existing RAQ-VAE comparisons.

There is no curriculum sampling in this baseline.

## Training

```bash
cd /workspace/yi/work/RAQ-VAE-main
/home/yi/.conda/envs/work/bin/python -B \
  scripts/launch_seq2seq_one_ds2_src256_k32_2048.py --gpu 4
```

Defaults are 200 epochs, real batch size 24, AdamW at `5e-4`, seed 42,
Cars196 training/validation data, and best-checkpoint selection by
`MSE_src + MSE_trg`.

`D=64` and `n_hid=64` are fixed because they are part of the original method;
in particular, the original `CdBk2CdBk` allocates 64-dimensional outputs.

## Channel evaluation

The shell script finds the latest normal training run.  An explicit run can be
selected with `ONE_DS2_RUN_DIR`.

```bash
cd /workspace/yi/work/RAQ-VAE-main
ONE_DS2_RUN_DIR=/absolute/path/to/run \
GPU_ID=0 \
bash scripts/test_seq2seq_one_ds2_src256_k32_2048_channel.sh
```

Its defaults match the Ours-RAQ evaluation script: target K=256, LDPC 1/2,
BPSK, AWGN, and SNR 0 dB.  Trailing CLI arguments override these defaults:

```bash
bash scripts/test_seq2seq_one_ds2_src256_k32_2048_channel.sh \
  --target-k 32 --ldpc-rate 3/4 --modulation qpsk \
  --channel rician --rician-k-factor 10 --snr 6 --seed 42
```

Aggregate MS-SSIM dB is computed by averaging the 24 raw Kodak MS-SSIM values
first and then applying `-10*log10(1-mean_ms_ssim)`.

## Exact rate identity

For target K, the payload is:

`128 * 128 * log2(K)` bits.

Ignoring mandatory block padding, the reported CBR is:

`log2(K) / (12 * LDPC_rate * modulation_bits)`.

Therefore target K=256, LDPC 1/2, and BPSK gives CBR `4/3`, exactly the same
source-index and physical-channel rate as the matching Ours-RAQ script.
