# Top-K adaptive RQ fixed-width RLE-mask results

## Evaluation contract

- Checkpoint: `quality_v2_B_larger_rate047_rq_ema_unet2_ds8x2_k4-2_d2-2`
- Dataset: 24 sorted Kodak-256 images
- Selection: exact Top-K independently for every image and scale
- Activation targets: 0%, 10%, ..., 100%
- Channel: Sionna 5G LDPC `k=128`, `n=256`, rate 1/2; BPSK; AWGN; SNR 0 dB
- Seed: 42, reset for every activation point
- Mask order: row-major raster order
- RLE: one start bit followed by fixed-width binary `(run_length - 1)`
- Run-length widths: shallow 10 bit (1024 tokens), deep 8 bit (256 tokens)
- RLE is forced; there is no raw-mask fallback

For a mask with \(R_s\) runs:

\[
B_{\mathrm{mask},0}=1+10R_0,\qquad
B_{\mathrm{mask},1}=1+8R_1.
\]

The shallow second-stage index uses 2 bit per active token; the deep
second-stage index uses 1 bit per active token. Segment boundaries, source
segment lengths, shapes, codebook sizes, and transmitted active counts are
assumed known out of band, matching the previous explicit-mask evaluation.
Their framing overhead is not counted.

## Per-scale source-bit results

All values are averages per image. `Old L2` is raw mask plus active
second-stage indices. `RLE L2` is the forced RLE mask plus the same active
indices.

| Active | Shallow/deep active | Mean runs shallow/deep | RLE mask shallow/deep | Old L2 shallow/deep | RLE L2 shallow/deep |
|---:|---:|---:|---:|---:|---:|
| 0% | 0 / 0 | 1.000 / 1.000 | 11.000 / 9.000 | 1024 / 256 | 11.000 / 9.000 |
| 10% | 102 / 26 | 133.875 / 36.750 | 1339.750 / 295.000 | 1228 / 282 | 1543.750 / 321.000 |
| 20% | 205 / 51 | 217.417 / 66.917 | 2175.167 / 536.333 | 1434 / 307 | 2585.167 / 587.333 |
| 30% | 307 / 77 | 265.000 / 90.000 | 2651.000 / 721.000 | 1638 / 333 | 3265.000 / 798.000 |
| 40% | 410 / 102 | 286.167 / 104.375 | 2862.667 / 836.000 | 1844 / 358 | 3682.667 / 938.000 |
| 50% | 512 / 128 | 298.000 / 110.208 | 2981.000 / 882.667 | 2048 / 384 | 4005.000 / 1010.667 |
| 60% | 614 / 154 | 293.792 / 105.167 | 2938.917 / 842.333 | 2252 / 410 | 4166.917 / 996.333 |
| 70% | 717 / 179 | 267.208 / 93.292 | 2673.083 / 747.333 | 2458 / 435 | 4107.083 / 926.333 |
| 80% | 819 / 205 | 212.042 / 74.042 | 2121.417 / 593.333 | 2662 / 461 | 3759.417 / 798.333 |
| 90% | 922 / 230 | 129.208 / 44.083 | 1293.083 / 353.667 | 2868 / 486 | 3137.083 / 583.667 |
| 100% | 1024 / 256 | 1.000 / 1.000 | 11.000 / 9.000 | 3072 / 512 | 2059.000 / 265.000 |

The raw mask itself is always 1024 bit at the shallow scale and 256 bit at
the deep scale. Fixed-width RLE reduces both only at the 0% and 100%
endpoints. At every target from 10% through 90%, both RLE masks are larger
than their raw masks.

## Complete packet and lossy quality results

The first stage is fixed at 2304 source bit per image. Coded lengths include
LDPC padding independently for every image and logical segment. Negative
savings mean that forced RLE increased the rate.

| Active | Raw source | RLE source | Source saving | Raw coded | RLE coded | Coded saving | PSNR | MS-SSIM | RLE bit errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 3584.000 | 2324.000 | +35.156% | 7168.000 | 5120.000 | +28.571% | 21.060909 | 0.814531463 | 0 |
| 10% | 3814.000 | 4168.750 | -9.301% | 7936.000 | 8885.333 | -11.962% | 21.823097 | 0.836399474 | 5 |
| 20% | 4045.000 | 5476.500 | -35.389% | 8448.000 | 11520.000 | -36.364% | 22.312665 | 0.848829837 | 0 |
| 30% | 4275.000 | 6367.000 | -48.936% | 8704.000 | 13152.000 | -51.103% | 22.584387 | 0.855973184 | 0 |
| 40% | 4506.000 | 6924.667 | -53.677% | 9216.000 | 14336.000 | -55.556% | 22.762457 | 0.860992812 | 0 |
| 50% | 4736.000 | 7319.667 | -54.554% | 9472.000 | 14912.000 | -57.432% | 22.369766 | 0.854459595 | 53 |
| 60% | 4966.000 | 7467.250 | -50.367% | 10240.000 | 15477.333 | -51.146% | 22.936921 | 0.864912144 | 12 |
| 70% | 5197.000 | 7337.417 | -41.186% | 10752.000 | 15242.667 | -41.766% | 23.169396 | 0.868905973 | 0 |
| 80% | 5427.000 | 6861.750 | -26.437% | 11008.000 | 14112.000 | -28.198% | 23.061067 | 0.866108739 | 12 |
| 90% | 5658.000 | 6024.750 | -6.482% | 11520.000 | 12480.000 | -8.333% | 23.472433 | 0.871473412 | 0 |
| 100% | 5888.000 | 4628.000 | +21.399% | 11776.000 | 9728.000 | +17.391% | 23.518708 | 0.870878600 | 0 |

The largest RLE packet occurs at 60% rather than exactly 50%, because packet
rate also contains the growing active second-stage index payload. The largest
mask-only RLE size is near 50%.

## Comparison with the previous explicit-mask 0 dB run

| Active | Explicit-mask PSNR | RLE PSNR | Delta PSNR | Explicit-mask MS-SSIM | RLE MS-SSIM | Delta MS-SSIM |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 21.051749 | 21.060909 | +0.009161 | 0.814069099 | 0.814531463 | +0.000462364 |
| 10% | 21.878882 | 21.823097 | -0.055785 | 0.838095600 | 0.836399474 | -0.001696125 |
| 20% | 22.305230 | 22.312665 | +0.007435 | 0.848086892 | 0.848829837 | +0.000742945 |
| 30% | 22.552088 | 22.584387 | +0.032298 | 0.855558263 | 0.855973184 | +0.000414921 |
| 40% | 22.762457 | 22.762457 | +0.000000 | 0.860992812 | 0.860992812 | +0.000000000 |
| 50% | 22.851077 | 22.369766 | -0.481311 | 0.864328913 | 0.854459595 | -0.009869318 |
| 60% | 22.980425 | 22.936921 | -0.043504 | 0.866058644 | 0.864912144 | -0.001146500 |
| 70% | 23.169396 | 23.169396 | +0.000000 | 0.868905973 | 0.868905973 | +0.000000000 |
| 80% | 23.335389 | 23.061067 | -0.274321 | 0.870274333 | 0.866108739 | -0.004165594 |
| 90% | 23.407148 | 23.472433 | +0.065285 | 0.867005775 | 0.871473412 | +0.004467637 |
| 100% | 23.518708 | 23.518708 | +0.000000 | 0.870878600 | 0.870878600 | +0.000000000 |

This is one seeded channel realization. Because RLE changes segment lengths,
later channel samples are not paired bit-for-bit with the explicit-mask run.
Small positive or negative deltas at points without RLE errors are therefore
ordinary channel-sample variation, not a coding gain. The large losses at
50% and 80% coincide with RLE length-field errors.

## RLE error propagation

No start-value bit failed. All 82 residual RLE source-bit errors occurred in
run-length fields:

- 10%: shallow 1 error and deep 4 errors; two scale frames affected.
- 50%: shallow 53 errors; six of 24 shallow frames affected; 755 semantic
  shallow-mask token errors; 186 zero-filled and 165 truncated refinement
  positions.
- 60%: shallow 6 errors and deep 6 errors; one frame per scale affected; 29
  shallow and 112 deep semantic mask errors.
- 80%: shallow 12 errors; one frame affected; 88 semantic mask errors and 88
  zero-filled refinement positions.

At 50%, the shallow semantic mask BER is 3.0721% even though only 53 of
71,544 transmitted shallow RLE source bits were wrong. A corrupted run length
moves subsequent run boundaries, so one residual channel error can alter many
mask tokens. This is the main robustness weakness of direct RLE.

## Conclusion

Forced fixed-width RLE is useful only for the degenerate all-STOP and
all-active endpoints on these Top-K masks. It increases source and coded rate
throughout 10%-90%, and length errors can amplify into many mask errors. A
practical transport should at minimum select raw versus RLE per image and
scale, and should protect the RLE start/length fields more strongly than the
ordinary index payload.
