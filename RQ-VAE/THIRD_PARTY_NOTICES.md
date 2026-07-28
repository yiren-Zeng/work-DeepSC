# Third-Party Notices

## Kakao Brain RQ-VAE

`models/rq_ema_quantizer.py` is derived from the EMA vector-quantization and
residual-quantization implementation in Kakao Brain's
[`rq-vae-transformer`](https://github.com/kakaobrain/rq-vae-transformer)
repository.

- Upstream commit: [`341395e562ac347f5eb62db9f5f08b9f2cc42a60`](https://github.com/kakaobrain/rq-vae-transformer/tree/341395e562ac347f5eb62db9f5f08b9f2cc42a60)
- Source file: [`rqvae/models/rqvae/quantizations.py`](https://github.com/kakaobrain/rq-vae-transformer/blob/341395e562ac347f5eb62db9f5f08b9f2cc42a60/rqvae/models/rqvae/quantizations.py)
- Interface reference: [`rqvae/models/rqvae/rqvae.py`](https://github.com/kakaobrain/rq-vae-transformer/blob/341395e562ac347f5eb62db9f5f08b9f2cc42a60/rqvae/models/rqvae/rqvae.py)
- Copyright: Copyright (c) 2022-present, Kakao Brain Corp.
- License: Apache License 2.0; see [`LICENSE.apache-2.0`](LICENSE.apache-2.0).

The derived module was modified to quantize existing NCHW U-Net feature maps
directly, return the host project's loss/feature/index tuple, and expose
codebook diagnostics. It does not include the upstream Encoder, Decoder,
RQ-Transformer, or pretrained weights.

The upstream repository separately licenses its stage-2 pretrained weights
under CC-BY-NC-SA 4.0. No such weights are included in this project, so that
weights license is not incorporated here.
