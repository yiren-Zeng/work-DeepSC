"""Strict RAQVAE_ONE baseline with only the spatial stride changed to x2.

Everything that defines the RAQVAE_ONE method is inherited unchanged from
``nets.model.RAQVAE_ONE``: source/target quantizers, the step-by-step LSTM
Seq2Seq codebook generator, random integer target-K sampling, reconstruction
and latent losses, and AdamW configuration. The source/target K constraints
are fixed to the comparison experiment (256 and 32..2048 respectively).
"""

from torch import nn

from nets.blocks import ResBlock
from nets.enc_dec import Encoder
from nets.model import RAQVAE_ONE


class DecoderStride2(nn.Module):
    """Original one-scale decoder layout with x2 output upsampling."""

    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        blocks.extend(
            ResBlock(channel, n_res_channel) for _ in range(n_res_block)
        )
        blocks.extend(
            [
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(
                    channel,
                    out_channel,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            ]
        )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.blocks(inputs)


class RAQVAE_ONE_DS2_SRC256_K32_2048(RAQVAE_ONE):
    """Original single-scale RAQVAE_ONE at a 128x128 latent resolution."""

    def __init__(self, args):
        if int(args.num_embeddings) != 256:
            raise ValueError("this baseline requires source K=256")
        if int(args.num_embeddings_min) != 32:
            raise ValueError("this baseline requires minimum target K=32")
        if int(args.num_embeddings_max) != 2048:
            raise ValueError("this baseline requires maximum target K=2048")
        if int(args.embedding_dim) != 64:
            raise ValueError(
                "the original CdBk2CdBk implementation requires embedding_dim=64"
            )

        super().__init__(args)

        # These are the only two model-structure changes from RAQVAE_ONE.
        self.encoder = Encoder(
            self.input_channels,
            self.channel,
            self.n_res_block,
            self.n_res_channel,
            stride=2,
        )
        self.decoder = DecoderStride2(
            self.embedding_dim,
            self.input_channels,
            self.channel,
            self.n_res_block,
            self.n_res_channel,
        )
