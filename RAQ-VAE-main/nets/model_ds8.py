"""Isolated 1/8 + 1/16 spatial-scale variant of the original RAQVAE_TWO."""
import torch
from torch import nn

from nets.blocks import ResBlock
from nets.model import RAQVAE_TWO


class EncoderStride8(nn.Module):
    """Original Encoder layout with its single downsampling convolution set to x8."""

    def __init__(self, in_channel, channel, n_res_block, n_res_channel):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, kernel_size=16, stride=8, padding=4)]
        blocks.extend(ResBlock(channel, n_res_channel) for _ in range(n_res_block))
        blocks.append(nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class DecoderStride8(nn.Module):
    """Original Decoder layout with its final transposed convolution set to x8."""

    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel):
        super().__init__()
        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]
        blocks.extend(ResBlock(channel, n_res_channel) for _ in range(n_res_block))
        blocks.extend([
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(channel, out_channel, kernel_size=16, stride=8, padding=4),
        ])
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class RAQVAE_TWO_DS8(RAQVAE_TWO):
    """RAQVAE_TWO with bottom/top scales 1/8 and 1/16 instead of 1/4 and 1/8.

    All inherited encode/decode, shared codebook, Seq2Seq, loss, K sampling and
    optimizer behavior remain identical to the original class.
    """

    def __init__(self, args):
        super().__init__(args)
        self.encoder_b = EncoderStride8(
            self.input_channels, self.channel, self.n_res_block, self.n_res_channel
        )
        self.decoder = DecoderStride8(
            self.embedding_dim * 2, self.input_channels, self.channel,
            self.n_res_block, self.n_res_channel,
        )

