import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.prelu(x)
        x = self.conv2(x)
        return x + residual


class UpSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, up_mode: str = "nearest"):
        super().__init__()
        self.res = ResidualBlock(in_channels)
        self.up_mode = up_mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.res(x)
        x = F.interpolate(
            x,
            scale_factor=2,
            mode=self.up_mode,
            align_corners=False if self.up_mode == "bilinear" else None,
        )
        x = self.conv(x)
        x = self.bn(x)
        return self.prelu(x)


class SemanticDecoder(nn.Module):
    """Four-scale decoder using the quantized shallower features as skip inputs."""

    def __init__(self, embedding_dims, out_channels: int, up_mode: str = "nearest"):
        super().__init__()
        self.embedding_dims = list(embedding_dims)
        self.num_scales = len(self.embedding_dims)

        deepest_channels = self.embedding_dims[-1]
        self.init = nn.Sequential(
            nn.Conv2d(deepest_channels, deepest_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        blocks = []
        in_channels = deepest_channels
        for scale in range(self.num_scales):
            if scale < self.num_scales - 1:
                out_channels_for_scale = self.embedding_dims[-2 - scale]
            else:
                out_channels_for_scale = self.embedding_dims[0]
            blocks.append(UpSampleBlock(in_channels, out_channels_for_scale, up_mode))
            in_channels = (
                out_channels_for_scale * 2
                if scale < self.num_scales - 1
                else out_channels_for_scale
            )
        self.up_blocks = nn.ModuleList(blocks)
        self.final = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, quantized_features):
        if len(quantized_features) != self.num_scales:
            raise ValueError(
                f"Expected {self.num_scales} feature scales, got {len(quantized_features)}"
            )

        x = self.init(quantized_features[-1])
        for scale, block in enumerate(self.up_blocks):
            x = block(x)
            if scale < self.num_scales - 1:
                skip = quantized_features[-2 - scale]
                x = torch.cat((x, skip), dim=1)
        return self.final(x)
