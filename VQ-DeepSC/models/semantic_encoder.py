import torch.nn as nn


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


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(in_channels)
        self.down = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU()
        self.tail = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.bn(x)
        x = self.act(x)
        return self.tail(x)


class SemanticEncoder(nn.Module):
    """Four-scale encoder with one 2x downsampling operation per scale."""

    def __init__(self, in_channels: int, num_downsample_blocks: int, base_channels: int):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.PReLU(),
        )

        blocks = []
        channels = base_channels
        for _ in range(num_downsample_blocks):
            blocks.append(DownSampleBlock(channels, channels * 2))
            channels *= 2
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = []
        x = self.init(x)
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features
