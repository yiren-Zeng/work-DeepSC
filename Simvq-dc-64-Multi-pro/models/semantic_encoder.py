import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = out + identity
        return out


class DownSampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.res2 = ResidualBlock(out_ch)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()
        self.tail = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.tail(x)
        return x


class SemanticEncoder(nn.Module):
    """
    四层编码器: 4次下采样
    输入 (B,3,256,256) → 输出 [(B,128,128,128), (B,256,64,64), (B,512,32,32), (B,1024,16,16)]
    """
    def __init__(self, in_channels, num_downsample_blocks, base_channels):
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        blocks = []
        ch = base_channels
        for _ in range(num_downsample_blocks):
            blocks.append(DownSampleBlock(ch, ch * 2))
            ch *= 2
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feats = []
        x = self.init(x)
        for b in self.blocks:
            x = b(x)
            feats.append(x)
        return feats
