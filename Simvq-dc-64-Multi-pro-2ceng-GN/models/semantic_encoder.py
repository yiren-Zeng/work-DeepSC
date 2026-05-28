import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.gn(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = out + identity
        return out


class DownSampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2):
        """
        下采样块
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param stride: 下采样步幅（2=2倍下采样，4=4倍下采样）
        """
        super().__init__()
        self.res1 = ResidualBlock(in_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.res2 = ResidualBlock(out_ch)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=out_ch)
        self.act = nn.PReLU()
        self.tail = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.gn(x)
        x = self.act(x)
        x = self.tail(x)
        return x


class SemanticEncoder(nn.Module):
    """
    二层编码器: 2次下采样
    第一层4倍下采样，第二层2倍下采样
    输入 (B,3,256,256) → 输出 [(B,128,64,64), (B,256,32,32)]
    """
    def __init__(self, in_channels, num_downsample_blocks, base_channels, strides=None):
        """
        :param strides: 各层下采样步幅列表，如 [4, 2]，默认全为2
        """
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
        )
        if strides is None:
            strides = [2] * num_downsample_blocks
        assert len(strides) == num_downsample_blocks, \
            f"strides长度({len(strides)})必须等于num_downsample_blocks({num_downsample_blocks})"

        blocks = []
        ch = base_channels
        for i in range(num_downsample_blocks):
            blocks.append(DownSampleBlock(ch, ch * 2, stride=strides[i]))
            ch *= 2
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feats = []
        x = self.init(x)
        for b in self.blocks:
            x = b(x)
            feats.append(x)
        return feats
