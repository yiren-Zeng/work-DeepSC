import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.checkpoint import checkpoint

from .attention import make_group_norm


def make_norm(channels, norm_type="batch", num_groups=32):
    if norm_type == "group":
        return make_group_norm(channels, num_groups)
    return nn.BatchNorm2d(channels)


def make_activation(name="prelu"):
    if name == "silu":
        return nn.SiLU(inplace=True)
    return nn.PReLU()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, norm_type="batch", num_groups=32, activation="prelu"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.norm = make_norm(channels, norm_type, num_groups)
        self.act = make_activation(activation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.conv2(out)
        out = out + identity
        return out


class DownSampleBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2,
                 norm_type="batch", num_groups=32, activation="prelu",
                 num_res_blocks=1, use_cascade_downsample=True):
        """
        下采样块
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param stride: 下采样步幅（2=2倍下采样，4=4倍下采样）
        """
        super().__init__()
        self.res1 = nn.Sequential(*[
            ResidualBlock(in_ch, norm_type, num_groups, activation)
            for _ in range(num_res_blocks)
        ])
        if use_cascade_downsample:
            self.down = self._make_cascade_downsample(
                in_ch, out_ch, stride, norm_type, num_groups, activation
            )
        else:
            self.down = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.res2 = nn.Sequential(*[
            ResidualBlock(out_ch, norm_type, num_groups, activation)
            for _ in range(num_res_blocks)
        ])
        self.norm = make_norm(out_ch, norm_type, num_groups)
        self.act = make_activation(activation)
        self.tail = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

    @staticmethod
    def _make_cascade_downsample(in_ch, out_ch, stride, norm_type, num_groups, activation):
        if stride in (4, 8):
            layers = []
            current_ch = in_ch
            remaining = stride
            while remaining > 1:
                next_ch = out_ch if remaining <= 2 else max(out_ch // 2, 32)
                layers.extend([
                    nn.Conv2d(current_ch, next_ch, kernel_size=3, stride=2, padding=1),
                    make_norm(next_ch, norm_type, num_groups),
                    make_activation(activation),
                ])
                current_ch = next_ch
                remaining //= 2
            if current_ch != out_ch:
                layers.append(nn.Conv2d(current_ch, out_ch, kernel_size=1))
            return nn.Sequential(*layers)
        return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

    def forward(self, x):
        x = self.res1(x)
        x = self.down(x)
        x = self.res2(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.tail(x)
        return x


class SemanticEncoder(nn.Module):
    """
    可配置多层编码器。

    层数由 num_downsample_blocks 决定；每层输出通道按 base_channels * 2**层号递增。
    strides 控制每层下采样倍率，例如 [4, 2] 表示 256 -> 64 -> 32。
    """
    def __init__(self, in_channels, num_downsample_blocks, base_channels, strides=None,
                 norm_type="batch", num_groups=32, activation="prelu",
                 num_res_blocks=1, use_cascade_downsample=True):
        """
        :param strides: 各层下采样步幅列表，如 [4, 2]，默认全为2
        """
        super().__init__()
        self.init = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            make_activation(activation),
        )
        if strides is None:
            strides = [2] * num_downsample_blocks
        assert len(strides) == num_downsample_blocks, \
            f"strides长度({len(strides)})必须等于num_downsample_blocks({num_downsample_blocks})"

        blocks = []
        ch = base_channels
        for i in range(num_downsample_blocks):
            blocks.append(DownSampleBlock(
                ch, ch * 2, stride=strides[i],
                norm_type=norm_type,
                num_groups=num_groups,
                activation=activation,
                num_res_blocks=num_res_blocks,
                use_cascade_downsample=use_cascade_downsample,
            ))
            ch *= 2
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feats = []
        x = self.init(x)
        use_checkpoint = self.training and os.environ.get("SIMVQ_GRADIENT_CHECKPOINTING", "0") == "1"
        for b in self.blocks:
            if use_checkpoint:
                x = checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
            feats.append(x)
        return feats
