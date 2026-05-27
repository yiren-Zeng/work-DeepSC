import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
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


class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode: str = "nearest"):
        super().__init__()
        self.res = ResidualBlock(in_ch)
        self.up_mode = up_mode
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.res(x)
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode,
                          align_corners=False if self.up_mode == "bilinear" else None)
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class SemanticDecoder(nn.Module):
    """
    单层解码器: 1次上采样
    输入 [F̂1(B,128,128,128)] → 输出 (B,3,256,256)
    """
    def __init__(self, embedding_dims, out_channels, up_mode: str = "nearest"):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.L = len(self.embedding_dims)
        self.up_mode = up_mode
        deepest_c = self.embedding_dims[-1]

        self.init = nn.Sequential(
            nn.Conv2d(deepest_c, deepest_c, 3, 1, 1),
            nn.PReLU(),
        )

        # 单层: 1个上采样块，无 skip connection
        in_ch = deepest_c
        out_ch = self.embedding_dims[0]
        self.up_blocks = nn.ModuleList([UpSampleBlock(in_ch, out_ch, up_mode=self.up_mode)])

        self.final = nn.ConvTranspose2d(out_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, quant_feats):
        assert len(quant_feats) == self.L, f"尺度数不匹配：{len(quant_feats)} vs L={self.L}"

        x = self.init(quant_feats[-1])

        for i, block in enumerate(self.up_blocks):
            x = block(x)
            # 单层无 skip connection

        out = self.final(x)
        return out
