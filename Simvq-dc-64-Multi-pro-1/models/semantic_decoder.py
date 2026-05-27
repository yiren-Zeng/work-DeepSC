import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipConnectionDropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        batch_size = x.shape[0]
        mask = (torch.rand(batch_size, 1, 1, 1, device=x.device) > self.p).float()
        return x * mask


class ResidualBlock(nn.Module):
    """标准 Post-Activation 残差块: Conv1→BN→PReLU→Conv2→BN→add→PReLU"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.prelu2 = nn.PReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.prelu2(out)
        return out


class UpSampleBlock(nn.Module):
    """上采样块: res→bilinear(×2)→Conv→BN→PReLU"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.res = ResidualBlock(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.res(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x


class SemanticDecoder(nn.Module):
    """
    四层解码器: 4次上采样 + skip connection (Dropout 正则)
    输入 [F̂1,F̂2,F̂3,F̂4] → 输出 (B,3,256,256)
    """
    def __init__(self, embedding_dims, out_channels, skip_dropout_p=None):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.L = len(self.embedding_dims)
        num_skip = self.L - 1  # 3

        # Skip Dropout
        if skip_dropout_p and len(skip_dropout_p) >= num_skip:
            self.skip_dropouts = nn.ModuleList([
                SkipConnectionDropout(p=skip_dropout_p[num_skip - 1 - i]) for i in range(num_skip)
            ])
        else:
            self.skip_dropouts = nn.ModuleList([
                SkipConnectionDropout(p=0.0) for _ in range(num_skip)
            ])

        deepest_c = self.embedding_dims[-1]

        self.init = nn.Sequential(
            nn.Conv2d(deepest_c, deepest_c, 3, 1, 1),
            nn.PReLU(),
        )

        blocks = []
        in_ch = deepest_c
        for i in range(self.L):
            if i < self.L - 1:
                out_ch = self.embedding_dims[-2 - i]
            else:
                out_ch = self.embedding_dims[0]
            blocks.append(UpSampleBlock(in_ch, out_ch))
            if i < self.L - 1:
                in_ch = out_ch * 2
            else:
                in_ch = out_ch
        self.up_blocks = nn.ModuleList(blocks)

        # 输出: Conv2d + Tanh (数据归一化到[-1,1])
        self.final = nn.Sequential(
            nn.Conv2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def set_skip_dropout_p(self, p_list):
        if p_list is None or len(p_list) < self.L - 1:
            return
        num_skip = self.L - 1
        for i in range(num_skip):
            self.skip_dropouts[i].p = p_list[num_skip - 1 - i]

    def forward(self, quant_feats):
        assert len(quant_feats) == self.L, f"尺度数不匹配：{len(quant_feats)} vs L={self.L}"

        x = self.init(quant_feats[-1])

        for i, block in enumerate(self.up_blocks):
            x = block(x)
            if i < self.L - 1:
                skip = quant_feats[-2 - i]
                skip = self.skip_dropouts[i](skip)         # 跳跃连接 Dropout
                x = torch.cat([x, skip], dim=1)

        out = self.final(x)
        return out
