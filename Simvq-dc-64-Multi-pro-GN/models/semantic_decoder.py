import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipConnectionDropout(nn.Module):
    def __init__(self, p=0.3):
        """
        跳跃连接 Dropout
        :param p: 丢弃概率，推荐设置在 0.2 到 0.5 之间
        """
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        batch_size = x.shape[0]
        # 生成 (B,1,1,1) 二值掩码：以概率 p 整个样本的 skip 被丢弃
        mask = (torch.rand(batch_size, 1, 1, 1, device=x.device) > self.p).float()
        return x * mask


class ResidualBlock(nn.Module):
    def __init__(self, channels):
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


class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode: str = "nearest"):
        super().__init__()
        self.res = ResidualBlock(in_ch)
        self.up_mode = up_mode
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.gn = nn.GroupNorm(num_groups=32, num_channels=out_ch)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.res(x)
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode,
                          align_corners=False if self.up_mode == "bilinear" else None)
        x = self.conv(x)
        x = self.gn(x)
        x = self.prelu(x)
        return x


class SemanticDecoder(nn.Module):
    """
    四层解码器: 4次上采样 + skip connection
    输入 [F̂1,F̂2,F̂3,F̂4] → 输出 (B,3,256,256)
    """
    def __init__(self, embedding_dims, out_channels, up_mode: str = "nearest", skip_dropout_p=None):
        """
        embedding_dims: 各尺度量化特征通道列表（如 [128,256,512,1024]，与编码器输出对应）
        out_channels:   输出图像通道数（RGB=3）
        up_mode:        上采样方式（"nearest" 或 "bilinear"）
        skip_dropout_p: 各层跳跃连接 Dropout 概率列表 [Layer0, Layer1, Layer2]（None 或空列表表示不使用）
        """
        super().__init__()
        self.embedding_dims = embedding_dims
        self.L = len(self.embedding_dims)
        self.up_mode = up_mode
        # 每层一个独立的 SkipConnectionDropout（前 L-1 层有 skip，最后一层无 skip）
        num_skip = self.L - 1  # 3
        if skip_dropout_p and len(skip_dropout_p) >= num_skip:
            # skip_dropout_p = [Layer0_p, Layer1_p, Layer2_p]
            # 循环中 i=0 → Layer2, i=1 → Layer1, i=2 → Layer0，所以反序
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
        for i in range(self.L):  # i=0..L-1；总计 L 个
            if i < self.L - 1:
                # 第 i 个块输出通道对齐到下一次要 concat 的浅层通道数
                out_ch = self.embedding_dims[-2 - i]  # 依次为 512, 256, 128 ...
            else:
                # 最后一块不再 concat，给一个合理的输出通道
                out_ch = self.embedding_dims[0]       # 例如 128
            blocks.append(UpSampleBlock(in_ch, out_ch, up_mode=self.up_mode))
            # 若还有下一次，需要为"concat 后"的通道数做准备
            if i < self.L - 1:
                in_ch = out_ch * 2  # 与下一次要拼接的同尺度特征通道相加
            else:
                in_ch = out_ch      # 最后一块后不再 concat
        self.up_blocks = nn.ModuleList(blocks)

        # 以转置卷积结束；此处只做通道映射，不改尺寸（stride=1）
        self.final = nn.ConvTranspose2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, quant_feats):
        """
        quant_feats: list[Tensor]，从浅到深排列：
            [F̂1(B,128,H), F̂2(B,256,H/2), F̂3(B,512,H/4), F̂4(B,1024,H/8)]
        返回：Î，形状 (B, out_channels, H, W)（假设 H=W=256）
        """
        assert len(quant_feats) == self.L, f"尺度数不匹配：{len(quant_feats)} vs L={self.L}"

        # 起点：最深层
        x = self.init(quant_feats[-1])

        # 逐块上采样并按需 concat
        for i, block in enumerate(self.up_blocks):
            x = block(x)  # 上采样×2 + Conv+GN+PReLU
            # 前 L-1 个块后与对应浅层量化特征 concat
            if i < self.L - 1:
                skip = quant_feats[-2 - i]  # 依次取 F̂3, F̂2, F̂1
                skip = self.skip_dropouts[i](skip)  # 各层独立 Dropout
                x = torch.cat([x, skip], dim=1)

        # 末端：转置卷积（不改尺寸）映射到 RGB
        out = self.final(x)
        return out
