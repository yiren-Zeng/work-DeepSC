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
        self.bn   = nn.BatchNorm2d(out_ch)
        self.prelu= nn.PReLU()
    def forward(self, x):
        x = self.res(x)
        x = F.interpolate(x, scale_factor=2, mode=self.up_mode,
                          align_corners=False if self.up_mode=="bilinear" else None)  # 上采样×2
        x = self.conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        return x

class SemanticDecoder(nn.Module):
    def __init__(self, embedding_dims, out_channels, up_mode: str = "nearest"):
        """
        embedding_dims: 各尺度量化特征通道列表（如 [128,256,512,1024]，与编码器输出对应）
        out_channels:   输出图像通道数（RGB=3）
        up_mode:        上采样方式（"nearest" 或 "bilinear"）
        """
        super().__init__()
        self.embedding_dims = embedding_dims
        self.L = len(self.embedding_dims)
        self.up_mode = up_mode
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
                #（例如第一块输出 512，随后与 F̂3(512) concat → 下个块输入通道变 1024）
                out_ch = self.embedding_dims[-2 - i]  # 依次为 512, 256, 128 ...
            else:
                # 最后一块不再 concat，给一个合理的输出通道（这里沿用最浅层的通道数）
                out_ch = self.embedding_dims[0]       # 例如 128
            blocks.append(UpSampleBlock(in_ch, out_ch, up_mode=self.up_mode))
            # 若还有下一次，需要为“concat 后”的通道数做准备
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
            x = block(x)  # 上采样×2 + Conv+BN+PReLU
            # 前 L-1 个块后与对应浅层量化特征 concat
            if i < self.L - 1:
                skip = quant_feats[-2 - i]  # 依次取 F̂3, F̂2, F̂1
                # if x.shape[2:] != skip.shape[2:]:
                #     # 若因四舍五入差 1，强制对齐（通常不会发生）
                #     x = F.interpolate(x, size=skip.shape[2:], mode=self.up_mode)
                x = torch.cat([x, skip], dim=1)

        # 末端：转置卷积（不改尺寸）映射到 RGB，再 Sigmoid 到 [0,1]
        out = self.final(x) #此时通道数为 3
        return out