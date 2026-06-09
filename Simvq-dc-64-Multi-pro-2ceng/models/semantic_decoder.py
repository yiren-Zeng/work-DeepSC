import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.utils.checkpoint import checkpoint

from .semantic_encoder import make_activation, make_norm


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
    def __init__(self, channels, norm_type="batch", num_groups=32, activation="prelu"):
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


class UpSampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_mode: str = "nearest", scale_factor: int = 2,
                 norm_type="batch", num_groups=32, activation="prelu",
                 num_res_blocks=1):
        """
        上采样块
        :param in_ch: 输入通道数
        :param out_ch: 输出通道数
        :param up_mode: 上采样方式
        :param scale_factor: 上采样倍率（2=2倍上采样，4=4倍上采样）
        """
        super().__init__()
        self.res = nn.Sequential(*[
            ResidualBlock(in_ch, norm_type, num_groups, activation)
            for _ in range(num_res_blocks)
        ])
        self.up_mode = up_mode
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = make_norm(out_ch, norm_type, num_groups)
        self.act = make_activation(activation)

    def forward(self, x):
        x = self.res(x)
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.up_mode,
                          align_corners=False if self.up_mode == "bilinear" else None)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class SemanticDecoder(nn.Module):
    """
    可配置多层解码器。

    上采样倍率与编码器 strides 反序对称；前 L-1 个上采样块使用 skip connection。
    """
    def __init__(self, embedding_dims, out_channels, up_mode: str = "nearest",
                 skip_dropout_p=None, upsample_scales=None,
                 norm_type="batch", num_groups=32, activation="prelu",
                 num_res_blocks=1):
        """
        embedding_dims: 各尺度量化特征通道列表（如 [128,256]，与编码器输出对应）
        out_channels:   输出图像通道数（RGB=3）
        up_mode:        上采样方式（"nearest" 或 "bilinear"）
        skip_dropout_p: 各层跳跃连接 Dropout 概率列表 [Layer0]（None 或空列表表示不使用）
        upsample_scales: 各层上采样倍率列表（如 [2, 4]），默认全为2
        """
        super().__init__()
        self.embedding_dims = embedding_dims
        self.L = len(self.embedding_dims)
        self.up_mode = up_mode

        if upsample_scales is None:
            upsample_scales = [2] * self.L
        assert len(upsample_scales) == self.L, \
            f"upsample_scales长度({len(upsample_scales)})必须等于L({self.L})"
        self.upsample_scales = upsample_scales

        # 每层一个独立的 SkipConnectionDropout（前 L-1 层有 skip，最后一层无 skip）
        num_skip = self.L - 1
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
            make_activation(activation),
        )

        blocks = []
        in_ch = deepest_c
        for i in range(self.L):  # i=0..L-1；总计 L 个
            if i < self.L - 1:
                # 第 i 个块输出通道对齐到下一次要 concat 的浅层通道数
                out_ch = self.embedding_dims[-2 - i]  # 依次为 128 ...
            else:
                # 最后一块不再 concat，给一个合理的输出通道
                out_ch = self.embedding_dims[0]       # 例如 128
            blocks.append(UpSampleBlock(
                in_ch, out_ch, up_mode=self.up_mode,
                scale_factor=self.upsample_scales[i],
                norm_type=norm_type,
                num_groups=num_groups,
                activation=activation,
                num_res_blocks=num_res_blocks,
            ))
            # 若还有下一次，需要为"concat 后"的通道数做准备
            if i < self.L - 1:
                in_ch = out_ch * 2  # 与下一次要拼接的同尺度特征通道相加
            else:
                in_ch = out_ch      # 最后一块后不再 concat
        self.up_blocks = nn.ModuleList(blocks)

        # 以转置卷积结束；此处只做通道映射，不改尺寸（stride=1）
        self.final = nn.ConvTranspose2d(in_ch, out_channels, kernel_size=3, stride=1, padding=1)
        self.tail_device = None
        self.tail_blocks = 0

    def set_tail_device(self, device, tail_blocks=1):
        self.tail_device = torch.device(device)
        self.tail_blocks = max(0, min(int(tail_blocks), len(self.up_blocks)))
        if self.tail_blocks > 0:
            for block in self.up_blocks[-self.tail_blocks:]:
                block.to(self.tail_device)
            self.final.to(self.tail_device)
        return self

    def set_skip_dropout_p(self, p_list):
        """
        动态设置各层跳跃连接 Dropout 概率
        p_list: 从浅层到深层排列，长度为 L-1
        """
        if p_list is None or len(p_list) < self.L - 1:
            return
        num_skip = self.L - 1
        # skip_dropouts[i] 对应 Layer(num_skip-1-i)，所以反序赋值
        for i in range(num_skip):
            self.skip_dropouts[i].p = p_list[num_skip - 1 - i]

    def forward(self, quant_feats):
        """
        quant_feats: list[Tensor]，从浅到深排列：
            [F̂1, F̂2, ... F̂L]
        返回：Î，形状 (B, out_channels, H, W)
        """
        assert len(quant_feats) == self.L, f"尺度数不匹配：{len(quant_feats)} vs L={self.L}"

        # 起点：最深层
        x = self.init(quant_feats[-1])
        use_checkpoint = self.training and os.environ.get("SIMVQ_GRADIENT_CHECKPOINTING", "0") == "1"

        # 逐块上采样并按需 concat
        for i, block in enumerate(self.up_blocks):
            block_device = next(block.parameters()).device
            if x.device != block_device:
                x = x.to(block_device, non_blocking=True)
            if use_checkpoint:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)  # 上采样 + Conv+BN+PReLU
            # 前 L-1 个块后与对应浅层量化特征 concat
            if i < self.L - 1:
                skip = quant_feats[-2 - i]  # 依次取 F̂1
                if skip.device != x.device:
                    skip = skip.to(x.device, non_blocking=True)
                skip = self.skip_dropouts[i](skip)  # 各层独立 Dropout
                x = torch.cat([x, skip], dim=1)

        # 末端：转置卷积（不改尺寸）映射到 RGB
        final_device = self.final.weight.device
        if x.device != final_device:
            x = x.to(final_device, non_blocking=True)
        out = self.final(x)
        return out
