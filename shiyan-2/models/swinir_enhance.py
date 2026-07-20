"""
轻量级 SwinIR 风格的后处理质量增强网络。

基于 SwinIR (Liang et al., ICCV 2021) 的 Residual Swin Transformer Block (RSTB)
设计，但适配为轻量级后处理模块，用于提升解码器输出的重建质量。

参考:
- SwinIR: Image Restoration Using Swin Transformer, arXiv:2108.10257
- Swin Transformer V2: Scaling Up Capacity and Resolution, arXiv:2111.09883
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def window_partition(x, window_size):
    """将特征图划分为不重叠的窗口。"""
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windows


def window_reverse(windows, window_size, H, W):
    """将窗口拼接回特征图。"""
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, -1, window_size, window_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, -1, H, W)
    return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """基于窗口的多头自注意力 (W-MSA)，支持相对位置偏置。"""
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        nn.init.trunc_normal_(self.qkv.weight, std=.02)
        nn.init.trunc_normal_(self.proj.weight, std=.02)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block，支持 W-MSA 和 SW-MSA。"""
    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        B, C, H, W = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()  # B H W C
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # B C H W

        # 循环移位
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # 窗口划分
        x_windows = window_partition(x, self.window_size)  # nW*B, C, ws, ws
        x_windows = x_windows.view(-1, C, self.window_size * self.window_size)
        x_windows = x_windows.permute(0, 2, 1).contiguous()  # nW*B, ws*ws, C

        # W-MSA / SW-MSA
        attn_windows = self.attn(x_windows)

        # 窗口合并
        attn_windows = attn_windows.permute(0, 2, 1).contiguous()
        attn_windows = attn_windows.view(-1, C, self.window_size, self.window_size)
        x = window_reverse(attn_windows, self.window_size, H, W)

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        x = shortcut + x

        # FFN
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm2(x)
        x = self.mlp(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = shortcut + x

        return x


class ResidualSwinTransformerBlock(nn.Module):
    """Residual Swin Transformer Block (RSTB): 多个 Swin Transformer Block + 残差连接。"""
    def __init__(self, dim, num_heads=4, window_size=8, num_stb=2, mlp_ratio=2.):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(num_stb):
            shift_size = 0 if (i % 2 == 0) else window_size // 2
            self.blocks.append(SwinTransformerBlock(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=shift_size, mlp_ratio=mlp_ratio))
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        shortcut = x
        for blk in self.blocks:
            x = blk(x)
        x = self.conv(x) + shortcut
        return x


class SwinIREnhance(nn.Module):
    """
    轻量级 SwinIR 风格的质量增强后处理网络。
    输入: 解码器重建图像 [B, 3, H, W] (值域约 [-1, 1])
    输出: 增强后图像 [B, 3, H, W]
    """
    def __init__(self, embed_dim=48, num_rstb=4, window_size=8, num_heads=4):
        super().__init__()
        self.shallow_feat = nn.Conv2d(3, embed_dim, 3, 1, 1)
        self.rstb_blocks = nn.ModuleList([
            ResidualSwinTransformerBlock(
                dim=embed_dim, num_heads=num_heads,
                window_size=window_size, num_stb=2, mlp_ratio=2.)
            for _ in range(num_rstb)
        ])
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.reconstruction = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(embed_dim, 3, 3, 1, 1),
        )

    def forward(self, x):
        shortcut = x
        feat = self.shallow_feat(x)
        body = feat
        for block in self.rstb_blocks:
            body = block(body)
        body = self.conv_after_body(body)
        feat = feat + body
        out = self.reconstruction(feat)
        return out + shortcut  # 全局残差连接，学习残差
