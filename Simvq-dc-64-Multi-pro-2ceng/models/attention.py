import torch
import torch.nn as nn


def make_group_norm(channels, preferred_groups=32):
    groups = min(preferred_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class BottleneckSelfAttention(nn.Module):
    """Lightweight non-local attention for the deepest feature map."""

    def __init__(self, channels, num_groups=32):
        super().__init__()
        hidden = max(channels // 8, 32)
        self.norm = make_group_norm(channels, num_groups)
        self.query = nn.Conv2d(channels, hidden, kernel_size=1)
        self.key = nn.Conv2d(channels, hidden, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        q = self.query(x_norm).reshape(b, -1, h * w).transpose(1, 2)
        k = self.key(x_norm).reshape(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k) / (k.shape[1] ** 0.5), dim=-1)
        v = self.value(x_norm).reshape(b, c, h * w).transpose(1, 2)
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.gamma * self.proj(out)


class BottleneckAttentionStack(nn.Module):
    def __init__(self, channels, num_blocks=1, num_groups=32):
        super().__init__()
        self.blocks = nn.Sequential(*[
            BottleneckSelfAttention(channels, num_groups=num_groups)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(x)
