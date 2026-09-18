import torch
import torch.nn as nn



def make_group_norm(channels, preferred_groups=32):
    groups = min(preferred_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


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