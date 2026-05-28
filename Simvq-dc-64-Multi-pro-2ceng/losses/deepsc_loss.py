import torch
import torch.nn as nn
import torch.nn.functional as F


def _gaussian_window(window_size, sigma, channels, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = torch.outer(g, g)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()


def _ssim_components(x, y, window_size=11, sigma=1.5, data_range=1.0):
    channels = x.shape[1]
    window = _gaussian_window(window_size, sigma, channels, x.device, x.dtype)
    padding = window_size // 2
    mu_x = F.conv2d(x, window, padding=padding, groups=channels)
    mu_y = F.conv2d(y, window, padding=padding, groups=channels)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, window, padding=padding, groups=channels) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, window, padding=padding, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channels) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    cs = (2 * sigma_xy + c2) / (sigma_x_sq + sigma_y_sq + c2)
    ssim = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim.clamp(0.0, 1.0).mean(), cs.clamp(0.0, 1.0).mean()


def ms_ssim_loss(x_hat, x, levels=5):
    x_hat = ((x_hat + 1.0) / 2.0).clamp(0.0, 1.0)
    x = ((x + 1.0) / 2.0).clamp(0.0, 1.0)
    weights = x.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    weights = weights[:levels]
    weights = weights / weights.sum()

    mcs = []
    ssim_val = None
    for level in range(levels):
        ssim_val, cs = _ssim_components(x_hat, x)
        if level < levels - 1:
            mcs.append(cs)
            x_hat = F.avg_pool2d(x_hat, kernel_size=2, stride=2)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            if min(x.shape[-2:]) < 11:
                weights = weights[:level + 2]
                weights = weights / weights.sum()
                break

    if not mcs:
        ms_ssim = ssim_val
    else:
        cs_stack = torch.stack(mcs)
        ms_ssim = torch.prod(cs_stack ** weights[:len(mcs)]) * (ssim_val ** weights[len(mcs)])
    return 1.0 - ms_ssim.clamp(0.0, 1.0)


class DeepSCLoss(nn.Module):
    """
    纯 SRC 单支路损失 + 多尺度 VQ 损失权重。
    """
    def __init__(self, layer_weights=None, mse_weight=1.0, ms_ssim_weight=0.0):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.layer_weights = list(layer_weights or [1, 1])
        self.mse_weight = mse_weight
        self.ms_ssim_weight = ms_ssim_weight

    def set_layer_weights(self, weights):
        """动态设置各层VQ损失权重"""
        self.layer_weights = list(weights)

    def forward(self, x, x_hat, vq_losses):
        mse = self.criterion(x_hat, x)
        if self.ms_ssim_weight > 0:
            perceptual = ms_ssim_loss(x_hat, x)
            recon_loss = self.mse_weight * mse + self.ms_ssim_weight * perceptual
        else:
            perceptual = x.new_tensor(0.0)
            recon_loss = mse
        # 按层权重加权VQ损失
        weighted_vq = sum(w * vl for w, vl in zip(self.layer_weights, vq_losses))
        return recon_loss, weighted_vq
