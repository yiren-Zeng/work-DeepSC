import torch
import torch.nn as nn


class DeepSCLoss(nn.Module):
    """单支路 VQ：重建 MSE + 各层 VQ 损失之和。"""

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, x_hat, vq_losses):
        recon_loss = self.criterion(x_hat, x)
        vq_sum = (
            torch.stack(vq_losses).sum()
            if isinstance(vq_losses, (list, tuple))
            else vq_losses
        )
        total = recon_loss + vq_sum
        return recon_loss, total, vq_sum
