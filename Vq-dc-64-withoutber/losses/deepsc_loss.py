import torch
import torch.nn as nn


class DeepSCLoss(nn.Module):
    """
    纯 SRC 单支路损失
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, x_hat, vq_losses):
        recon_loss = self.criterion(x_hat, x)
        latent_loss = torch.stack(vq_losses).sum() if isinstance(vq_losses, (list, tuple)) else vq_losses
        return recon_loss, latent_loss