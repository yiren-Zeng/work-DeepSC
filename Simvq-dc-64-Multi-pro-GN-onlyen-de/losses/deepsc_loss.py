import torch
import torch.nn as nn


class DeepSCLoss(nn.Module):
    """
    纯重建损失（无VQ损失）
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, x_hat):
        recon_loss = self.criterion(x_hat, x)
        return recon_loss
