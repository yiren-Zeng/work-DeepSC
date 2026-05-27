import torch
import torch.nn as nn


class DeepSCLoss(nn.Module):
    """
    SRC + RAQ 双支路损失，无 repulsion_loss
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, x_hat_src, x_hat_trg, latent_loss_src, latent_loss_trg):
        recon_loss = self.criterion(x_hat_src, x) + self.criterion(x_hat_trg, x)

        latent_loss_src_sum = torch.stack(latent_loss_src).sum() if isinstance(latent_loss_src, (list, tuple)) else latent_loss_src
        latent_loss_trg_sum = torch.stack(latent_loss_trg).sum() if isinstance(latent_loss_trg, (list, tuple)) else latent_loss_trg
        latent_loss = latent_loss_src_sum + latent_loss_trg_sum

        return recon_loss, latent_loss
