import torch
import torch.nn as nn

class DeepSCLoss(nn.Module):
    """
        兼容单支路（原 VQ）与 RAQ 双支路：
          - L_rec = L1(x_hat, x)
          - L_vq  = sum(vq_losses)
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, x_hat_src, x_hat_trg, latent_loss_src, latent_loss_trg):
        """
        x:               原图像 (B,3,H,W)
        x_hat_src:       使用原始码本重建的图像 (B,3,H,W)
        x_hat_trg:       使用自适应码本重建的图像 (B,3,H,W)
        latent_loss_src: 原始码本的量化损失
        latent_loss_trg: 自适应码本的量化损失
        """
        # 计算重建损失（源码头本和目标码本的重建损失之和
        recon_loss = self.criterion(x_hat_src, x) + self.criterion(x_hat_trg, x)

        # 潜在损失（即量化损失，包括源码头本和目标码本）
        latent_loss_src_sum = torch.stack(latent_loss_src).sum() if isinstance(latent_loss_src, (list, tuple)) else latent_loss_src # 将列表的值都加起来成一个
        latent_loss_trg_sum = torch.stack(latent_loss_trg).sum() if isinstance(latent_loss_trg, (list, tuple)) else latent_loss_trg # 将列表的值都加起来成一个
        latent_loss = latent_loss_src_sum + latent_loss_trg_sum

        return recon_loss, latent_loss


