# import torch
# import torch.nn as nn
#
# class DeepSCLoss(nn.Module):
#     """
#         兼容单支路（原 VQ）与 RAQ 双支路：
#           - L_rec = L1(x_hat, x)
#           - L_vq  = sum(vq_losses)
#     """
#     def __init__(self):
#         super().__init__()
#         self.criterion = nn.MSELoss()
#
#     def forward(self, x, x_hat_src, x_hat_trg, latent_loss_src, latent_loss_trg):
#         """
#         x:               原图像 (B,3,H,W)
#         x_hat_src:       使用原始码本重建的图像 (B,3,H,W)
#         x_hat_trg:       使用自适应码本重建的图像 (B,3,H,W)
#         latent_loss_src: 原始码本的量化损失
#         latent_loss_trg: 自适应码本的量化损失
#         """
#         # 计算重建损失（源码头本和目标码本的重建损失之和
#         recon_loss = self.criterion(x_hat_src, x) + self.criterion(x_hat_trg, x)
#
#         # 潜在损失（即量化损失，包括源码头本和目标码本）
#         latent_loss_src_sum = torch.stack(latent_loss_src).sum() if isinstance(latent_loss_src, (list, tuple)) else latent_loss_src # 将列表的值都加起来成一个
#         latent_loss_trg_sum = torch.stack(latent_loss_trg).sum() if isinstance(latent_loss_trg, (list, tuple)) else latent_loss_trg # 将列表的值都加起来成一个
#         latent_loss = latent_loss_src_sum + latent_loss_trg_sum
#
#         return recon_loss, latent_loss
#
#


import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSCLoss(nn.Module):
    """
        兼容单支路（原 VQ）与 RAQ 双支路，并加入 RAQ 码本排斥损失 (Min-L2 Repulsion Loss)
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, x_hat_src, x_hat_trg, latent_loss_src, latent_loss_trg, W_trg_list=None):

        # 1. 计算重建损失
        recon_loss = self.criterion(x_hat_src, x) + self.criterion(x_hat_trg, x)

        # 2. 计算量化损失
        latent_loss_src_sum = torch.stack(latent_loss_src).sum() if isinstance(latent_loss_src, (list, tuple)) else latent_loss_src
        latent_loss_trg_sum = torch.stack(latent_loss_trg).sum() if isinstance(latent_loss_trg, (list, tuple)) else latent_loss_trg

        latent_loss = latent_loss_src_sum + latent_loss_trg_sum

        # 3. 计算最小L2距离排斥损失
        # 目标：最大化码字间最小L2距离 → 等价于最小化 -min_dist
        # 这比余弦正交更贴合 VQ 的 L2 量化机制，且不强迫码字正交，只要求它们分开
        repulsion_loss = 0.0
        if W_trg_list is not None:
            for w_trg in W_trg_list:
                # w_trg: [K_trg, D]
                # 计算所有码字对之间的 L2 距离平方矩阵
                norm_sq = torch.sum(w_trg ** 2, dim=1)  # [K_trg]
                dist_sq = norm_sq.unsqueeze(1) + norm_sq.unsqueeze(0) - 2 * torch.matmul(w_trg, w_trg.t())  # [K_trg, K_trg]
                # 去掉对角线（自己和自己距离为0）
                K = dist_sq.size(0)
                eye = torch.eye(K, device=dist_sq.device).bool()
                dist_sq = dist_sq.masked_fill(eye, float('inf'))
                # 取最小距离，取负号使其成为损失（最小化负距离 = 最大化距离）
                min_dist_sq = dist_sq.min()
                repulsion_loss += (-min_dist_sq)

        total_latent_loss = latent_loss + 0.05 * repulsion_loss

        return recon_loss, total_latent_loss, latent_loss_src_sum, latent_loss_trg_sum
