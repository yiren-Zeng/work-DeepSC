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
        兼容单支路（原 VQ）与 RAQ 双支路，并加入 RAQ 码本分布对齐损失 (Alignment Loss)
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

        # 3. 计算分布排斥损失
        align_loss = 0.0
        # 只对生成的动态码本RAQ施加排斥，强迫它的码本展开
        if W_trg_list is not None:
            for w_trg in W_trg_list:
                # w_trg: [K_trg, D]
                # 计算目标码本向量两两之间的余弦相似度
                norm_w = F.normalize(w_trg, p=2, dim=1)
                sim_matrix = torch.matmul(norm_w, norm_w.t())  # [K_trg, K_trg]

                # 我们希望非对角线元素（不同码字间的相似度）尽可能趋近于 0（正交）或负数（排斥）
                # 减去对角线（自己和自己的相似度永远是 1）
                eye = torch.eye(sim_matrix.size(0), device=sim_matrix.device)
                off_diagonal = sim_matrix * (1 - eye)

                # 惩罚过高的相似度（使用 L2 惩罚）
                align_loss += torch.mean(off_diagonal ** 2)

        # 【修改】：将权重从 1.0 降低到 0.05，给目标码本留下足够的“扩张与细化”空间
        total_latent_loss = latent_loss + 0.05 * align_loss

        return recon_loss, total_latent_loss, latent_loss_src_sum, latent_loss_trg_sum
