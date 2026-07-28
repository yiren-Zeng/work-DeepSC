import torch
import torch.nn as nn


class DeepSCLoss(nn.Module):
    """
        兼容单支路（原 VQ）与 RAQ 双支路，并加入 RAQ 码本分布对齐损失 (Alignment Loss)
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, x, x_hat_src, x_hat_trg, latent_loss_src, latent_loss_trg, W_src_list=None, W_trg_list=None):

        # 1. 计算重建损失
        recon_loss = self.criterion(x_hat_src, x) + self.criterion(x_hat_trg, x)

        # 2. 计算量化损失
        latent_loss_src_sum = torch.stack(latent_loss_src).sum() if isinstance(latent_loss_src,
                                                                               (list, tuple)) else latent_loss_src
        latent_loss_trg_sum = torch.stack(latent_loss_trg).sum() if isinstance(latent_loss_trg,
                                                                               (list, tuple)) else latent_loss_trg
        latent_loss = latent_loss_src_sum + latent_loss_trg_sum

        # 3. 【核心新增】：计算分布对齐损失 (Alignment Loss)
        align_loss = 0.0
        # 判断是否传入了码本列表（训练的时候传入了）
        if W_src_list is not None and W_trg_list is not None:
            for w_src, w_trg in zip(W_src_list, W_trg_list):
                # w_src: [K_src, D], w_trg: [K_trg, D]
                # 分别对第 0 维度（K 的维度）求均值和标准差，使其形状变成 [D]
                src_mean = w_src.detach().mean(dim=0)
                trg_mean = w_trg.mean(dim=0)

                src_std = w_src.detach().std(dim=0)
                trg_std = w_trg.std(dim=0)

                # 用 MSE 强迫生成的 W_trg 分布贴近 W_src
                align_loss += self.criterion(trg_mean, src_mean)
                align_loss += self.criterion(trg_std, src_std)

        # 【修改】：将权重从 1.0 降低到 0.05，给目标码本留下足够的“扩张与细化”空间
        total_latent_loss = latent_loss + 0.1 * align_loss

        return recon_loss, total_latent_loss