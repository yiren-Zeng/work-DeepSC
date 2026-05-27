import torch
import torch.nn as nn


class DeepSCLoss(nn.Module):
    """
    纯 SRC 单支路损失 + 多尺度VQ损失权重
    """
    def __init__(self, layer_weights=None):
        super().__init__()
        self.criterion = nn.MSELoss()
        # 各层VQ损失权重 [Layer0, Layer1, Layer2, Layer3]
        self.layer_weights = list(layer_weights or [1, 1, 1, 1])

    def set_layer_weights(self, weights):
        """动态设置各层VQ损失权重"""
        self.layer_weights = list(weights)

    def forward(self, x, x_hat, vq_losses):
        recon_loss = self.criterion(x_hat, x)
        # 按层权重加权VQ损失
        weighted_vq = sum(w * vl for w, vl in zip(self.layer_weights, vq_losses))
        return recon_loss, weighted_vq
