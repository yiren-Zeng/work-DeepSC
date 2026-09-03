"""Loss used by the dedicated independent RAQ-RVQ training scheme."""

import torch
import torch.nn as nn


class DeepSCLoss(nn.Module):
    """Joint source/independent-RAQ reconstruction and VQ loss."""

    def __init__(self, layer_weights=None, mse_weight=1.0):
        super().__init__()
        self.criterion = nn.MSELoss()
        self.layer_weights = list(layer_weights or [1, 1])
        self.mse_weight = float(mse_weight)

    def set_layer_weights(self, weights):
        self.layer_weights = list(weights)

    def _reconstruction_loss(self, x, x_hat):
        if x.device != x_hat.device:
            x = x.to(x_hat.device, non_blocking=True)
        return self.mse_weight * self.criterion(x_hat, x)

    def _weighted_vq_loss(self, vq_losses, device):
        return sum(
            weight * loss.to(device, non_blocking=True)
            for weight, loss in zip(self.layer_weights, vq_losses)
        )

    def forward(
        self,
        x,
        x_hat,
        vq_losses,
        x_hat_raq=None,
        vq_losses_raq=None,
    ):
        recon_loss = self._reconstruction_loss(x, x_hat)
        src_vq_loss = self._weighted_vq_loss(vq_losses, recon_loss.device)
        raq_vq_loss = torch.zeros((), device=recon_loss.device)
        weighted_vq = src_vq_loss

        if x_hat_raq is not None and vq_losses_raq is not None:
            recon_loss = recon_loss + self._reconstruction_loss(x, x_hat_raq)
            raq_vq_loss = self._weighted_vq_loss(
                vq_losses_raq, recon_loss.device
            )
            weighted_vq = weighted_vq + raq_vq_loss

        return recon_loss, weighted_vq
