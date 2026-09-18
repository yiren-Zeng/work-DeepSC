import torch
import torch.nn as nn


class DeepSCLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.reconstruction_criterion = nn.MSELoss()

    def forward(self, images, reconstructed_images, vq_losses):
        reconstruction_loss = self.reconstruction_criterion(
            reconstructed_images, images
        )
        if isinstance(vq_losses, (list, tuple)):
            vq_loss = torch.stack(vq_losses).sum()
        else:
            vq_loss = vq_losses
        return reconstruction_loss, vq_loss
