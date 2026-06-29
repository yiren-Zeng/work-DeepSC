import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentFeatureDistillationLoss(nn.Module):
    """Match RAQ quantized latent features to detached SRC quantized features."""

    def forward(self, src_features, raq_features, device=None):
        if src_features is None or raq_features is None:
            raise ValueError("Latent distillation requires both SRC and RAQ feature lists.")
        if len(src_features) != len(raq_features):
            raise ValueError("SRC and RAQ feature lists must have the same length.")
        if not src_features:
            return torch.zeros((), device=device or "cpu")

        target_device = device or raq_features[0].device
        loss = torch.zeros((), device=target_device)
        for src_feat, raq_feat in zip(src_features, raq_features):
            src_target = src_feat.detach().to(target_device, non_blocking=True)
            raq_pred = raq_feat.to(target_device, non_blocking=True)
            loss = loss + F.mse_loss(raq_pred, src_target)
        return loss / len(src_features)
