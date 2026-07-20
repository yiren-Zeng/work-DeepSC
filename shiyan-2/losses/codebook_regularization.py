import torch
import torch.nn as nn
import torch.nn.functional as F


class CodebookRepulsionLoss(nn.Module):
    """Hinge repulsion for keeping source codewords separated."""

    def __init__(self, margin=0.5, normalize=True):
        super().__init__()
        self.margin = float(margin)
        self.normalize = bool(normalize)

    def forward(self, codebooks, device=None):
        if not codebooks:
            return torch.zeros((), device=device or "cpu")

        target_device = device or codebooks[0].device
        loss = torch.zeros((), device=target_device)
        valid_layers = 0
        for codebook in codebooks:
            weight = codebook.to(target_device, non_blocking=True)
            if weight.size(0) < 2:
                continue
            if self.normalize:
                weight = F.normalize(weight, p=2, dim=1, eps=1e-12)
            norm_sq = torch.sum(weight ** 2, dim=1)
            dist_sq = norm_sq.unsqueeze(1) + norm_sq.unsqueeze(0) - 2 * torch.matmul(weight, weight.t())
            off_diag = ~torch.eye(dist_sq.size(0), device=target_device, dtype=torch.bool)
            layer_loss = F.relu(self.margin - dist_sq[off_diag]).mean()
            loss = loss + layer_loss
            valid_layers += 1

        if valid_layers == 0:
            return loss
        return loss / valid_layers
