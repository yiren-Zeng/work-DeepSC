import torch
from torch import nn
import torch.nn.functional as F

class Quantizer(nn.Module):
    """
    Lightweight vector-quantizer that snaps latent z_e (B,H,W,D) to the
    **nearest rows** of an arbitrary codebook weight matrix W [K,D].
    """
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, z_e: torch.Tensor, embed_weight: torch.Tensor):
        """
        Args:
            z_e: [B, H, W, D]
            embed_weight: [K, D]
        Returns:
            z_q: quantized latent [B, H, W, D]
            diff: commitment + codebook loss (VQ-style)
            embed_ind: [B, H, W] chosen indices in 0..K-1
        """
        flatten = z_e.reshape(-1, self.embedding_dim)  # [BHW, D]
        # squared L2 distance to all code vectors
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ embed_weight.t()
            + embed_weight.pow(2).sum(1, keepdim=True).t()
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*z_e.shape[:-1])
        z_q = F.embedding(embed_ind, embed_weight)

        # VQ losses
        commitment_cost = 0.25
        diff = commitment_cost * (z_q.detach() - z_e).pow(2).mean() + (z_q - z_e.detach()).pow(2).mean()

        # Straight-through
        z_q = z_e + (z_q - z_e).detach()
        return z_q, diff, embed_ind
