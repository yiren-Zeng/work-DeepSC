import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    传统 VQ 量化器：标准 nn.Embedding，无 SimVQ 投影层
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 传统 VQ: 直接使用可训练 Embedding
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs: torch.Tensor):
        """
        标准量化前向

        Args:
            inputs: (B, C, H, W)
        Returns:
            vq_loss: 量化损失
            quantized: 量化后特征 (B, C, H, W)
            encoding_idx: 量化索引 (B, H, W)
        """
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        embed_weight = self.embedding.weight

        B, H, W, C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)

        # L2 距离计算
        d = torch.sum(flat ** 2, dim=1, keepdim=True) + \
            torch.sum(embed_weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,nd->bn', flat, embed_weight)

        encoding_idx = torch.argmin(d, dim=1)

        # One-hot + 查表
        encodings = torch.zeros(encoding_idx.size(0), embed_weight.size(0),
                                device=inputs_bhwc.device, dtype=inputs_bhwc.dtype)
        encodings.scatter_(1, encoding_idx.view(-1, 1), 1.0)
        quantized_flat = encodings @ embed_weight
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        # VQ Loss
        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # STE
        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()

        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx.view(B, H, W)

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, H, W = encoding_indices.shape
        flat_idx = encoding_indices.reshape(-1)
        quantized_flat = F.embedding(flat_idx, self.embedding.weight)
        C = quantized_flat.shape[-1]
        quantized_bhwc = quantized_flat.view(B, H, W, C)
        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()
