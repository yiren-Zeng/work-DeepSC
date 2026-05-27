import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectedEmbedding(nn.Module):
    """
    SimVQ 风格的 Embedding 包装器：冻结底层 Embedding + 可训练线性投影层
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embed.weight, mean=0, std=embedding_dim ** -0.5)
        for p in self.embed.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.embedding_dim = embedding_dim

    def forward(self, ids):
        return self.proj(self.embed(ids))

    def projected_weight(self):
        """返回投影后的码本权重 [num_embeddings, embedding_dim]"""
        return self.proj(self.embed.weight)


class VectorQuantizer(nn.Module):
    """
    SimVQ 量化器：冻结底层 + 可训练投影层
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.codebook = ProjectedEmbedding(num_embeddings, embedding_dim)

    @staticmethod
    def compute_codebook_stats(encoding_idx, num_embeddings):
        flat_idx = encoding_idx.reshape(-1)
        usage_counts = torch.bincount(flat_idx, minlength=num_embeddings).float()

        active_mask = usage_counts > 0
        active_count = active_mask.sum().item()
        dead_count = num_embeddings - active_count
        active_ratio = active_count / num_embeddings

        total = flat_idx.numel()
        if total > 0:
            probs = usage_counts / total
            probs = probs[probs > 0]
            entropy = -torch.sum(probs * torch.log2(probs))
            perplexity = 2 ** entropy
        else:
            perplexity = torch.tensor(1.0)

        return {
            'active_ratio': active_ratio,
            'perplexity': perplexity.item(),
            'active_count': int(active_count),
            'dead_count': int(dead_count),
            'usage_counts': usage_counts
        }

    @staticmethod
    def compute_min_l2_distance(embed_weight, collapse_threshold=0.1):
        norm_sq = torch.sum(embed_weight ** 2, dim=1)
        dist_sq = norm_sq.unsqueeze(1) + norm_sq.unsqueeze(0) - 2 * torch.matmul(embed_weight, embed_weight.t())

        K = dist_sq.size(0)
        eye = torch.eye(K, device=dist_sq.device).bool()
        dist_sq = dist_sq.masked_fill(eye, float('inf'))

        min_dist_sq = dist_sq.min()
        min_l2_dist = torch.sqrt(min_dist_sq.clamp(min=0)).item()

        nearest_dist_sq = dist_sq.min(dim=1).values
        nearest_dist = torch.sqrt(nearest_dist_sq.clamp(min=0))
        collapse_mask = nearest_dist < collapse_threshold
        collapse_count = collapse_mask.sum().item()

        return {
            'min_l2_dist': min_l2_dist,
            'collapse_count': int(collapse_count),
            'collapse_ratio': collapse_count / K if K > 0 else 0.0
        }

    def _quantize_core(self, inputs_bhwc, embed_weight):
        B, H, W, C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)

        d = torch.sum(flat ** 2, dim=1, keepdim=True) + \
            torch.sum(embed_weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,nd->bn', flat, embed_weight)

        encoding_idx = torch.argmin(d, dim=1)

        encodings = torch.zeros(encoding_idx.size(0), embed_weight.size(0),
                                device=inputs_bhwc.device, dtype=inputs_bhwc.dtype)
        encodings.scatter_(1, encoding_idx.view(-1, 1), 1.0)
        quantized_flat = encodings @ embed_weight
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()

        return vq_loss, quantized_bhwc, encoding_idx.view(B, H, W), encodings, flat

    def forward(self, inputs: torch.Tensor):
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        quant_codebook = self.codebook.projected_weight()

        vq_loss, quantized_bhwc, encoding_idx, _, _ = self._quantize_core(
            inputs_bhwc, quant_codebook
        )

        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx

    def forward_raq(self, inputs: torch.Tensor, embed_weight: torch.Tensor):
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()

        vq_loss, quantized_bhwc, encoding_idx, _, _ = self._quantize_core(
            inputs_bhwc, embed_weight
        )

        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices: torch.Tensor,
                              raq_weight: torch.Tensor = None) -> torch.Tensor:
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, H, W = encoding_indices.shape
        flat_idx = encoding_indices.reshape(-1)

        if raq_weight is None:
            weight = self.codebook.projected_weight()
        else:
            weight = raq_weight

        quantized_flat = F.embedding(flat_idx, weight)
        C = quantized_flat.shape[-1]
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()