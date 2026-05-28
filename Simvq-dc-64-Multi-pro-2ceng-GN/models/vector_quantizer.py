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

    def forward(self, inputs: torch.Tensor):
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        embed_weight = self.codebook.projected_weight()

        B, H, W, C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)

        # Einsum 距离（在归一化空间中计算）
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

        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx.view(B, H, W)

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, H, W = encoding_indices.shape
        flat_idx = encoding_indices.reshape(-1)
        weight = self.codebook.projected_weight()
        quantized_flat = F.embedding(flat_idx, weight)
        C = quantized_flat.shape[-1]
        quantized_bhwc = quantized_flat.view(B, H, W, C)
        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def compute_codebook_stats(encoding_idx, num_embeddings):
        """
        计算码本利用率统计信息

        Args:
            encoding_idx: 量化索引张量，任意形状
            num_embeddings: 码本大小

        Returns:
            dict: {
                'active_ratio': 活跃码字比例 (0~1), 越高越好
                'perplexity': 码本困惑度 (1~num_embeddings), 越接近 num_embeddings 越好
                'active_count': 活跃码字数量
                'dead_count': 死码字数量 (从未被使用)
                'usage_counts': 每个码字的使用次数 [num_embeddings]
            }
        """
        flat_idx = encoding_idx.reshape(-1)

        # 统计每个码字的使用次数
        usage_counts = torch.bincount(flat_idx, minlength=num_embeddings).float()

        # 活跃码字数 (使用次数 > 0 的码字)
        active_mask = usage_counts > 0
        active_count = active_mask.sum().item()
        dead_count = num_embeddings - active_count

        # 活跃比例
        active_ratio = active_count / num_embeddings

        # 困惑度 (Perplexity): 2^H, H = -sum(p * log2(p))
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
        """
        计算码本中码字间的最小 L2 距离，以及坍缩码字统计

        Args:
            embed_weight: 码本权重 [K, D]
            collapse_threshold: 坍缩判定阈值，两个码字 L2 距离小于此值视为坍缩对

        Returns:
            dict: {
                'min_l2_dist': 码字间最小 L2 距离，越大越好
                'collapse_count': 参与坍缩的码字数量（有至少一个邻居距离 < threshold）
                'collapse_ratio': 参与坍缩的码字占总码本的比例 (0~1)
            }
        """
        # 计算所有码字对之间的 L2 距离矩阵
        norm_sq = torch.sum(embed_weight ** 2, dim=1)  # [K]
        dist_sq = norm_sq.unsqueeze(1) + norm_sq.unsqueeze(0) - 2 * torch.matmul(embed_weight, embed_weight.t())  # [K, K]

        K = dist_sq.size(0)
        eye = torch.eye(K, device=dist_sq.device).bool()
        dist_sq = dist_sq.masked_fill(eye, float('inf'))

        # 最小 L2 距离
        min_dist_sq = dist_sq.min()
        min_l2_dist = torch.sqrt(min_dist_sq.clamp(min=0)).item()

        # 坍缩码字统计：每个码字找其最近邻距离，若 < threshold 则视为坍缩
        nearest_dist_sq = dist_sq.min(dim=1).values  # [K] 每个码字到最近邻的距离平方
        nearest_dist = torch.sqrt(nearest_dist_sq.clamp(min=0))
        collapse_mask = nearest_dist < collapse_threshold
        collapse_count = collapse_mask.sum().item()

        return {
            'min_l2_dist': min_l2_dist,
            'collapse_count': int(collapse_count),
            'collapse_ratio': collapse_count / K if K > 0 else 0.0
        }
