import torch
import torch.nn as nn
import torch.nn.functional as F
import os


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

    def transformed_weight(self):
        return self.codebook.projected_weight()

    @staticmethod
    @torch.no_grad()
    def _nearest_code_indices(flat, embed_weight, max_distance_elements=16_777_216):
        """Return exact nearest-code indices without materializing an oversized matrix."""
        max_distance_elements = int(
            os.environ.get("SIMVQ_MAX_DISTANCE_ELEMENTS", str(max_distance_elements))
        )
        codebook_size = embed_weight.size(0)
        chunk_size = max(1, max_distance_elements // codebook_size)
        embed_norm_sq = torch.sum(embed_weight.detach() ** 2, dim=1)
        indices = []
        for start in range(0, flat.size(0), chunk_size):
            chunk = flat[start:start + chunk_size].detach()
            distances = torch.sum(chunk ** 2, dim=1, keepdim=True) + embed_norm_sq - 2 * \
                torch.einsum('bd,nd->bn', chunk, embed_weight.detach())
            indices.append(torch.argmin(distances, dim=1))
        return torch.cat(indices, dim=0)

    def forward(self, inputs: torch.Tensor):
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        embed_weight = self.transformed_weight()

        B, H, W, C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)

        # The nearest-code decision is non-differentiable. Compute it in
        # bounded chunks, then gather selected projected embeddings directly.
        encoding_idx = self._nearest_code_indices(flat, embed_weight)
        quantized_flat = F.embedding(encoding_idx, embed_weight)
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()

        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx.view(B, H, W)

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices: torch.Tensor, output_spatial_size=None) -> torch.Tensor:
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, H, W = encoding_indices.shape
        flat_idx = encoding_indices.reshape(-1)
        weight = self.transformed_weight()
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
    def compute_min_l2_distance(embed_weight, collapse_threshold=0.1, max_reference_codes=4096):
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
        # Compute exact nearest-neighbor distances in bounded chunks. A dense
        # K x K matrix is not viable for large experimental codebooks.
        K = embed_weight.size(0)
        if K > max_reference_codes:
            reference_indices = torch.linspace(
                0, K - 1, steps=max_reference_codes, device=embed_weight.device
            ).long()
        else:
            reference_indices = torch.arange(K, device=embed_weight.device)
        reference_weight = embed_weight[reference_indices]
        norm_sq = torch.sum(embed_weight ** 2, dim=1)
        chunk_size = max(1, 16_777_216 // K)
        nearest_chunks = []
        for start in range(0, reference_weight.size(0), chunk_size):
            end = min(start + chunk_size, reference_weight.size(0))
            chunk = reference_weight[start:end]
            dist_sq = torch.sum(chunk ** 2, dim=1, keepdim=True) + norm_sq.unsqueeze(0) - 2 * \
                torch.matmul(chunk, embed_weight.t())
            row = torch.arange(end - start, device=embed_weight.device)
            col = reference_indices[start:end]
            dist_sq[row, col] = float('inf')
            nearest_chunks.append(dist_sq.min(dim=1).values)
        nearest_dist_sq = torch.cat(nearest_chunks)
        min_l2_dist = torch.sqrt(nearest_dist_sq.min().clamp(min=0)).item()

        # 坍缩码字统计：每个码字找其最近邻距离，若 < threshold 则视为坍缩
        nearest_dist = torch.sqrt(nearest_dist_sq.clamp(min=0))
        collapse_mask = nearest_dist < collapse_threshold
        sampled_collapse_count = collapse_mask.sum().item()
        collapse_ratio = sampled_collapse_count / reference_weight.size(0) if reference_weight.size(0) > 0 else 0.0
        collapse_count = round(collapse_ratio * K)

        return {
            'min_l2_dist': min_l2_dist,
            'collapse_count': int(collapse_count),
            'collapse_ratio': collapse_ratio,
            'distance_reference_count': reference_weight.size(0),
            'distance_stats_exact': reference_weight.size(0) == K,
        }


class ChannelwiseVectorQuantizer(VectorQuantizer):
    """
    Channel-wise SimVQ quantizer.

    Each token is one full channel activation map. The codebook is trained at a
    fixed spatial codeword size, and feature maps with different resolution are
    resized before lookup and resized back after dequantization.
    """
    def __init__(self, num_embeddings, codeword_shape, commitment_cost):
        self.codeword_shape = tuple(int(v) for v in codeword_shape)
        embedding_dim = self.codeword_shape[0] * self.codeword_shape[1]
        super().__init__(num_embeddings, embedding_dim, commitment_cost)

    def _resize_to_codeword(self, inputs: torch.Tensor) -> torch.Tensor:
        if tuple(inputs.shape[-2:]) == self.codeword_shape:
            return inputs
        return F.interpolate(
            inputs,
            size=self.codeword_shape,
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, inputs: torch.Tensor):
        B, C, H, W = inputs.shape
        resized = self._resize_to_codeword(inputs)
        Hq, Wq = resized.shape[-2:]
        flat = resized.reshape(B * C, Hq * Wq)

        embed_weight = self.transformed_weight()
        encoding_idx = self._nearest_code_indices(flat, embed_weight)
        quantized_flat = F.embedding(encoding_idx, embed_weight)
        quantized_small = quantized_flat.view(B, C, Hq, Wq)

        e_latent_loss = F.mse_loss(quantized_small.detach(), resized)
        q_latent_loss = F.mse_loss(quantized_small, resized.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized_small = resized + (quantized_small - resized).detach()
        if (Hq, Wq) != (H, W):
            quantized = F.interpolate(
                quantized_small,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        else:
            quantized = quantized_small
        return vq_loss, quantized.contiguous(), encoding_idx.view(B, C)

    @torch.no_grad()
    def get_quantized_features(
        self,
        encoding_indices: torch.Tensor,
        output_spatial_size=None,
    ) -> torch.Tensor:
        if encoding_indices.dim() == 1:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, C = encoding_indices.shape
        flat_idx = encoding_indices.reshape(-1)
        quantized_flat = F.embedding(flat_idx, self.transformed_weight())
        quantized = quantized_flat.view(B, C, *self.codeword_shape)
        if output_spatial_size is not None and tuple(output_spatial_size) != self.codeword_shape:
            quantized = F.interpolate(
                quantized,
                size=tuple(output_spatial_size),
                mode="bilinear",
                align_corners=False,
            )
        return quantized.contiguous()


class VanillaVectorQuantizer(nn.Module):
    """
    原始 VQ 量化器：码本 Embedding 直接参与训练，不使用 SimVQ 的冻结底层
    Embedding + 投影层结构。
    """
    compute_codebook_stats = staticmethod(VectorQuantizer.compute_codebook_stats)
    compute_min_l2_distance = staticmethod(VectorQuantizer.compute_min_l2_distance)

    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def transformed_weight(self):
        return self.embedding.weight

    def forward(self, inputs: torch.Tensor):
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)

        embed_weight = self.transformed_weight()
        encoding_idx = VectorQuantizer._nearest_code_indices(flat, embed_weight)
        quantized_flat = F.embedding(encoding_idx, embed_weight)
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()
        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx.view(B, H, W)

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices: torch.Tensor, output_spatial_size=None) -> torch.Tensor:
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, H, W = encoding_indices.shape
        flat_idx = encoding_indices.reshape(-1)
        quantized_flat = F.embedding(flat_idx, self.transformed_weight())
        return quantized_flat.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
