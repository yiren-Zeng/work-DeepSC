import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectedEmbedding(nn.Module):
    """
    SimVQ 风格的 Embedding 包装器：冻结底层 Embedding + 可训练线性投影层
    对外表现为一个标准 Embedding 接口，调用时自动完成投影
    """
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        # 1. 底层 Embedding: 冻结，不参与梯度更新
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embed.weight, mean=0, std=embedding_dim ** -0.5)
        for p in self.embed.parameters():
            p.requires_grad = False

        # 2. 线性投影层: 实际训练的是这个投影层
        self.proj = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # 兼容读取 embedding_dim
        self.embedding_dim = embedding_dim

    def forward(self, ids):
        """
        ids: [K, 1] 或 [K]
        返回: 投影后的 Embedding 向量
        """
        return self.proj(self.embed(ids))

    def projected_weight(self):
        """返回投影后的码本权重 [num_embeddings, embedding_dim]"""
        return self.proj(self.embed.weight)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # === SimVQ 改进: 使用 ProjectedEmbedding (冻结底层 + 可训练投影层) ===
        self.codebook = ProjectedEmbedding(num_embeddings, embedding_dim)

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

    def _quantize_core(self, inputs_bhwc, embed_weight):
        """
        核心量化逻辑：使用 SimVQ 的 Einsum 距离计算与自适应损失结构
        """
        B, H, W, C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C) # (B*H*W, C)

        # === SimVQ 改进 3: 高效的 Einsum 距离计算 ===
        # 相比原有的 L2 展开式，这种写法在处理高维特征时显存占用更低
        # d = ||z||^2 + ||e||^2 - 2 * <z, e>
        d = torch.sum(flat ** 2, dim=1, keepdim=True) + \
            torch.sum(embed_weight ** 2, dim=1) - 2 * \
            torch.einsum('bd,nd->bn', flat, embed_weight)

        # 最近邻索引
        encoding_idx = torch.argmin(d, dim=1) # (BHW,)
        
        # 生成 One-hot 编码用于计算量化后的特征
        encodings = torch.zeros(encoding_idx.size(0), embed_weight.size(0),
                                device=inputs_bhwc.device, dtype=inputs_bhwc.dtype)
        encodings.scatter_(1, encoding_idx.view(-1, 1), 1.0)  # (BHW,N)

        quantized_flat = encodings @ embed_weight  # (BHW,C)
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        # 计算损失
        # e_latent_loss: 承诺损失 (Commitment Loss)，强迫编码器特征靠近码本
        # q_latent_loss: 码本损失 (Embedding Loss)，强迫码本靠近编码器特征
        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # 梯度直通估计器 (STE)
        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()

        return vq_loss, quantized_bhwc, encoding_idx.view(B, H, W), encodings, flat

    def forward(self, inputs: torch.Tensor):
        """
        Source 支路：使用经过投影后的源码本进行量化
        """
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        
        # 获取投影后的码本权重
        quant_codebook = self.codebook.projected_weight()
        
        vq_loss, quantized_bhwc, encoding_idx, encodings, flat = self._quantize_core(
            inputs_bhwc, quant_codebook
        )

        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx

    def forward_raq(self, inputs: torch.Tensor, embed_weight: torch.Tensor):
        """
        RAQ 支路：使用 Transformer 生成的动态码本权重
        由于外部传入的 W_trg 已经是经过计算的最终权重，因此不再应用额外的投影
        """
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        
        vq_loss, quantized_bhwc, encoding_idx, _, _ = self._quantize_core(
            inputs_bhwc, embed_weight
        )

        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices: torch.Tensor,
                               raq_weight: torch.Tensor = None) -> torch.Tensor:
        """
        重建接口：如果使用源码本重建，需要手动应用投影层以保持特征一致性
        """
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, H, W = encoding_indices.shape
        flat_idx = encoding_indices.reshape(-1)

        # 核心逻辑：如果是 Source 支路，必须使用投影后的权重
        if raq_weight is None:
            weight = self.codebook.projected_weight()
        else:
            weight = raq_weight

        quantized_flat = F.embedding(flat_idx, weight)
        C = quantized_flat.shape[-1]
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()