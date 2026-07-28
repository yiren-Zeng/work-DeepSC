"""
按论文改写的向量量化器（VQ）：最近邻查找 + ST 估计器 + 嵌入损失与“承诺损失”（通过 vq_loss 返回）
注意：输入/输出使用 BCHW，与论文一致；内部计算临时转 BHWC 便于 view。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim   = embedding_dim  # K（每个向量维度）
        self.num_embeddings  = num_embeddings # N（码本向量个数）
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps


        # 码本：N x K
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # 按常用做法均匀初始化
        self.embeddings.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(self.num_embeddings))
        self.register_buffer("ema_embed", self.embeddings.weight.data.clone())

    def forward(self, inputs: torch.Tensor):
        """
        inputs:  (B, C=embedding_dim, H, W)
        返回：
          vq_loss:        标量，包含 codebook loss + commitment loss
          quantized:      (B,C,H,W) 量化后的特征（带 ST 梯度）
          perplexity:     码本使用熵的指数
          encoding_idx:   (B,H,W) 每个位置选择的码本索引
        """
        # BCHW -> BHWC
        inputs_bhwc = inputs.permute(0,2,3,1).contiguous()
        B,H,W,C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)                        # (B*H*W, C)

        # L2 距离：||z-e||^2 = ||z||^2 + ||e||^2 - 2 z·e
        emb_w = self.embeddings.weight                        # (N, C)
        distances = (flat.pow(2).sum(dim=1, keepdim=True)     # (BHW,1)
                    + emb_w.pow(2).sum(dim=1).unsqueeze(0)    # (1,N)
                    - 2.0 * flat @ emb_w.t())                 # (BHW,N)

        # 最近邻索引 & one-hot
        encoding_idx = torch.argmin(distances, dim=1)         # (BHW,)
        encodings = torch.zeros(encoding_idx.size(0), self.num_embeddings,
                                device=inputs.device, dtype=inputs.dtype)
        encodings.scatter_(1, encoding_idx.view(-1,1), 1.0)   # (BHW,N)

        # 量化向量
        quantized_flat = encodings @ emb_w                    # (BHW,C)
        quantized_bhwc = quantized_flat.view(B, H, W, C)      # (B,H,W,C)

        # ------- EMA update((train only)) -------
        if self.training:
            with torch.no_grad():
                # 统计每个簇的被选次数与被分配向量和
                cluster_size = encodings.sum(0)                              # (N,)
                embed_sum = encodings.t() @ flat                             # (N,C)

                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1-self.decay)
                self.ema_embed.mul_(self.decay).add_(embed_sum,    alpha=1-self.decay)

                # Laplace 平滑防止空簇
                n = self.ema_cluster_size.sum()
                cluster_size = ((self.ema_cluster_size + self.eps) /
                                (n + self.num_embeddings * self.eps) * n)
                self.embeddings.weight.data.copy_(self.ema_embed / cluster_size.unsqueeze(1))

        # 损失（VQ-VAE 标准写法）
        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # ST 估计器：前向用量化值，反传把梯度拷贝回 inputs
        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()

        # 困惑度（码本使用率）
        avg_probs = encodings.mean(dim=0)
        perplexity = torch.exp(-(avg_probs * (avg_probs + 1e-10).log()).sum())

        # BHWC -> BCHW
        quantized = quantized_bhwc.permute(0,3,1,2).contiguous()
        encoding_idx = encoding_idx.view(B, H, W)
        return vq_loss, quantized, perplexity, encoding_idx

    def forward_with_weight(self, inputs: torch.Tensor, embed_weight: torch.Tensor):
        """
        使用【外部码本权重 embed_weight(K̃,C)】进行量化；保持与 forward 基本一致的返回风格。
        Args:
            inputs:       (B, C, H, W)
            embed_weight: (K_tilde, C)
        Returns:
            vq_loss:      标量（codebook + commitment）
            quantized:    (B,C,H,W) 量化后的特征（带 ST 梯度）
            encoding_idx: (B,H,W) 最近邻索引
        """
        # BCHW -> BHWC
        inputs_bhwc = inputs.permute(0,2,3,1).contiguous()
        B,H,W,C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)  # (BHW, C)

        self.emb_w = embed_weight
        # L2 距离
        distances = (flat.pow(2).sum(dim=1, keepdim=True)
                    + self.emb_w.pow(2).sum(dim=1).unsqueeze(0)
                    - 2.0 * flat @ self.emb_w.t())

        # 最近邻
        encoding_idx = torch.argmin(distances, dim=1)         # (BHW,)
        encodings = torch.zeros(encoding_idx.size(0), self.emb_w.size(0),
                                device=inputs.device, dtype=inputs.dtype)
        encodings.scatter_(1, encoding_idx.view(-1,1), 1.0)   # (BHW,K̃)

        # 量化向量
        quantized_flat = encodings @ self.emb_w                    # (BHW,C)
        quantized_bhwc = quantized_flat.view(B, H, W, C)      # (B,H,W,C)

        # 损失
        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # ST
        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()

        # 输出
        quantized = quantized_bhwc.permute(0,3,1,2).contiguous()
        encoding_idx = encoding_idx.view(B, H, W)
        return vq_loss, quantized, encoding_idx

    @torch.no_grad()
    def get_quantized_features_src(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """
        按给定的索引（B,H,W）或（H,W）回查码本，返回 (B,C,H,W)
        """
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)   # (1,H,W)
        B,H,W = encoding_indices.shape
        flat_idx = encoding_indices.reshape(-1)                 # (BHW,)
        quantized_flat = self.embeddings(flat_idx)              # (BHW,C)
        C = quantized_flat.shape[-1]
        quantized_bhwc = quantized_flat.view(B, H, W, C)        # (B,H,W,C)
        return quantized_bhwc.permute(0,3,1,2).contiguous()     # (B,C,H,W)

