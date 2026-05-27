import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # === SimVQ 改进 1: 冻结底层 Embedding ===
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        nn.init.normal_(self.embeddings.weight, mean=0, std=self.embedding_dim**-0.5)
        
        for p in self.embeddings.parameters():
            p.requires_grad = False
            
        # === SimVQ 改进 2: 引入线性投影层 ===
        # 模型实际训练的是这个投影层，而非底层的 Embedding 向量
        self.embedding_proj = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

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
        quant_codebook = self.embedding_proj(self.embeddings.weight)
        
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
            weight = self.embedding_proj(self.embeddings.weight)
        else:
            weight = raq_weight

        quantized_flat = F.embedding(flat_idx, weight)
        C = quantized_flat.shape[-1]
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()