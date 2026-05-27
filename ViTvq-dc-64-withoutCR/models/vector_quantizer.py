import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import Attention, Mlp


#################################################################################
#                     QBridge-ViT NoCompress Core Components                    #
#                                                                               #
#  去掉原始 VQBridge 中的隐式压缩-恢复结构：                                     #
#    原始: PatchEmbed(空间+通道压缩) → ViT → FinalLayer+unpatchify(恢复)        #
#    现在: 直接将每个码字作为独立token → ViT自注意力 → Linear(无压缩)             #
#                                                                               #
#  关键区别：                                                                    #
#    - 不再使用 PatchEmbed（无空间压缩 8×8→2×2）                                #
#    - hidden_size = embedding_dim（无通道压缩 128→64）                         #
#    - 不再使用 FinalLayer + unpatchify（无恢复操作）                            #
#    - 新增可学习位置编码，替代 PatchEmbed 的隐式位置信息                         #
#################################################################################


class ViTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VQBridgeNoCompress(nn.Module):
    """
    无隐式压缩-恢复的 VQBridge。

    将码本中每个码字视为独立 token，直接在完整维度上做自注意力变换。
    不使用 PatchEmbed（空间压缩）和 FinalLayer+unpatchify（通道/空间恢复）。

    对比原始 VQBridge:
      原始: [K,D] → reshape为[1,D,h,w] → PatchEmbed(空间+通道压缩) → ViT → FinalLayer(恢复) → unpatchify(恢复) → [1,D,h,w] → reshape回[K,D]
      现在: [K,D] → unsqueeze为[1,K,D] → +位置编码 → ViT(D不变) → Linear(D→D) → squeeze回[K,D]

    整个过程不需要 [1,C,H,W] 的 2D 图像格式，直接在 [1,K,D] 的 token 序列上操作。
    """
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 可学习位置编码：为每个码字位置提供位置信息
        # 替代原始 PatchEmbed 中通过卷积隐式获取的位置信息
        self.pos_embed = nn.Parameter(torch.zeros(1, num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ViT blocks: hidden_size = embedding_dim，无通道压缩
        self.blocks = nn.ModuleList([
            ViTBlock(embedding_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        # 最终投影层：D → D，无维度变化
        # zero-init 使初始为恒等映射，与原始 VQBridge 一致
        self.norm_final = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Zero-out output layers: 初始为恒等映射
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        """
        x: (K, D) 码本 embedding，每个码字已经是独立向量

        数据流:
          [K, D] → unsqueeze → [1, K, D]
          → + pos_embed → ViT blocks → norm → linear → [1, K, D]
          → squeeze → [K, D]

        无需 reshape 为 [1,C,H,W]，直接在 token 序列上操作。
        """
        # [K, D] -> [1, K, D]
        x = x.unsqueeze(0)

        # 加入可学习位置编码
        x = x + self.pos_embed

        # ViT 自注意力：token 间全局交互，维度保持 D 不变
        for block in self.blocks:
            x = block(x)

        # 最终投影：D → D，零初始化保证初始为恒等
        x = self.norm_final(x)
        x = self.linear(x)

        # [1, K, D] -> [K, D]
        x = x.squeeze(0)

        return x


#################################################################################
#                     QBridge NoCompress Configurations                         #
#################################################################################

def QBridgeNoCompress_XS(**kwargs):
    """depth=1, 轻量级"""
    return VQBridgeNoCompress(depth=1, num_heads=4, mlp_ratio=2.0, **kwargs)

def QBridgeNoCompress_S(**kwargs):
    """depth=2, 标准版（对应原始 QBridge-S/4）"""
    return VQBridgeNoCompress(depth=2, num_heads=4, mlp_ratio=2.0, **kwargs)

def QBridgeNoCompress_B(**kwargs):
    """depth=4, 较大版"""
    return VQBridgeNoCompress(depth=4, num_heads=8, mlp_ratio=2.0, **kwargs)

def QBridgeNoCompress_L(**kwargs):
    """depth=6, 大版"""
    return VQBridgeNoCompress(depth=6, num_heads=8, mlp_ratio=2.0, **kwargs)


QBridge_models = {
    'QBridgeNoCompress-XS': QBridgeNoCompress_XS,
    'QBridgeNoCompress-S': QBridgeNoCompress_S,
    'QBridgeNoCompress-B': QBridgeNoCompress_B,
    'QBridgeNoCompress-L': QBridgeNoCompress_L,
}


#################################################################################
#              VectorQuantizer with QBridge-ViT NoCompress                      #
#################################################################################

class VectorQuantizer(nn.Module):
    """
    VQ 量化器 + QBridge-ViT 无压缩版码本变换

    核心思想与原始版本相同：将码本 embedding 通过 Transformer 变换，
    使码字之间建立全局依赖关系，提升码本利用率。

    区别在于变换模块：
    - 原始: VQBridge 将码本视为 2D 图像，通过 PatchEmbed 压缩 → ViT → FinalLayer 恢复
    - 现在: VQBridgeNoCompress 将码本中每个码字视为独立 token，直接做自注意力，无压缩-恢复
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost,
                 decay=0.99, eps=1e-5, QB_type='QBridgeNoCompress-S', emb_nograd=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 原始码本 embedding（可学习）
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # 可选：冻结原始 embedding，只训练 QBridge
        if emb_nograd:
            self.embedding.requires_grad_(False)

        # 如果 num_embeddings 不是完全平方数，需要 padding 到完全平方数
        # 这是为了让 VQBridgeNoCompress 的 pos_embed 大小与实际码字数对齐
        self.h_ = int(math.ceil(math.sqrt(num_embeddings)))
        self.w_ = self.h_
        self.padded_num_embeddings = self.h_ * self.w_
        if self.padded_num_embeddings != num_embeddings:
            pad_weight = torch.zeros(self.padded_num_embeddings - num_embeddings, embedding_dim)
            self.register_buffer('embed_pad', pad_weight)

        # QBridge-ViT 无压缩模型：对码本做变换
        # 注意：传入 num_embeddings 和 embedding_dim，而非 input_size 和 in_channels
        self.qbridge = QBridge_models[QB_type](
            num_embeddings=self.padded_num_embeddings, embedding_dim=embedding_dim
        )

    def _get_transformed_codebook(self):
        """
        将原始码本通过 QBridge-ViT NoCompress 变换，返回变换后的码本 [n_e, e_dim]

        数据流（对比原始版本）:
          原始: [n_e, e_dim] → reshape为[1,e_dim,h_,w_] → VQBridge(含压缩-恢复) → reshape回[n_e, e_dim]
          现在: [n_e, e_dim] → 直接传入 VQBridgeNoCompress → [n_e, e_dim]

        不再需要 reshape 为 [1,C,H,W] 的 2D 图像格式。
        """
        # 获取 embedding 权重，若需要则 padding
        if self.padded_num_embeddings != self.num_embeddings:
            emb_weight = torch.cat([self.embedding.weight, self.embed_pad], dim=0)
        else:
            emb_weight = self.embedding.weight

        # [padded_n_e, e_dim] → VQBridgeNoCompress → [padded_n_e, e_dim]
        emb_weights = self.qbridge(emb_weight)

        # 截取实际码本大小
        emb_weights = emb_weights[:self.num_embeddings]  # [n_e, e_dim]

        return emb_weights

    def forward(self, inputs: torch.Tensor):
        """
        QBridge-ViT NoCompress 量化前向

        Args:
            inputs: (B, C, H, W)
        Returns:
            vq_loss: 量化损失
            quantized: 量化后特征 (B, C, H, W)
            encoding_idx: 量化索引 (B, H, W)
        """
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)

        # 获取 QBridge-ViT 变换后的码本
        embedding = self._get_transformed_codebook()

        # L2 距离计算
        d = torch.sum(flat ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding ** 2, dim=1) - 2 * \
            torch.einsum('bd,nd->bn', flat, embedding)

        encoding_idx = torch.argmin(d, dim=1)

        # 查找量化结果
        quantized_flat = embedding[encoding_idx]
        quantized_bhwc = quantized_flat.view(B, H, W, C)

        # VQ Loss
        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # STE (Straight-Through Estimator)
        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()

        quantized = quantized_bhwc.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_idx.view(B, H, W)

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        """根据索引从变换后的码本获取量化特征"""
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, H, W = encoding_indices.shape

        # 获取变换后的码本
        embedding = self._get_transformed_codebook()

        flat_idx = encoding_indices.reshape(-1)
        quantized_flat = F.embedding(flat_idx, embedding)
        C = quantized_flat.shape[-1]
        quantized_bhwc = quantized_flat.view(B, H, W, C)
        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()

    @torch.no_grad()
    def output_codebook(self):
        """输出变换后的完整码本 [1, n_e, e_dim]"""
        return self._get_transformed_codebook().unsqueeze(0)

    @staticmethod
    def compute_codebook_stats(encoding_idx, num_embeddings):
        """
        计算码本利用率统计信息
        """
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
        """
        计算码本中码字间的最小 L2 距离，以及坍缩码字统计
        """
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
