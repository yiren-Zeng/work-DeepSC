import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp


#################################################################################
#                          QBridge-ViT Core Components                         #
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


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class VQBridge(nn.Module):
    """
    VQBridge model with a Transformer backbone.
    Transforms the codebook via ViT self-attention for full codebook utilization.
    """
    def __init__(
        self,
        input_size=8,
        patch_size=4,
        in_channels=128,
        head_hidden_size=16,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        hidden_size = head_hidden_size * num_heads
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)

        self.blocks = nn.ModuleList([
            ViTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Zero-out output layers: QBridge starts as identity mapping
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x):
        """
        x: (N, C, H, W) codebook as 2D image
        """
        x = self.x_embedder(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(x)
        x = self.unpatchify(x)
        return x


#################################################################################
#                          QBridge Configurations                              #
#################################################################################

def QBridge_XS_2(**kwargs):
    return VQBridge(depth=2, patch_size=2, head_hidden_size=8, num_heads=4, **kwargs)

def QBridge_S_2(**kwargs):
    return VQBridge(depth=2, patch_size=2, head_hidden_size=16, num_heads=4, **kwargs)

def QBridge_S_4(**kwargs):
    return VQBridge(depth=2, patch_size=4, head_hidden_size=16, num_heads=4, **kwargs)

def QBridge_S_8(**kwargs):
    return VQBridge(depth=2, patch_size=8, head_hidden_size=16, num_heads=4, **kwargs)

def QBridge_B_2(**kwargs):
    return VQBridge(depth=2, patch_size=2, head_hidden_size=32, num_heads=4, **kwargs)

def QBridge_B_4(**kwargs):
    return VQBridge(depth=2, patch_size=4, head_hidden_size=32, num_heads=4, **kwargs)

def QBridge_B_8(**kwargs):
    return VQBridge(depth=2, patch_size=8, head_hidden_size=32, num_heads=4, **kwargs)

def QBridge_L_2(**kwargs):
    return VQBridge(depth=2, patch_size=2, head_hidden_size=64, num_heads=4, **kwargs)

def QBridge_L_4(**kwargs):
    return VQBridge(depth=2, patch_size=4, head_hidden_size=64, num_heads=4, **kwargs)

def QBridge_L_8(**kwargs):
    return VQBridge(depth=2, patch_size=8, head_hidden_size=64, num_heads=4, **kwargs)


QBridge_models = {
    'QBridge-XS/2': QBridge_XS_2,
    'QBridge-S/2': QBridge_S_2,
    'QBridge-S/4': QBridge_S_4,
    'QBridge-S/8': QBridge_S_8,
    'QBridge-B/2': QBridge_B_2,
    'QBridge-B/4': QBridge_B_4,
    'QBridge-B/8': QBridge_B_8,
    'QBridge-L/2': QBridge_L_2,
    'QBridge-L/4': QBridge_L_4,
    'QBridge-L/8': QBridge_L_8,
}


#################################################################################
#              VectorQuantizer with QBridge-ViT (FVQ style)                    #
#################################################################################

class VectorQuantizer(nn.Module):
    """
    VQ 量化器 + QBridge-ViT 码本变换（改编自 VQBridge-ViT / FVQ）
    
    核心思想：将码本 embedding reshape 为 2D 图像，通过 ViT 自注意力变换，
    使码字之间建立空间依赖关系，从而实现接近 100% 的码本利用率。
    
    与传统 VQ 的区别：
    - 传统 VQ: 每个码字独立，直接用 nn.Embedding 查表
    - QBridge-ViT: 码本先经过 ViT 变换，码字之间有交互，再用于最近邻查找
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, 
                 decay=0.99, eps=1e-5, QB_type='QBridge-S/4', emb_nograd=False):
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

        # 将码本数量拆解为 2D 空间维度 (h_ x w_)
        # num_embeddings 必须能开平方，若不能则向上取到最近的完全平方数
        self.h_ = int(math.ceil(math.sqrt(num_embeddings)))
        self.w_ = self.h_
        # 如果 num_embeddings 不是完全平方数，需要 padding
        self.padded_num_embeddings = self.h_ * self.w_
        if self.padded_num_embeddings != num_embeddings:
            # 需要对 embedding 进行 padding
            pad_weight = torch.zeros(self.padded_num_embeddings - num_embeddings, embedding_dim)
            self.register_buffer('embed_pad', pad_weight)

        # QBridge-ViT 模型：对码本做变换
        self.qbridge = QBridge_models[QB_type](
            input_size=self.h_, in_channels=embedding_dim
        )

    def _get_transformed_codebook(self):
        """
        将原始码本通过 QBridge-ViT 变换，返回变换后的码本 [n_e, e_dim]
        """
        # 获取 embedding 权重，若需要则 padding
        if self.padded_num_embeddings != self.num_embeddings:
            emb_weight = torch.cat([self.embedding.weight, self.embed_pad], dim=0)
        else:
            emb_weight = self.embedding.weight

        # [n_e, e_dim] -> [1, e_dim, h_, w_]
        emb_weights = emb_weight.permute(1, 0).reshape(1, self.embedding_dim, self.h_, self.w_)

        # QBridge-ViT 变换
        emb_weights = self.qbridge(emb_weights)  # [1, e_dim, h_, w_]

        # [1, e_dim, h_, w_] -> [padded_n_e, e_dim]
        emb_weights = emb_weights.reshape(1, self.embedding_dim, self.padded_num_embeddings).permute(0, 2, 1).squeeze(0)

        # 截取实际码本大小
        emb_weights = emb_weights[:self.num_embeddings]  # [n_e, e_dim]

        return emb_weights

    def forward(self, inputs: torch.Tensor):
        """
        QBridge-ViT 量化前向

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
