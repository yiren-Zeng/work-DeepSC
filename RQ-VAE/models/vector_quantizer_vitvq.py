import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp

from .vector_quantizer import VectorQuantizer as SimVQVectorQuantizer


class ViTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=lambda: nn.GELU(approximate="tanh"),
            drop=0,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class VQBridgeNoCompress(nn.Module):
    """Transform codebook tokens with ViT blocks without spatial compression."""

    def __init__(self, num_embeddings, embedding_dim, depth=2, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList([
            ViTBlock(embedding_dim, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm_final = nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        residual = x
        x = x.unsqueeze(0) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        delta = self.linear(self.norm_final(x)).squeeze(0)
        return residual + delta


QBRIDGE_MODELS = {
    "QBridgeNoCompress-XS": lambda **kwargs: VQBridgeNoCompress(depth=1, num_heads=4, **kwargs),
    "QBridgeNoCompress-S": lambda **kwargs: VQBridgeNoCompress(depth=2, num_heads=4, **kwargs),
    "QBridgeNoCompress-B": lambda **kwargs: VQBridgeNoCompress(depth=4, num_heads=8, **kwargs),
    "QBridgeNoCompress-L": lambda **kwargs: VQBridgeNoCompress(depth=6, num_heads=8, **kwargs),
}


class ViTvqNoCompressVectorQuantizer(nn.Module):
    """VQ with a learnable ViT NoCompress codebook transformation."""

    compute_codebook_stats = staticmethod(SimVQVectorQuantizer.compute_codebook_stats)
    compute_min_l2_distance = staticmethod(SimVQVectorQuantizer.compute_min_l2_distance)

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        commitment_cost,
        qbridge_type="QBridgeNoCompress-S",
        emb_nograd=False,
    ):
        super().__init__()
        if qbridge_type not in QBRIDGE_MODELS:
            raise ValueError(f"Unknown ViTvq NoCompress QBridge type: {qbridge_type}")
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        if emb_nograd:
            self.embedding.requires_grad_(False)

        side = int(math.ceil(math.sqrt(num_embeddings)))
        self.padded_num_embeddings = side * side
        if self.padded_num_embeddings != num_embeddings:
            self.register_buffer(
                "embed_pad",
                torch.zeros(self.padded_num_embeddings - num_embeddings, embedding_dim),
            )
        self.qbridge = QBRIDGE_MODELS[qbridge_type](
            num_embeddings=self.padded_num_embeddings,
            embedding_dim=embedding_dim,
        )

    def transformed_weight(self):
        weight = self.embedding.weight
        if self.padded_num_embeddings != self.num_embeddings:
            weight = torch.cat([weight, self.embed_pad], dim=0)
        return self.qbridge(weight)[:self.num_embeddings]

    def forward(self, inputs):
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        B, H, W, C = inputs_bhwc.shape
        flat = inputs_bhwc.view(-1, C)
        weight = self.transformed_weight()
        encoding_idx = SimVQVectorQuantizer._nearest_code_indices(flat, weight)
        quantized_bhwc = F.embedding(encoding_idx, weight).view(B, H, W, C)
        e_latent_loss = F.mse_loss(quantized_bhwc.detach(), inputs_bhwc)
        q_latent_loss = F.mse_loss(quantized_bhwc, inputs_bhwc.detach())
        vq_loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized_bhwc = inputs_bhwc + (quantized_bhwc - inputs_bhwc).detach()
        return vq_loss, quantized_bhwc.permute(0, 3, 1, 2).contiguous(), encoding_idx.view(B, H, W)

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices):
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        B, H, W = encoding_indices.shape
        quantized = F.embedding(encoding_indices.reshape(-1), self.transformed_weight())
        return quantized.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
