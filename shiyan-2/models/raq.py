import torch
from torch import nn
import math
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from .transformer import TransformerCodebookGen
from .vector_quantizer import ProjectedEmbedding


class RAQ(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        n_embed_src: int,
        n_embed_min_trg: int,
        n_embed_max_trg: int,
        device: str = "cuda:2",
        generator_type: str = "encoder_decoder",
        allocation_conditioned: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim  # [128, 256, 512, 1024]
        self.n_embed_src = n_embed_src
        self.n_embed_min_trg = n_embed_min_trg
        self.n_embed_max_trg = n_embed_max_trg
        self.allocation_conditioned = bool(allocation_conditioned)

        # === SimVQ 改进: 使用 ProjectedEmbedding (冻结底层 + 可训练投影层) ===
        self.trg_embed = ProjectedEmbedding(n_embed_max_trg, embedding_dim).to(self.device)

        # =========================================================
        # 动态配置 Transformer
        # =========================================================

        # 1. d_model 直接等于当前层的 embedding_dim，不进行压缩
        current_d_model = self.embedding_dim

        # 2. 动态设置 FeedForward 层大小，通常是 d_model 的 4 倍
        current_dim_feedforward = current_d_model * 4

        # 3. 检查 nhead 是否合法 (必须能被 d_model 整除)
        # 你的 config 是 [128, 256, 512, 1024]，它们都能被 4 或 8 整除。
        # 这里为了稳妥，使用 8 (128/8=16 也是够用的)
        current_nhead = 8

        self.generator = TransformerCodebookGen(
            src_dim=self.embedding_dim,
            embed_layer_dec = self.trg_embed,
            d_model=current_d_model,
            nhead=current_nhead,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=current_dim_feedforward,
            dropout=0.1,
            device=device,
            generator_type=generator_type,
            max_len=max(n_embed_src, n_embed_max_trg),
        ).to(self.device)

        if self.allocation_conditioned:
            self.allocation_condition = nn.Sequential(
                nn.Linear(3, embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim),
            ).to(self.device)


    def generate_codebook_transformer(
        self,
        k_trg: int,
        vq_weight: torch.Tensor,
        allocation=None,
    ) -> torch.Tensor:
        assert self.n_embed_min_trg <= k_trg <= self.n_embed_max_trg
        if self.allocation_conditioned:
            if allocation is None:
                raise ValueError("allocation-conditioned RAQ requires (K_total, K1, K2)")
            k_total, k_first, k_second = (int(value) for value in allocation)
            max_bits = max(1, int(math.log2(self.n_embed_max_trg)))
            condition_values = vq_weight.new_tensor([
                math.log2(k_total) / max_bits,
                math.log2(k_first) / max_bits,
                math.log2(k_second) / max_bits,
            ]).unsqueeze(0)
            condition_token = self.allocation_condition(condition_values)
            vq_weight = torch.cat([condition_token, vq_weight], dim=0)
        trg_ids = torch.arange(k_trg, device=self.device, dtype=torch.long).unsqueeze(1)
        if self.allocation_conditioned and self.training:
            W = activation_checkpoint(
                self.generator,
                trg_ids,
                vq_weight,
                use_reentrant=False,
            )
        else:
            W = self.generator(trg_ids, vq_weight)
        return W
