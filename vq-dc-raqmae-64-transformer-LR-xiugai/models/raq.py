import torch
from torch import nn
import torch.nn.functional as F
from .transformer import TransformerCodebookGen



class RAQ(nn.Module):
    def __init__(self, embedding_dim: int, n_embed_src: int, n_embed_min_trg: int, n_embed_max_trg: int,
                 device: str = "cuda:2"):
        super().__init__()
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim  # [128, 256, 512, 1024]
        self.n_embed_src = n_embed_src
        self.n_embed_min_trg = n_embed_min_trg
        self.n_embed_max_trg = n_embed_max_trg

        # 【核心修改 1】：彻底删除 self.src_embed = nn.Embedding(...)
        # 只保留 target 端的 Embedding (因为 target 依然需要生成 Query IDs)
        self.trg_embed = nn.Embedding(n_embed_max_trg, embedding_dim).to(self.device)

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
            embed_layer_dec=self.trg_embed,
            d_model=current_d_model,  # <--- 关键修改：直接使用当前维度
            nhead=current_nhead,  # <--- 关键修改
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=current_dim_feedforward,  # <--- 关键修改：按比例放大
            dropout=0.1,
            device=device
        ).to(self.device)


    def generate_codebook_transformer(self, k_trg: int, vq_weight: torch.Tensor) -> torch.Tensor:
        assert self.n_embed_min_trg <= k_trg <= self.n_embed_max_trg
        trg_ids = torch.arange(k_trg, device=self.device, dtype=torch.long).unsqueeze(1)
        W = self.generator(trg_ids, vq_weight)
        return W
