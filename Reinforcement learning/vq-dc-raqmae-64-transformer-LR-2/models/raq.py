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

        self.src_embed = nn.Embedding(n_embed_src, embedding_dim).to(self.device)
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
            embed_layer_enc=self.src_embed,
            embed_layer_dec=self.trg_embed,
            d_model=current_d_model,  # <--- 关键修改：直接使用当前维度
            nhead=current_nhead,  # <--- 关键修改
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=current_dim_feedforward,  # <--- 关键修改：按比例放大
            dropout=0.1,
            device=device
        ).to(self.device)

    # ----------------------
    # RAQ (Transformer-driven)
    # ----------------------
    def set_src_weight(self, weight: torch.Tensor):
        assert weight.shape[1] == self.embedding_dim, "Dim mismatch for src weight"
        assert weight.shape[0] == self.n_embed_src, "K_src mismatch"
        with torch.no_grad():
            self.src_embed.weight.copy_(weight.to(self.device))

    def generate_codebook_transformer(self, k_trg: int) -> torch.Tensor:
        assert self.n_embed_min_trg <= k_trg <= self.n_embed_max_trg
        src_ids = torch.arange(self.n_embed_src, device=self.device, dtype=torch.long).unsqueeze(1)
        trg_ids = torch.arange(k_trg, device=self.device, dtype=torch.long).unsqueeze(1)

        # 1. 获取 Transformer 预测的 偏移量 (Delta)
        delta_W = self.generator(src_ids, trg_ids)  # [k_trg, embedding_dim]

        # 2. 【核心修改】：构建物理基底 (Base)
        # 将现有的 src_embed.weight (大小为 K_src) 循环填充/上采样到 k_trg 大小
        src_weight = self.src_embed.weight.detach()
        num_repeats = (k_trg // self.n_embed_src) + 1
        base_W = src_weight.repeat(num_repeats, 1)[:k_trg, :]  # 截断到目标长度

        # 【新增】：对称性破缺噪声 (Symmetry-Breaking Jitter)
        # 加入一个极其微小的正态分布噪声（方差 1e-4），这足以让 argmin 命中不同的克隆体，而不会破坏图像语义
        jitter = torch.randn_like(base_W) * 1e-4
        base_W = base_W + jitter

        # 3. 最终码本 = 物理基底 + 网络预测的残差偏移
        W_trg = base_W + delta_W

        return W_trg
