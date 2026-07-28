import torch
from torch import nn
import torch.nn.functional as F
from .seq2seq import CdBkEncoder, CdBkDecoder, CdBk2CdBk
from .clustering import dkm, inverse_dkm, RAQClusterArgs

class RAQ(nn.Module):
    """
    Unified RAQ interface:
      - generate_codebook_seq2seq(k_trg): data-driven generator (needs training)
      - generate_codebook_dkm(src_weight, k_trg, args): down-rate via clustering
      - generate_codebook_ikm(src_weight, k_trg, args): up-rate via inverse-dkm
      - quantize_with(W, z_e): quantize any latent to the given codebook
    """
    def __init__(self, embedding_dim: int, n_embed_src: int,n_embed_min_trg:int, n_embed_max_trg: int, device: str = "cuda:2"):
        super().__init__()
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.n_embed_src = n_embed_src
        self.n_embed_min_trg = n_embed_min_trg
        self.n_embed_max_trg = n_embed_max_trg

        # Embedding tables for seq2seq (dec input) & placeholder for src (set via set_src_weight)
        self.src_embed = nn.Embedding(n_embed_src, embedding_dim).to(self.device)
        self.trg_embed = nn.Embedding(n_embed_max_trg, embedding_dim).to(self.device)

        # LSTM encoder/decoder
        self.enc = CdBkEncoder(embedding_dim, embedding_dim, n_layers=2, dropout=0.5).to(self.device)
        self.dec = CdBkDecoder(embedding_dim, embedding_dim, n_layers=2, dropout=0.5).to(self.device)
        self.cbk2cbk = CdBk2CdBk(self.enc, self.dec, self.src_embed, self.trg_embed, device = self.device).to(self.device)


    # ----------------------
    # Seq2Seq RAQ (data-driven)
    # ----------------------
    def set_src_weight(self, weight: torch.Tensor):
        """把【量化器里的源码本权重 E】拷贝到 Seq2Seq 的查表层"""
        assert weight.shape[1] == self.embedding_dim, "Dim mismatch for src weight"
        assert weight.shape[0] == self.n_embed_src,  "K_src mismatch — init RAQ with correct n_embed_src"
        with torch.no_grad():
            self.src_embed.weight.copy_(weight.to(self.device))

    def generate_codebook_seq2seq(self, k_trg: int) -> torch.Tensor:
        assert self.n_embed_min_trg <= k_trg <= self.n_embed_max_trg, f"k_trg={k_trg} must be in [{self.n_embed_min_trg}, {self.n_embed_max_trg}]"
        src_ids = torch.arange(self.n_embed_src, device=self.device, dtype=torch.long).unsqueeze(1) # [S,1]
        trg_ids = torch.arange(k_trg,            device=self.device, dtype=torch.long).unsqueeze(1) # [T,1]
        W = self.cbk2cbk(src_ids, trg_ids).squeeze(1)                             # [T, D]
        return W

    # ----------------------
    # Model-based RAQ
    # ----------------------
    def generate_codebook_dkm(self, src_weight: torch.Tensor, k_trg: int, args: RAQClusterArgs) -> torch.Tensor:
        W, _ = dkm(src_weight.to(self.device), k=k_trg, args=args)
        return W

    def generate_codebook_ikm(self, src_weight: torch.Tensor, k_trg: int, args: RAQClusterArgs):
        W, loss = inverse_dkm(src_weight.to(self.device), k_target=k_trg, embedding_dim=self.embedding_dim, args=args)
        return W

    @torch.no_grad()
    def get_quantized_features_raq(self, encoding_indices: torch.Tensor, embed_weight: torch.Tensor) -> torch.Tensor:
        """
        使用指定的码本权重进行重建
        Args:
            encoding_indices: 编码索引，形状为 (B, H, W) 或 (H, W)
            embed_weight: 码本权重，形状为 (num_embeddings, embedding_dim)

        Returns:
            量化后的特征，形状为 (B, C, H, W)
        """
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)  # (1,H,W)

        B, H, W = encoding_indices.shape

        # 展平索引: (B, H, W) -> (B*H*W,)
        flat_idx = encoding_indices.reshape(-1)  # (BHW,)

        # 从指定码本中查找对应的向量
        # flat_indices: (B*H*W,), embed_weight: (K, C)，这个就是指定的码本 -> quantized: (B*H*W, C)
        quantized_flat = F.embedding(flat_idx, embed_weight)  # (BHW,C)

        # 获取特征维度
        C = quantized_flat.shape[-1]

        # 恢复空间维度: (B*H*W, C) -> (B, H, W, C)
        quantized_bhwc = quantized_flat.view(B, H, W, C)  # (B,H,W,C)

        # 调整维度顺序: (B, H, W, C) -> (B, C, H, W)
        return quantized_bhwc.permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)

