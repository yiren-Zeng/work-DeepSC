import torch
from torch import nn
from .transformer import TransformerCodebookGen
from .vector_quantizer import ProjectedEmbedding


class RAQ(nn.Module):
    def __init__(self, embedding_dim: int, n_embed_src: int, n_embed_min_trg: int, n_embed_max_trg: int,
                 device: str = "cuda:2"):
        super().__init__()
        self.device = torch.device(device)
        self.embedding_dim = embedding_dim
        self.n_embed_src = n_embed_src
        self.n_embed_min_trg = n_embed_min_trg
        self.n_embed_max_trg = n_embed_max_trg

        self.trg_embed = ProjectedEmbedding(n_embed_max_trg, embedding_dim).to(self.device)

        current_d_model = self.embedding_dim
        current_dim_feedforward = current_d_model * 4
        current_nhead = 8

        self.generator = TransformerCodebookGen(
            src_dim=self.embedding_dim,
            embed_layer_dec=self.trg_embed,
            d_model=current_d_model,
            nhead=current_nhead,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=current_dim_feedforward,
            dropout=0.1,
            device=device
        ).to(self.device)

    def generate_codebook_transformer(self, k_trg: int, vq_weight: torch.Tensor) -> torch.Tensor:
        assert self.n_embed_min_trg <= k_trg <= self.n_embed_max_trg
        trg_ids = torch.arange(k_trg, device=self.device, dtype=torch.long).unsqueeze(1)
        W = self.generator(trg_ids, vq_weight)
        return W