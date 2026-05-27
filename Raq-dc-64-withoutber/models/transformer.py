import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerCodebookGen(nn.Module):
    def __init__(self,
                 src_dim: int,
                 embed_layer_dec: nn.Embedding,
                 d_model: int,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 device: str = "cuda:2"):
        super().__init__()

        self.device = torch.device(device)
        self.embed_layer_dec = embed_layer_dec

        trg_dim = self.embed_layer_dec.embedding_dim

        self.src_project = nn.Linear(src_dim, d_model) if src_dim != d_model else nn.Identity()
        self.trg_project = nn.Linear(trg_dim, d_model) if trg_dim != d_model else nn.Identity()

        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(d_model=d_model,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)

        self.out_project = nn.Linear(d_model, trg_dim) if d_model != trg_dim else nn.Identity()

    def forward(self, trg_ids: torch.Tensor, vq_weight: torch.Tensor) -> torch.Tensor:
        src_emb = self.src_project(vq_weight).unsqueeze(1)
        src_emb = self.pos_encoder(src_emb)

        trg_emb = self.embed_layer_dec(trg_ids)
        trg_emb = self.trg_project(trg_emb)
        trg_emb = self.pos_encoder(trg_emb)

        output = self.transformer(src=src_emb, tgt=trg_emb)

        output = self.out_project(output)

        return output.squeeze(1)