import torch
from torch import nn

class CdBkEncoder(nn.Module):
    """
    LSTM encoder over codebook embeddings.
    """
    def __init__(self, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # x: [seq_len, batch=1, emb_dim]
        embedded = self.dropout(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell


class CdBkDecoder(nn.Module):
    """
    LSTM decoder that predicts target **embedding vectors** (continuous), not token IDs.
    """
    def __init__(self, emb_dim: int, hid_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor):
        # x: [1, batch=1, emb_dim]
        embedded = self.dropout(x)
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        out = output.squeeze(0)   # [batch=1, emb_dim]
        return out, hidden, cell


class CdBk2CdBk(nn.Module):
    """
    Codebook-to-Codebook generator (Seq2Seq) with **cross-forcing**:
      - Even steps <= 2*|src| feed src tokens (teacher-forcing on source)
      - Otherwise feed previous target output (free-running on target)
    It **returns a sequence of target embedding vectors**, which we stack as the new codebook.
    """
    def __init__(self, encoder: CdBkEncoder, decoder: CdBkDecoder,
                 embed_layer_enc: nn.Embedding, embed_layer_dec: nn.Embedding,
                 device: str = "cuda:2"):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_layer_enc = embed_layer_enc
        self.embed_layer_dec = embed_layer_dec
        self.device = torch.device(device)

        assert encoder.hid_dim == decoder.hid_dim, "Encoder/decoder hid-dim mismatch"
        assert encoder.n_layers == decoder.n_layers, "Encoder/decoder n_layers mismatch"

    def embed_src(self, cd_ids: torch.Tensor):
        # cd_ids: [seq_len] or [seq_len,1]
        return self.embed_layer_enc(cd_ids.unsqueeze(0))  # [1, seq_len, emb_dim] -> we will transpose if needed

    def embed_trg(self, cd_ids: torch.Tensor):
        return self.embed_layer_dec(cd_ids.unsqueeze(0))

    def forward(self, src_ids: torch.Tensor, trg_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_ids: [S, 1] tensor of source codebook indices (0..K_src-1)
            trg_ids: [T, 1] tensor of desired target "token positions" (0..K_trg-1)
        Returns:
            outputs: [T, 1, emb_dim] — predicted target embedding vectors (stacked = target codebook)
        """
        batch_size = trg_ids.shape[1]
        trg_len = trg_ids.shape[0]

        emb_dim = self.embed_layer_enc.embedding_dim
        outputs = torch.zeros(trg_len, batch_size, emb_dim, device=self.device)

        # Initial hidden state encodes the **source** codebook (as a sequence)
        hidden, cell = self.encoder(self.embed_layer_enc(src_ids))

        # Start decoding
        inp = self.embed_trg(trg_ids[0])
        for t in range(trg_len):
            out, hidden, cell = self.decoder(inp, hidden, cell)
            outputs[t] = out  # out ~ target embedding vector at position t

            # Cross-forcing schedule
            if (t % 2 == 0) and (t < src_ids.shape[0] * 2):
                inp = self.embed_src(src_ids[t // 2])
            else:
                inp = out.unsqueeze(0)
                # inp = self.embed_trg(trg_ids[t]) ,原来是这个的，但是这个是预测的，所以要使用out

        return outputs  # [T, 1, emb_dim]
