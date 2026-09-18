import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Standard trainable-embedding vector quantizer with a straight-through estimator."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, inputs: torch.Tensor):
        inputs_bhwc = inputs.permute(0, 2, 3, 1).contiguous()
        batch_size, height, width, channels = inputs_bhwc.shape
        flat_inputs = inputs_bhwc.view(-1, channels)
        embedding_weight = self.embedding.weight

        distances = (
            torch.sum(flat_inputs**2, dim=1, keepdim=True)
            + torch.sum(embedding_weight**2, dim=1)
            - 2 * torch.einsum("bd,nd->bn", flat_inputs, embedding_weight)
        )
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = F.embedding(encoding_indices, embedding_weight)
        quantized = quantized.view(batch_size, height, width, channels)

        encoder_loss = F.mse_loss(quantized.detach(), inputs_bhwc)
        codebook_loss = F.mse_loss(quantized, inputs_bhwc.detach())
        vq_loss = codebook_loss + self.commitment_cost * encoder_loss

        quantized = inputs_bhwc + (quantized - inputs_bhwc).detach()
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        return vq_loss, quantized, encoding_indices.view(batch_size, height, width)

    @torch.no_grad()
    def get_quantized_features(self, encoding_indices: torch.Tensor) -> torch.Tensor:
        if encoding_indices.dim() == 2:
            encoding_indices = encoding_indices.unsqueeze(0)
        batch_size, height, width = encoding_indices.shape
        quantized = F.embedding(encoding_indices.reshape(-1), self.embedding.weight)
        quantized = quantized.view(batch_size, height, width, self.embedding_dim)
        return quantized.permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def compute_codebook_stats(encoding_indices: torch.Tensor, num_embeddings: int):
        flat_indices = encoding_indices.reshape(-1)
        usage_counts = torch.bincount(flat_indices, minlength=num_embeddings).float()
        active_count = int((usage_counts > 0).sum().item())
        total = flat_indices.numel()

        if total:
            probabilities = usage_counts / total
            probabilities = probabilities[probabilities > 0]
            entropy = -torch.sum(probabilities * torch.log2(probabilities))
            perplexity = float(torch.pow(2.0, entropy).item())
        else:
            perplexity = 1.0

        return {
            "active_ratio": active_count / num_embeddings,
            "perplexity": perplexity,
            "active_count": active_count,
            "dead_count": num_embeddings - active_count,
            "usage_counts": usage_counts,
        }

    @staticmethod
    def compute_min_l2_distance(embedding_weight: torch.Tensor, collapse_threshold: float = 0.1):
        norm_squared = torch.sum(embedding_weight**2, dim=1)
        distance_squared = (
            norm_squared.unsqueeze(1)
            + norm_squared.unsqueeze(0)
            - 2 * embedding_weight @ embedding_weight.t()
        )
        codebook_size = distance_squared.size(0)
        diagonal = torch.eye(codebook_size, device=distance_squared.device, dtype=torch.bool)
        distance_squared = distance_squared.masked_fill(diagonal, float("inf"))

        nearest_distances = torch.sqrt(distance_squared.min(dim=1).values.clamp_min(0))
        collapse_count = int((nearest_distances < collapse_threshold).sum().item())
        return {
            "min_l2_dist": float(nearest_distances.min().item()),
            "collapse_count": collapse_count,
            "collapse_ratio": collapse_count / codebook_size if codebook_size else 0.0,
        }
