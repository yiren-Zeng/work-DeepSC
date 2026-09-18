import torch
import torch.nn as nn

from .semantic_decoder import SemanticDecoder
from .semantic_encoder import SemanticEncoder
from .vector_quantizer import VectorQuantizer


class DeepSC(nn.Module):
    """Channel-free, four-scale VQ image autoencoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_downsample_blocks: int,
        base_channels: int,
        num_embeddings_list,
        embedding_dim_list,
        commitment_cost: float,
    ):
        super().__init__()
        if not (
            num_downsample_blocks
            == len(num_embeddings_list)
            == len(embedding_dim_list)
        ):
            raise ValueError("The number of scales and VQ configuration lengths must match")

        self.semantic_encoder = SemanticEncoder(
            in_channels, num_downsample_blocks, base_channels
        )
        self.semantic_decoder = SemanticDecoder(embedding_dim_list, out_channels)
        self.vector_quantizers = nn.ModuleList(
            [
                VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
                for num_embeddings, embedding_dim in zip(
                    num_embeddings_list, embedding_dim_list
                )
            ]
        )
        self.num_embeddings_list = list(num_embeddings_list)
        self.embedding_dim_list = list(embedding_dim_list)

    def encode(self, images):
        encoder_features = self.semantic_encoder(images)
        quantized_features = []
        encoding_indices = []
        vq_losses = []

        for feature, quantizer in zip(encoder_features, self.vector_quantizers):
            vq_loss, quantized, indices = quantizer(feature)
            quantized_features.append(quantized)
            encoding_indices.append(indices)
            vq_losses.append(vq_loss)

        return quantized_features, encoding_indices, vq_losses

    def forward(self, images):
        quantized_features, encoding_indices, vq_losses = self.encode(images)
        reconstructed_images = self.semantic_decoder(quantized_features)
        return {
            "reconstructed_images": reconstructed_images,
            "indices": encoding_indices,
            "vq_losses": vq_losses,
        }

    def forward_train(self, images):
        return self(images)

    def forward_val(self, images):
        return self(images)

    def forward_test(self, images):
        _, encoding_indices, _ = self.encode(images)
        return {"indices": encoding_indices}

    def reconstruct_from_indices(self, all_encoding_indices):
        if len(all_encoding_indices) != len(self.vector_quantizers):
            raise ValueError(
                f"Expected {len(self.vector_quantizers)} index scales, "
                f"got {len(all_encoding_indices)}"
            )
        quantized_features = [
            quantizer.get_quantized_features(indices)
            for quantizer, indices in zip(
                self.vector_quantizers, all_encoding_indices
            )
        ]
        return self.semantic_decoder(quantized_features)

    @torch.no_grad()
    def compute_codebook_utilization(self, dataloader, max_batches=None, device=None):
        if device is None:
            device = next(self.parameters()).device

        was_training = self.training
        self.eval()
        all_indices = [[] for _ in self.vector_quantizers]

        for batch_index, images in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            encoder_features = self.semantic_encoder(images)
            for scale, (feature, quantizer) in enumerate(
                zip(encoder_features, self.vector_quantizers)
            ):
                _, _, indices = quantizer(feature)
                all_indices[scale].append(indices.cpu())

        if was_training:
            self.train()

        results = {"src": []}
        for scale, quantizer in enumerate(self.vector_quantizers):
            if not all_indices[scale]:
                raise ValueError("Cannot compute codebook utilization from an empty dataloader")
            combined_indices = torch.cat(all_indices[scale], dim=0)
            stats = VectorQuantizer.compute_codebook_stats(
                combined_indices, self.num_embeddings_list[scale]
            )
            stats.update(
                VectorQuantizer.compute_min_l2_distance(quantizer.embedding.weight)
            )
            results["src"].append(stats)
        return results

    @staticmethod
    def print_codebook_utilization(results, num_embeddings_list=None):
        print("\n" + "=" * 80)
        print("Codebook Utilization Report")
        print("=" * 80)
        for scale, stats in enumerate(results["src"]):
            codebook_size = (
                num_embeddings_list[scale]
                if num_embeddings_list is not None
                else stats["active_count"] + stats["dead_count"]
            )
            print(
                f"Layer {scale} (K={codebook_size}) | "
                f"active {stats['active_count']}/{codebook_size} "
                f"({stats['active_ratio']:.2%}) | "
                f"perplexity {stats['perplexity']:.1f} | "
                f"min L2 {stats['min_l2_dist']:.4f} | "
                f"collapsed {stats['collapse_count']} "
                f"({stats['collapse_ratio']:.2%})"
            )
        print("=" * 80 + "\n")
