import torch
import torch.nn as nn

from .semantic_decoder_last2_cascade import SemanticDecoderLast2Cascade
from .semantic_encoder import SemanticEncoder
from .vector_quantizer import VectorQuantizer


class DeepSCLast2Cascade(nn.Module):
    """Four-stage image autoencoder with VQ only at physical scales 2 and 3."""

    MODEL_TYPE = "last2_copy_cascade"
    ACTIVE_SCALES = (2, 3)

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_downsample_blocks: int,
        base_channels: int,
        full_num_embeddings_list,
        full_embedding_dim_list,
        commitment_cost: float,
    ):
        super().__init__()
        if num_downsample_blocks != 4:
            raise ValueError("Last-two cascade model requires four encoder stages")
        if len(full_num_embeddings_list) != 4:
            raise ValueError("Full codebook label must contain four values")
        if len(full_embedding_dim_list) != 4:
            raise ValueError("Full embedding dimension list must contain four values")

        self.full_num_embeddings_list = list(full_num_embeddings_list)
        self.full_embedding_dim_list = list(full_embedding_dim_list)
        self.num_embeddings_list = [
            self.full_num_embeddings_list[scale] for scale in self.ACTIVE_SCALES
        ]
        self.embedding_dim_list = [
            self.full_embedding_dim_list[scale] for scale in self.ACTIVE_SCALES
        ]

        self.semantic_encoder = SemanticEncoder(
            in_channels, num_downsample_blocks, base_channels
        )
        self.semantic_decoder = SemanticDecoderLast2Cascade(
            self.full_embedding_dim_list, out_channels
        )
        self.vector_quantizers = nn.ModuleList(
            [
                VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
                for num_embeddings, embedding_dim in zip(
                    self.num_embeddings_list, self.embedding_dim_list
                )
            ]
        )

    def encode(self, images):
        encoder_features = self.semantic_encoder(images)
        quantized_features = []
        encoding_indices = []
        vq_losses = []

        for local_index, physical_scale in enumerate(self.ACTIVE_SCALES):
            vq_loss, quantized, indices = self.vector_quantizers[local_index](
                encoder_features[physical_scale]
            )
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
            "active_scales": list(self.ACTIVE_SCALES),
        }

    def forward_train(self, images):
        return self(images)

    def forward_val(self, images):
        return self(images)

    def forward_test(self, images):
        _, encoding_indices, _ = self.encode(images)
        return {
            "indices": encoding_indices,
            "active_scales": list(self.ACTIVE_SCALES),
        }

    def reconstruct_from_indices(self, encoding_indices):
        if len(encoding_indices) != len(self.vector_quantizers):
            raise ValueError(
                f"Expected two index tensors for scales {self.ACTIVE_SCALES}, "
                f"got {len(encoding_indices)}"
            )
        quantized_features = [
            quantizer.get_quantized_features(indices)
            for quantizer, indices in zip(
                self.vector_quantizers, encoding_indices
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
            for local_index, physical_scale in enumerate(self.ACTIVE_SCALES):
                _, _, indices = self.vector_quantizers[local_index](
                    encoder_features[physical_scale]
                )
                all_indices[local_index].append(indices.cpu())

        if was_training:
            self.train()

        results = {"src": []}
        for local_index, quantizer in enumerate(self.vector_quantizers):
            if not all_indices[local_index]:
                raise ValueError("Cannot compute codebook utilization from an empty dataloader")
            combined_indices = torch.cat(all_indices[local_index], dim=0)
            stats = VectorQuantizer.compute_codebook_stats(
                combined_indices, self.num_embeddings_list[local_index]
            )
            stats.update(
                VectorQuantizer.compute_min_l2_distance(quantizer.embedding.weight)
            )
            stats["scale"] = self.ACTIVE_SCALES[local_index]
            results["src"].append(stats)
        return results

    def print_codebook_utilization(self, results):
        print("\n" + "=" * 80)
        print("Last-Two Codebook Utilization Report")
        print("=" * 80)
        for local_index, stats in enumerate(results["src"]):
            physical_scale = stats["scale"]
            codebook_size = self.num_embeddings_list[local_index]
            print(
                f"Scale {physical_scale} (K={codebook_size}) | "
                f"active {stats['active_count']}/{codebook_size} "
                f"({stats['active_ratio']:.2%}) | "
                f"perplexity {stats['perplexity']:.1f} | "
                f"min L2 {stats['min_l2_dist']:.4f} | "
                f"collapsed {stats['collapse_count']} "
                f"({stats['collapse_ratio']:.2%})"
            )
        print("=" * 80 + "\n")
