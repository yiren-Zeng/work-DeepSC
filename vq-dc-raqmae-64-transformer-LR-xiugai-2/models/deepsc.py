import torch
import torch.nn as nn
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer


class DeepSC(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_downsample_blocks,
        base_channels,
        num_embeddings_list,
        embedding_dim_list,
        commitment_cost,
        device,
    ):
        super(DeepSC, self).__init__()
        self.semantic_encoder = SemanticEncoder(in_channels, num_downsample_blocks, base_channels)
        self.semantic_decoder = SemanticDecoder(embedding_dim_list, out_channels)
        self.vector_quantizers = nn.ModuleList()
        for i in range(num_downsample_blocks):
            self.vector_quantizers.append(
                VectorQuantizer(
                    num_embeddings_list[i],
                    embedding_dim_list[i],
                    commitment_cost,
                )
            )

        self.device = device
        self.num_embeddings_list = num_embeddings_list
        self.embedding_dim_list = embedding_dim_list

    def forward_train(self, x):
        """单支路 VQ：无量化索引信道扰动，标准 VQ-VAE 式重建。"""
        encoder_features = self.semantic_encoder(x)
        quantized_list = []
        vq_losses = []
        indices_list = []

        for i, feat in enumerate(encoder_features):
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)
            indices_list.append(encoding_idx)
            quantized_list.append(quantized_clean)

        reconstructed_images = self.semantic_decoder(quantized_list)
        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "indices": indices_list,
        }

    def forward_val(self, x):
        """验证：与训练相同（无信道）。"""
        return self.forward_train(x)

    def forward_test(self, x):
        """测试：仅信源编码与 VQ 索引（供物理层链路评估脚本使用）。"""
        encoder_features = self.semantic_encoder(x)
        indices_src = []
        for i, feat in enumerate(encoder_features):
            _, _, encoding_idx = self.vector_quantizers[i](feat)
            indices_src.append(encoding_idx)
        return {"indices_src": indices_src}

    def reconstruct_from_indices(self, all_encoding_indices, codebooks=None):
        """
        Args:
            all_encoding_indices: 各层的编码索引列表
            codebooks: 若为 None，则使用各层 VQ 的投影源码本
        """
        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices):
            raq_weight = codebooks[i] if codebooks is not None else None
            quantized = self.vector_quantizers[i].get_quantized_features(
                encoding_indices, raq_weight=raq_weight
            )
            quantized_features.append(quantized)
        reconstructed_image = self.semantic_decoder(quantized_features)
        return reconstructed_image
