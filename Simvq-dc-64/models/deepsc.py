import torch
import torch.nn as nn
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer
from .channel import FiniteBlocklengthChannel
from config import Config


class DeepSC(nn.Module):
    """
    单层 U-Net + 纯 SRC 支路 + SimVQ
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_downsample_blocks,
                 base_channels,
                 num_embeddings_list,
                 embedding_dim_list,
                 commitment_cost,
                 device
                 ):
        super(DeepSC, self).__init__()
        self.semantic_encoder = SemanticEncoder(in_channels, num_downsample_blocks, base_channels)
        self.semantic_decoder = SemanticDecoder(embedding_dim_list, out_channels)
        self.vector_quantizers = nn.ModuleList()
        for i in range(num_downsample_blocks):
            self.vector_quantizers.append(VectorQuantizer(
                num_embeddings_list[i],
                embedding_dim_list[i],
                commitment_cost
            ))

        self.device = device
        self.num_embeddings_list = num_embeddings_list
        self.embedding_dim_list = embedding_dim_list

        self.channel = FiniteBlocklengthChannel(
            channel_coding_rate=Config.CHANNEL_CODING_RATE_TRAIN,
            coded_block_length_bits=Config.BLOCK_LENGTH,
            device=self.device
        )

    def _sample_mod_bits(self, snr_db):
        if snr_db < 4.0:
            return random.choice([1, 2])
        elif snr_db < 8.0:
            return random.choice([1, 2, 4])
        else:
            return random.choice([2, 4])

    def forward_train(self, x):
        snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = Config.CHANNEL_CODING_RATE_TRAIN

        encoder_features = self.semantic_encoder(x)

        quantized_corrupted = []
        vq_losses = []

        for i, feat in enumerate(encoder_features):
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)

            corrupted_idx, _ = self.channel.apply_channel_noise(
                encoding_idx,
                self.num_embeddings_list[i],
                snr_tensor,
                current_rc,
                mod_bits=current_mod_bits
            )

            quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)

            quantized_final = quantized_clean + (quantized_noisy - quantized_clean).detach()
            quantized_corrupted.append(quantized_final)

        reconstructed_images = self.semantic_decoder(quantized_corrupted)

        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "current_snr": snr_db,
        }

    def forward_val(self, x):
        snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = Config.CHANNEL_CODING_RATE_VAL

        encoder_features = self.semantic_encoder(x)

        quantized_corrupted = []
        vq_losses = []

        for i, feat in enumerate(encoder_features):
            vq_loss, _, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)

            corrupted_idx, _ = self.channel.apply_channel_noise(
                encoding_idx,
                self.num_embeddings_list[i],
                snr_tensor,
                current_rc,
                mod_bits=current_mod_bits
            )

            quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)
            quantized_corrupted.append(quantized_noisy)

        reconstructed_images = self.semantic_decoder(quantized_corrupted)

        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "current_snr": snr_db,
        }

    def forward_test(self, x):
        encoder_features = self.semantic_encoder(x)
        indices_list = []
        for i, feat in enumerate(encoder_features):
            _, _, encoding_idx = self.vector_quantizers[i](feat)
            indices_list.append(encoding_idx)
        return {"indices": indices_list}

    def reconstruct_from_indices(self, all_encoding_indices):
        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices):
            quantized = self.vector_quantizers[i].get_quantized_features(encoding_indices)
            quantized_features.append(quantized)
        reconstructed_image = self.semantic_decoder(quantized_features)
        return reconstructed_image
