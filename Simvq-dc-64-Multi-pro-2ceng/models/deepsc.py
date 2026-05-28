import torch
import torch.nn as nn
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer
from .channel import FiniteBlocklengthChannel
from .attention import BottleneckAttentionStack


class DeepSC(nn.Module):
    """
    Configurable multi-layer U-Net + SimVQ.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_downsample_blocks,
                 base_channels,
                 num_embeddings_list,
                 embedding_dim_list,
                 commitment_cost,
                 device,
                 strides=None,
                 skip_dropout_p=None,
                 channel_coding_rate_train=0.5,
                 channel_coding_rate_val=0.5,
                 block_length=256,
                 snr_range_db=None,
                 norm_type="batch",
                 norm_groups=32,
                 activation="prelu",
                 encoder_res_blocks=1,
                 decoder_res_blocks=1,
                 upsample_mode="nearest",
                 use_bottleneck_attention=False,
                 bottleneck_attention_blocks=1,
                 ):
        super(DeepSC, self).__init__()
        if len(num_embeddings_list) != num_downsample_blocks:
            raise ValueError("num_embeddings_list length must match num_downsample_blocks")
        if len(embedding_dim_list) != num_downsample_blocks:
            raise ValueError("embedding_dim_list length must match num_downsample_blocks")
        if strides is not None and len(strides) != num_downsample_blocks:
            raise ValueError("strides length must match num_downsample_blocks")

        self.semantic_encoder = SemanticEncoder(
            in_channels, num_downsample_blocks, base_channels, strides=strides,
            norm_type=norm_type,
            num_groups=norm_groups,
            activation=activation,
            num_res_blocks=encoder_res_blocks,
        )
        if strides is not None:
            upsample_scales = list(reversed(strides))
        else:
            upsample_scales = None
        self.semantic_decoder = SemanticDecoder(
            embedding_dim_list, out_channels,
            up_mode=upsample_mode,
            skip_dropout_p=skip_dropout_p,
            upsample_scales=upsample_scales,
            norm_type=norm_type,
            num_groups=norm_groups,
            activation=activation,
            num_res_blocks=decoder_res_blocks,
        )
        if use_bottleneck_attention:
            self.bottleneck_attention = BottleneckAttentionStack(
                embedding_dim_list[-1],
                num_blocks=bottleneck_attention_blocks,
                num_groups=norm_groups,
            )
        else:
            self.bottleneck_attention = nn.Identity()
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
        self.channel_coding_rate_train = channel_coding_rate_train
        self.channel_coding_rate_val = channel_coding_rate_val
        self.snr_range_db = snr_range_db or [0, 15]
        self.channel_prob = 1.0

        self.channel = FiniteBlocklengthChannel(
            channel_coding_rate=channel_coding_rate_train,
            coded_block_length_bits=block_length,
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
        snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = self.channel_coding_rate_train

        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])

        quantized_corrupted = []
        vq_losses = []
        use_channel = random.random() < self.channel_prob

        for i, feat in enumerate(encoder_features):
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)

            if use_channel:
                corrupted_idx, _ = self.channel.apply_channel_noise(
                    encoding_idx,
                    self.num_embeddings_list[i],
                    snr_tensor,
                    current_rc,
                    mod_bits=current_mod_bits
                )
                quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)
                quantized_final = quantized_clean + (quantized_noisy - quantized_clean).detach()
            else:
                quantized_final = quantized_clean
            quantized_corrupted.append(quantized_final)

        reconstructed_images = self.semantic_decoder(quantized_corrupted)

        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "current_snr": snr_db if use_channel else None,
            "channel_used": use_channel,
            "channel_prob": self.channel_prob,
        }

    def forward_val(self, x):
        snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = self.channel_coding_rate_val

        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])

        quantized_corrupted = []
        vq_losses = []
        use_channel = random.random() < self.channel_prob

        for i, feat in enumerate(encoder_features):
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)

            if use_channel:
                corrupted_idx, _ = self.channel.apply_channel_noise(
                    encoding_idx,
                    self.num_embeddings_list[i],
                    snr_tensor,
                    current_rc,
                    mod_bits=current_mod_bits
                )
                quantized_final = self.vector_quantizers[i].get_quantized_features(corrupted_idx)
            else:
                quantized_final = quantized_clean
            quantized_corrupted.append(quantized_final)

        reconstructed_images = self.semantic_decoder(quantized_corrupted)

        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "current_snr": snr_db if use_channel else None,
            "channel_used": use_channel,
            "channel_prob": self.channel_prob,
        }

    def forward_test(self, x):
        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])
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

    def set_channel_prob(self, channel_prob):
        self.channel_prob = float(max(0.0, min(1.0, channel_prob)))
