import torch
import torch.nn as nn
import os
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import ChannelwiseVectorQuantizer, VanillaVectorQuantizer, VectorQuantizer
from .channel import FiniteBlocklengthChannel
from .attention import BottleneckAttentionStack
from .swinir_enhance import SwinIREnhance


class DeepSC(nn.Module):
    """
    Configurable multi-layer U-Net + SimVQ.
    Supports optional SwinIR quality enhancement post-processing.
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
                 use_cascade_downsample=True,
                 use_bottleneck_attention=False,
                 bottleneck_attention_blocks=1,
                 use_swinir_enhance=False,
                 swinir_enhance_blocks=4,
                 quantizer_type="simvq",
                 quantizer_axis_list=None,
                 cvq_codeword_shapes=None,
                 nested_channel_dropout_alpha=0.0,
                 vitvq_qbridge_type="QBridgeNoCompress-S",
                 vitvq_emb_nograd=False,
                 ):
        super(DeepSC, self).__init__()
        if len(num_embeddings_list) != num_downsample_blocks:
            raise ValueError("num_embeddings_list length must match num_downsample_blocks")
        if len(embedding_dim_list) != num_downsample_blocks:
            raise ValueError("embedding_dim_list length must match num_downsample_blocks")
        if strides is not None and len(strides) != num_downsample_blocks:
            raise ValueError("strides length must match num_downsample_blocks")
        if quantizer_axis_list is None:
            quantizer_axis_list = ["patch"] * num_downsample_blocks
        if len(quantizer_axis_list) != num_downsample_blocks:
            raise ValueError("quantizer_axis_list length must match num_downsample_blocks")
        if cvq_codeword_shapes is None:
            cvq_codeword_shapes = [None] * num_downsample_blocks
        if len(cvq_codeword_shapes) != num_downsample_blocks:
            raise ValueError("cvq_codeword_shapes length must match num_downsample_blocks")

        self.semantic_encoder = SemanticEncoder(
            in_channels, num_downsample_blocks, base_channels, strides=strides,
            norm_type=norm_type,
            num_groups=norm_groups,
            activation=activation,
            num_res_blocks=encoder_res_blocks,
            use_cascade_downsample=use_cascade_downsample,
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
        self.quantizer_type = quantizer_type
        self.quantizer_axis_list = list(quantizer_axis_list)
        self.nested_channel_dropout_alpha = float(nested_channel_dropout_alpha)
        self.vector_quantizers = nn.ModuleList()
        for i in range(num_downsample_blocks):
            if quantizer_type == "none":
                continue
            elif quantizer_type == "simvq" and self.quantizer_axis_list[i] == "channel":
                quantizer = ChannelwiseVectorQuantizer(
                    num_embeddings_list[i], cvq_codeword_shapes[i], commitment_cost
                )
            elif quantizer_type == "simvq":
                quantizer = VectorQuantizer(
                    num_embeddings_list[i], embedding_dim_list[i], commitment_cost
                )
            elif quantizer_type == "vq":
                quantizer = VanillaVectorQuantizer(
                    num_embeddings_list[i], embedding_dim_list[i], commitment_cost
                )
            elif quantizer_type == "vitvq_nocompress":
                from .vector_quantizer_vitvq import ViTvqNoCompressVectorQuantizer

                quantizer = ViTvqNoCompressVectorQuantizer(
                    num_embeddings_list[i],
                    embedding_dim_list[i],
                    commitment_cost,
                    qbridge_type=vitvq_qbridge_type,
                    emb_nograd=vitvq_emb_nograd,
                )
            else:
                raise ValueError(f"Unknown quantizer_type={quantizer_type!r}")
            self.vector_quantizers.append(quantizer)

        self.device = device
        self.encoder_device = device
        self.decoder_device = device
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

        # 可选 SwinIR 质量增强后处理
        if use_swinir_enhance:
            self.swinir_enhance = SwinIREnhance(
                embed_dim=48, num_rstb=swinir_enhance_blocks,
                window_size=8, num_heads=4)
        else:
            self.swinir_enhance = nn.Identity()

    def _maybe_apply_nested_channel_dropout(self, layer_index, feat):
        if (
            not self.training
            or self.nested_channel_dropout_alpha <= 0
            or self.quantizer_axis_list[layer_index] != "channel"
            or random.random() >= self.nested_channel_dropout_alpha
        ):
            return feat
        c_keep = random.randint(1, feat.shape[1])
        dropped = feat.clone()
        dropped[:, c_keep:, :, :] = 0
        return dropped

    def enable_model_parallel(self, encoder_device, decoder_device):
        self.encoder_device = torch.device(encoder_device)
        self.decoder_device = torch.device(decoder_device)
        self.device = self.encoder_device
        self.semantic_encoder.to(self.encoder_device)
        self.bottleneck_attention.to(self.encoder_device)
        self.vector_quantizers.to(self.encoder_device)
        self.channel.to(self.encoder_device)
        self.semantic_decoder.to(self.decoder_device)
        self.swinir_enhance.to(self.decoder_device)
        tail_device = os.environ.get("SIMVQ_DECODER_TAIL_DEVICE", "")
        if tail_device:
            tail_blocks = int(os.environ.get("SIMVQ_DECODER_TAIL_BLOCKS", "1"))
            self.semantic_decoder.set_tail_device(tail_device, tail_blocks=tail_blocks)
        return self

    def _to_encoder_device(self, x):
        if x.device != self.encoder_device:
            return x.to(self.encoder_device, non_blocking=True)
        return x

    def _to_decoder_device(self, features):
        if self.decoder_device == self.encoder_device:
            return features
        return [feat.to(self.decoder_device, non_blocking=True) for feat in features]

    def _sample_mod_bits(self, snr_db):
        if snr_db < 4.0:
            return random.choice([1, 2])
        elif snr_db < 8.0:
            return random.choice([1, 2, 4])
        else:
            return random.choice([2, 4])

    def forward_train(self, x):
        x = self._to_encoder_device(x)
        snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        snr_tensor = torch.tensor(snr_db, device=self.encoder_device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = self.channel_coding_rate_train

        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])

        quantized_corrupted = []
        vq_losses = []
        use_channel = self.quantizer_type != "none" and random.random() < self.channel_prob

        for i, feat in enumerate(encoder_features):
            if self.quantizer_type == "none":
                quantized_corrupted.append(feat)
                vq_losses.append(feat.new_zeros(()))
                continue
            feat = self._maybe_apply_nested_channel_dropout(i, feat)
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
                quantized_noisy = self.vector_quantizers[i].get_quantized_features(
                    corrupted_idx, output_spatial_size=feat.shape[-2:]
                )
                quantized_final = quantized_clean + (quantized_noisy - quantized_clean).detach()
            else:
                quantized_final = quantized_clean
            quantized_corrupted.append(quantized_final)

        quantized_corrupted = self._to_decoder_device(quantized_corrupted)
        reconstructed_images = self.semantic_decoder(quantized_corrupted)
        reconstructed_images = self.swinir_enhance(reconstructed_images)

        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "current_snr": snr_db if use_channel else None,
            "channel_used": use_channel,
            "channel_prob": self.channel_prob,
        }

    def forward_val(self, x):
        x = self._to_encoder_device(x)
        snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        snr_tensor = torch.tensor(snr_db, device=self.encoder_device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = self.channel_coding_rate_val

        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])

        quantized_corrupted = []
        vq_losses = []
        use_channel = self.quantizer_type != "none" and random.random() < self.channel_prob

        for i, feat in enumerate(encoder_features):
            if self.quantizer_type == "none":
                quantized_corrupted.append(feat)
                vq_losses.append(feat.new_zeros(()))
                continue
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
                quantized_final = self.vector_quantizers[i].get_quantized_features(
                    corrupted_idx, output_spatial_size=feat.shape[-2:]
                )
            else:
                quantized_final = quantized_clean
            quantized_corrupted.append(quantized_final)

        quantized_corrupted = self._to_decoder_device(quantized_corrupted)
        reconstructed_images = self.semantic_decoder(quantized_corrupted)
        reconstructed_images = self.swinir_enhance(reconstructed_images)

        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "current_snr": snr_db if use_channel else None,
            "channel_used": use_channel,
            "channel_prob": self.channel_prob,
        }

    def forward_test(self, x):
        x = self._to_encoder_device(x)
        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])
        if self.quantizer_type == "none":
            return {"indices": encoder_features}
        indices_list = []
        feature_shapes = []
        for i, feat in enumerate(encoder_features):
            _, _, encoding_idx = self.vector_quantizers[i](feat)
            indices_list.append(encoding_idx)
            feature_shapes.append(tuple(feat.shape[-2:]))
        return {"indices": indices_list, "feature_shapes": feature_shapes}

    def reconstruct_from_indices(self, all_encoding_indices, feature_shapes=None):
        if self.quantizer_type == "none":
            all_encoding_indices = self._to_decoder_device(all_encoding_indices)
            reconstructed_image = self.semantic_decoder(all_encoding_indices)
            return self.swinir_enhance(reconstructed_image)
        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices):
            encoding_indices = encoding_indices.to(self.encoder_device, non_blocking=True)
            output_spatial_size = feature_shapes[i] if feature_shapes is not None else None
            quantized = self.vector_quantizers[i].get_quantized_features(
                encoding_indices, output_spatial_size=output_spatial_size
            )
            quantized_features.append(quantized)
        quantized_features = self._to_decoder_device(quantized_features)
        reconstructed_image = self.semantic_decoder(quantized_features)
        reconstructed_image = self.swinir_enhance(reconstructed_image)
        return reconstructed_image

    def set_channel_prob(self, channel_prob):
        self.channel_prob = float(max(0.0, min(1.0, channel_prob)))
