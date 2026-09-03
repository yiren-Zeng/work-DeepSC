import torch
import torch.nn as nn
import math
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer
from .channel import FiniteBlocklengthChannel
from .raq import RAQ
from .independent_raq_rvq import quantize_independent_raq_rvq
from utils.raq_rvq import validate_independent_rvq_k_lists


class DeepSC(nn.Module):
    """
    Dedicated two-scale, two-stage independent RAQ-RVQ model.
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
                 raq_min_trg=None,
                 raq_max_trg=None,
                 independent_raq_rvq_depth=2,
                 independent_raq_rvq_k_lists=None,
                 raq_min_trg_list=None,
                 raq_max_trg_list=None,
                 ):
        super(DeepSC, self).__init__()
        if len(num_embeddings_list) != num_downsample_blocks:
            raise ValueError("num_embeddings_list length must match num_downsample_blocks")
        if len(embedding_dim_list) != num_downsample_blocks:
            raise ValueError("embedding_dim_list length must match num_downsample_blocks")
        if strides is not None and len(strides) != num_downsample_blocks:
            raise ValueError("strides length must match num_downsample_blocks")
        # 把输入图像 x 编码成多层特征 encoder_features
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

        # 把量化后的多层特征重新解码成图像
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
        self.vector_quantizers = nn.ModuleList()
        for i in range(num_downsample_blocks):
            self.vector_quantizers.append(VectorQuantizer(
                num_embeddings_list[i],
                embedding_dim_list[i],
                commitment_cost,
            ))

        self.independent_raq_rvq_depth = int(independent_raq_rvq_depth)
        self.independent_raq_rvq_k_lists = (
            [list(stage_sizes) for stage_sizes in independent_raq_rvq_k_lists]
            if independent_raq_rvq_k_lists is not None else None
        )
        if self.independent_raq_rvq_depth != 2:
            raise ValueError("independent RAQ-RVQ currently requires depth=2")
        def expand_raq_bound(values, scalar, name):
            if values is None:
                if scalar is None:
                    return None
                return [int(scalar)] * num_downsample_blocks
            expanded = [int(value) for value in values]
            if len(expanded) != num_downsample_blocks:
                raise ValueError(
                    f"{name} length must match num_downsample_blocks"
                )
            return expanded

        self.raq_min_trg_list = expand_raq_bound(
            raq_min_trg_list, raq_min_trg, "raq_min_trg_list"
        )
        self.raq_max_trg_list = expand_raq_bound(
            raq_max_trg_list, raq_max_trg, "raq_max_trg_list"
        )
        self.raq_min_trg = (
            int(raq_min_trg)
            if raq_min_trg is not None
            else (
                min(self.raq_min_trg_list)
                if self.raq_min_trg_list is not None else None
            )
        )
        self.raq_max_trg = (
            int(raq_max_trg)
            if raq_max_trg is not None
            else (
                max(self.raq_max_trg_list)
                if self.raq_max_trg_list is not None else None
            )
        )
        self.raqs = nn.ModuleList()
        self.raqs_rvq_stage2 = nn.ModuleList()
        if self.raq_min_trg_list is None or self.raq_max_trg_list is None:
            raise ValueError(
                "RAQ enabled but target min/max bounds are not configured."
            )
        for layer_index, (min_k, max_k) in enumerate(
            zip(self.raq_min_trg_list, self.raq_max_trg_list)
        ):
            if min_k < 2 or min_k > max_k:
                raise ValueError(
                    f"RAQ layer {layer_index} target range must satisfy "
                    f"2 <= min <= max, got [{min_k},{max_k}]"
                )
            if min_k & (min_k - 1) != 0 or max_k & (max_k - 1) != 0:
                raise ValueError(
                    f"RAQ layer {layer_index} target bounds must be powers "
                    f"of two, got [{min_k},{max_k}]"
                )
        self.independent_raq_rvq_k_lists = (
            validate_independent_rvq_k_lists(
                self.independent_raq_rvq_k_lists,
                num_scales=num_downsample_blocks,
                rvq_depth=self.independent_raq_rvq_depth,
                min_k=self.raq_min_trg_list,
                max_k=self.raq_max_trg_list,
            )
        )
        for Ki, Di, min_k, max_k in zip(
            num_embeddings_list,
            embedding_dim_list,
            self.raq_min_trg_list,
            self.raq_max_trg_list,
        ):
            self.raqs.append(
                RAQ(
                    embedding_dim=Di,
                    n_embed_src=Ki,
                    n_embed_min_trg=min_k,
                    n_embed_max_trg=max_k,
                    device=device,
                )
            )
            self.raqs_rvq_stage2.append(
                RAQ(
                    embedding_dim=Di,
                    n_embed_src=Ki,
                    n_embed_min_trg=min_k,
                    n_embed_max_trg=max_k,
                    device=device,
                )
            )

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

    def _select_source_quantizer(self, layer_index):
        return (
            self.vector_quantizers[layer_index],
            int(self.num_embeddings_list[layer_index]),
            "src",
        )

    def _decode_features(self, quantized_features):
        quantized_features = self._to_decoder_device(quantized_features)
        return self.semantic_decoder(quantized_features)




    def _forward_independent_raq_rvq(
        self,
        encoder_features,
        rvq_k_lists,
        use_channel,
        snr_tensor,
        current_rc,
        current_mod_bits,
        ste_channel,
    ):
        """Train four independently generated scale/stage RAQ codebooks."""
        rvq_k_lists = validate_independent_rvq_k_lists(
            rvq_k_lists,
            num_scales=len(encoder_features),
            rvq_depth=self.independent_raq_rvq_depth,
            min_k=self.raq_min_trg_list,
            max_k=self.raq_max_trg_list,
        )
        quantized_raq = []
        vq_losses_raq = []

        for scale_index, (feat, stage_k_list) in enumerate(
            zip(encoder_features, rvq_k_lists)
        ):
            source_quantizer, _, _ = self._select_source_quantizer(scale_index)
            source_codebook = source_quantizer.transformed_weight()
            stage_generators = (
                self.raqs[scale_index],
                self.raqs_rvq_stage2[scale_index],
            )
            stage_codebooks = [
                generator.generate_codebook_transformer(
                    int(stage_k), source_codebook
                )
                for generator, stage_k in zip(
                    stage_generators, stage_k_list
                )
            ]
            rvq = quantize_independent_raq_rvq(
                source_quantizer,
                feat,
                stage_codebooks,
            )
            clean_sum_raw = rvq["quantized_raw"]
            if use_channel:
                noisy_sum_raw = torch.zeros_like(clean_sum_raw)
                for stage_index, (
                    stage_indices,
                    stage_k,
                    stage_codebook,
                ) in enumerate(zip(
                    rvq["indices"],
                    stage_k_list,
                    stage_codebooks,
                )):
                    corrupted_indices, _ = self.channel.apply_channel_noise(
                        stage_indices,
                        int(stage_k),
                        snr_tensor,
                        current_rc,
                        mod_bits=current_mod_bits,
                    )
                    noisy_sum_raw = noisy_sum_raw + (
                        source_quantizer.get_quantized_features(
                            corrupted_indices,
                            output_spatial_size=feat.shape[-2:],
                            codebook_weight=stage_codebook,
                        )
                    )
                quantized_final = (
                    rvq["quantized"]
                    + (noisy_sum_raw - clean_sum_raw).detach()
                    if ste_channel
                    else noisy_sum_raw
                )
            else:
                quantized_final = rvq["quantized"]

            quantized_raq.append(quantized_final)
            vq_losses_raq.append(rvq["loss"])

        reconstructed_images_raq = self._decode_features(quantized_raq)
        return {
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "rvq_k_lists": [
                list(stage_sizes) for stage_sizes in rvq_k_lists
            ],
        }


    def _forward_impl(
        self,
        x,
        channel_coding_rate,
        ste_channel=False,
        raq_rvq_k_lists=None,
    ):
        x = self._to_encoder_device(x)
        snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        snr_tensor = torch.tensor(snr_db, device=self.encoder_device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = channel_coding_rate

        encoder_features = self.semantic_encoder(x)
        independent_k_lists = validate_independent_rvq_k_lists(
            (
                raq_rvq_k_lists
                if raq_rvq_k_lists is not None
                else self.independent_raq_rvq_k_lists
            ),
            num_scales=len(encoder_features),
            rvq_depth=self.independent_raq_rvq_depth,
            min_k=self.raq_min_trg_list,
            max_k=self.raq_max_trg_list,
        )
        quantized_src = []
        vq_losses_src = []
        use_channel = random.random() < self.channel_prob

        for i, feat in enumerate(encoder_features):
            source_quantizer, source_k, _ = (
                self._select_source_quantizer(i)
            )
            vq_loss, quantized_clean, encoding_idx = source_quantizer(feat)
            vq_losses_src.append(vq_loss)

            if use_channel:
                corrupted_idx, _ = self.channel.apply_channel_noise(
                    encoding_idx,
                    source_k,
                    snr_tensor,
                    current_rc,
                    mod_bits=current_mod_bits,
                )
                quantized_noisy = source_quantizer.get_quantized_features(
                    corrupted_idx, output_spatial_size=feat.shape[-2:]
                )
                quantized_final = (
                    quantized_clean
                    + (quantized_noisy - quantized_clean).detach()
                    if ste_channel
                    else quantized_noisy
                )
            else:
                quantized_final = quantized_clean
            quantized_src.append(quantized_final)

        reconstructed_images_src = self._decode_features(quantized_src)
        result = {
            "reconstructed_images_src": reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "current_snr": snr_db if use_channel else None,
        }

        rvq_result = self._forward_independent_raq_rvq(
            encoder_features,
            independent_k_lists,
            use_channel,
            snr_tensor,
            current_rc,
            current_mod_bits,
            ste_channel,
        )
        result.update(rvq_result)
        return result

    def forward_train(self, x, raq_rvq_k_lists=None):
        return self._forward_impl(
            x,
            channel_coding_rate=self.channel_coding_rate_train,
            ste_channel=True,
            raq_rvq_k_lists=raq_rvq_k_lists,
        )

    def forward_val(self, x, raq_rvq_k_lists=None):
        return self._forward_impl(
            x,
            channel_coding_rate=self.channel_coding_rate_val,
            ste_channel=False,
            raq_rvq_k_lists=raq_rvq_k_lists,
        )


    def _forward_test_independent_raq_rvq(self, encoder_features):
        """Encode with four trained, independently generated RAQ codebooks."""
        rvq_k_lists = validate_independent_rvq_k_lists(
            self.independent_raq_rvq_k_lists,
            num_scales=len(encoder_features),
            rvq_depth=self.independent_raq_rvq_depth,
            min_k=self.raq_min_trg_list,
            max_k=self.raq_max_trg_list,
        )
        indices_by_scale = []
        codebooks_by_scale = []
        feature_shapes = []
        diagnostics = []

        for scale_index, (feat, stage_k_list) in enumerate(
            zip(encoder_features, rvq_k_lists)
        ):
            source_quantizer, _, source_route = self._select_source_quantizer(
                scale_index
            )
            source_codebook = source_quantizer.transformed_weight()
            stage_generators = (
                self.raqs[scale_index],
                self.raqs_rvq_stage2[scale_index],
            )
            stage_codebooks = [
                generator.generate_codebook_transformer(
                    int(stage_k), source_codebook
                )
                for generator, stage_k in zip(
                    stage_generators, stage_k_list
                )
            ]
            rvq = quantize_independent_raq_rvq(
                source_quantizer,
                feat,
                stage_codebooks,
            )
            residual_mse_energies = [
                float(value.item())
                for value in rvq["residual_mse_per_depth"]
            ]
            stage_diagnostics = []
            for stage_index, (
                indices,
                stage_k,
                stage_codebook,
                residual_mse,
            ) in enumerate(zip(
                rvq["indices"],
                stage_k_list,
                stage_codebooks,
                residual_mse_energies,
            )):
                bits_per_index = int(stage_k).bit_length() - 1
                stage_diagnostics.append({
                    "stage_index": stage_index,
                    "k": int(stage_k),
                    "bits_per_index": bits_per_index,
                    "index_min": int(indices.detach().min().item()),
                    "index_max": int(indices.detach().max().item()),
                    "codebook_size": int(stage_codebook.shape[0]),
                    "payload_bits": int(
                        indices.numel() * bits_per_index
                    ),
                    "residual_mse_energy": residual_mse,
                })

            token_count = int(
                feat.shape[0] * feat.shape[-2] * feat.shape[-1]
            )
            expected_payload_bits = token_count * sum(
                int(stage_k).bit_length() - 1
                for stage_k in stage_k_list
            )
            payload_bits = sum(
                stage["payload_bits"] for stage in stage_diagnostics
            )
            diagnostics.append({
                "scale_index": scale_index,
                "source_route": source_route,
                "k_total": math.prod(stage_k_list),
                "stage_k_list": list(stage_k_list),
                "input_mse_energy": float(
                    feat.detach().pow(2).mean().item()
                ),
                "residual_mse_energies": residual_mse_energies,
                "final_residual_mse_energy": residual_mse_energies[-1],
                "quantized_sum_mse_energy": float(
                    rvq["quantized_raw"].detach().pow(2).mean().item()
                ),
                "stage_diagnostics": stage_diagnostics,
                "payload_bits": payload_bits,
                "baseline_payload_bits": expected_payload_bits,
                "bit_budget_matches": (
                    payload_bits == expected_payload_bits
                ),
                "independent_codebook_identity_verified": (
                    stage_codebooks[0] is not stage_codebooks[1]
                ),
            })
            indices_by_scale.append(rvq["indices"])
            codebooks_by_scale.append(stage_codebooks)
            feature_shapes.append(tuple(feat.shape[-2:]))

        return {
            "indices": indices_by_scale,
            "feature_shapes": feature_shapes,
            "num_embeddings_list": [
                list(stage_sizes) for stage_sizes in rvq_k_lists
            ],
            "branch": "independent_raq_rvq",
            "codebooks": codebooks_by_scale,
            "rvq_k_lists": [
                list(stage_sizes) for stage_sizes in rvq_k_lists
            ],
            "independent_raq_rvq_enabled": True,
            "rvq_depth": self.independent_raq_rvq_depth,
            "rvq_diagnostics": diagnostics,
        }


    def forward_test(self, x):
        x = self._to_encoder_device(x)
        encoder_features = self.semantic_encoder(x)
        return self._forward_test_independent_raq_rvq(encoder_features)

    def reconstruct_from_indices(self, all_encoding_indices, feature_shapes=None, codebooks=None):
        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices):
            output_spatial_size = feature_shapes[i] if feature_shapes is not None else None
            if not encoding_indices:
                raise ValueError(f"RVQ scale {i} has no stage indices")
            if codebooks is None or not isinstance(codebooks[i], (list, tuple)):
                raise ValueError("RVQ indices require matching nested codebooks")
            if len(encoding_indices) != len(codebooks[i]):
                raise ValueError(
                    f"RVQ scale {i} has {len(encoding_indices)} index stages "
                    f"but {len(codebooks[i])} codebook stages"
                )
            quantized = None
            for stage_indices, stage_codebook in zip(
                encoding_indices, codebooks[i]
            ):
                stage_indices = stage_indices.to(
                    self.encoder_device, non_blocking=True
                )
                quantized_stage = self.vector_quantizers[
                    i
                ].get_quantized_features(
                    stage_indices,
                    output_spatial_size=output_spatial_size,
                    codebook_weight=stage_codebook,
                )
                quantized = (
                    quantized_stage
                    if quantized is None
                    else quantized + quantized_stage
                )
            quantized_features.append(quantized)
        quantized_features = self._to_decoder_device(quantized_features)
        return self.semantic_decoder(quantized_features)

    def set_channel_prob(self, channel_prob):
        self.channel_prob = float(max(0.0, min(1.0, channel_prob)))
