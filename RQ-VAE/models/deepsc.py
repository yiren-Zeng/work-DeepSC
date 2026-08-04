import torch
import torch.nn as nn
import os
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import ChannelwiseVectorQuantizer, VanillaVectorQuantizer, VectorQuantizer
from .rq_ema_quantizer import RQEMAQuantizer
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
                 rq_depth_list=None,
                 rq_ema_decay=0.99,
                 rq_restart_unused_codes=True,
                 rq_shared_codebook=True,
                 rq_codebook_size_lists=None,
                 ):
        super(DeepSC, self).__init__()
        quantizer_type = str(quantizer_type).lower()
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
        if rq_depth_list is None:
            rq_depth_list = [1] * num_downsample_blocks
        if len(rq_depth_list) != num_downsample_blocks:
            raise ValueError("rq_depth_list length must match num_downsample_blocks")
        if any(int(depth) < 1 for depth in rq_depth_list):
            raise ValueError("rq_depth_list entries must be positive")
        if rq_codebook_size_lists is None:
            rq_codebook_size_lists = [
                [int(size)] * int(depth)
                for size, depth in zip(num_embeddings_list, rq_depth_list)
            ]
        rq_codebook_size_lists = [
            [int(size) for size in scale_sizes]
            for scale_sizes in rq_codebook_size_lists
        ]
        if len(rq_codebook_size_lists) != num_downsample_blocks:
            raise ValueError(
                "rq_codebook_size_lists length must match "
                "num_downsample_blocks"
            )
        for scale, (scale_sizes, depth) in enumerate(
            zip(rq_codebook_size_lists, rq_depth_list)
        ):
            if len(scale_sizes) != int(depth):
                raise ValueError(
                    f"scale {scale} codebook count must equal its RQ depth"
                )
            if any(size < 2 for size in scale_sizes):
                raise ValueError("RQ codebook sizes must be at least 2")
        if (
            quantizer_type in {"rq_ema", "residual_simvq"}
            and not rq_shared_codebook
        ):
            raise ValueError(
                f"{quantizer_type} requires one shared codebook across RQ "
                "depths per scale"
            )
        if (
            quantizer_type == "stagewise_residual_simvq"
            and rq_shared_codebook
        ):
            raise ValueError(
                "stagewise_residual_simvq requires independent codebooks "
                "and rq_shared_codebook=False"
            )


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
        if use_bottleneck_attention:
            self.bottleneck_attention = BottleneckAttentionStack(
                embedding_dim_list[-1],
                num_blocks=bottleneck_attention_blocks,
                num_groups=norm_groups,
            )
        else:
            self.bottleneck_attention = nn.Identity() # nn.Identity() 的意思是什么都不做，输入什么就输出什么
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
            elif quantizer_type == "rq_ema":
                if self.quantizer_axis_list[i] != "patch":
                    raise ValueError("rq_ema only supports direct patch-wise feature quantization")
                quantizer = RQEMAQuantizer(
                    num_embeddings=num_embeddings_list[i],
                    embedding_dim=embedding_dim_list[i],
                    rq_depth=int(rq_depth_list[i]),
                    decay=float(rq_ema_decay),
                    restart_unused_codes=bool(rq_restart_unused_codes),
                    shared_codebook=bool(rq_shared_codebook),
                )
            elif quantizer_type == "residual_simvq":
                if self.quantizer_axis_list[i] != "patch":
                    raise ValueError(
                        "residual_simvq only supports direct patch-wise "
                        "feature quantization"
                    )
                # Keep the optional implementation isolated from legacy model
                # imports; old SimVQ/VQ/RQ-EMA checkpoints do not depend on it.
                from .residual_simvq_quantizer import ResidualSimVQQuantizer

                quantizer = ResidualSimVQQuantizer(
                    num_embeddings=num_embeddings_list[i],
                    embedding_dim=embedding_dim_list[i],
                    rq_depth=int(rq_depth_list[i]),
                    commitment_cost=float(commitment_cost),
                    shared_codebook=bool(rq_shared_codebook),
                )
            elif quantizer_type == "stagewise_residual_simvq":
                if self.quantizer_axis_list[i] != "patch":
                    raise ValueError(
                        "stagewise_residual_simvq only supports direct "
                        "patch-wise feature quantization"
                    )
                from .stagewise_residual_simvq_quantizer import (
                    StagewiseResidualSimVQQuantizer,
                )

                quantizer = StagewiseResidualSimVQQuantizer(
                    num_embeddings_per_depth=rq_codebook_size_lists[i],
                    embedding_dim=embedding_dim_list[i],
                    commitment_cost=float(commitment_cost),
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
        self.rq_depth_list = [int(depth) for depth in rq_depth_list]
        self.rq_codebook_size_lists = [
            list(scale_sizes) for scale_sizes in rq_codebook_size_lists
        ]
        self.rq_ema_decay = float(rq_ema_decay)
        self.rq_restart_unused_codes = bool(rq_restart_unused_codes)
        self.rq_shared_codebook = bool(rq_shared_codebook)
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

    def enable_model_parallel(self, encoder_device, decoder_device): # 用来把模型拆到多张 GPU 上
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

    @staticmethod
    def _quantizer_diagnostics(quantizer):
        """Return detached diagnostics without imposing an interface on legacy VQs."""
        getter = getattr(quantizer, "get_last_diagnostics", None)
        if callable(getter):
            return getter()
        diagnostics = getattr(quantizer, "last_diagnostics", None)
        if callable(diagnostics):
            return diagnostics()
        return diagnostics if isinstance(diagnostics, dict) else {}

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
        quantizer_diagnostics = []
        use_channel = self.quantizer_type != "none" and random.random() < self.channel_prob

        for i, feat in enumerate(encoder_features):
            if self.quantizer_type == "none":
                quantized_corrupted.append(feat)
                vq_losses.append(feat.new_zeros(()))
                quantizer_diagnostics.append({})
                continue
            feat = self._maybe_apply_nested_channel_dropout(i, feat)
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)
            quantizer_diagnostics.append(self._quantizer_diagnostics(self.vector_quantizers[i]))

            if use_channel:
                codebook_sizes = (
                    self.rq_codebook_size_lists[i]
                    if self.quantizer_type == "stagewise_residual_simvq"
                    else self.num_embeddings_list[i]
                )
                corrupted_idx, _ = self.channel.apply_channel_noise(
                    encoding_idx,
                    codebook_sizes,
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
            "mod_bits": current_mod_bits if use_channel else None,
            "quantizer_diagnostics": quantizer_diagnostics,
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
        quantizer_diagnostics = []
        use_channel = self.quantizer_type != "none" and random.random() < self.channel_prob

        for i, feat in enumerate(encoder_features):
            if self.quantizer_type == "none":
                quantized_corrupted.append(feat)
                vq_losses.append(feat.new_zeros(()))
                quantizer_diagnostics.append({})
                continue
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)
            quantizer_diagnostics.append(self._quantizer_diagnostics(self.vector_quantizers[i]))

            if use_channel:
                codebook_sizes = (
                    self.rq_codebook_size_lists[i]
                    if self.quantizer_type == "stagewise_residual_simvq"
                    else self.num_embeddings_list[i]
                )
                corrupted_idx, _ = self.channel.apply_channel_noise(
                    encoding_idx,
                    codebook_sizes,
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
            "mod_bits": current_mod_bits if use_channel else None,
            "quantizer_diagnostics": quantizer_diagnostics,
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

    def _require_adaptive_rq_eval(self):
        if self.training:
            raise RuntimeError(
                "adaptive RQ is eval-only; call model.eval() before using it"
            )
        if self.quantizer_type != "rq_ema":
            raise ValueError(
                "adaptive RQ is only available when quantizer_type='rq_ema'"
            )
        if any(depth != 2 for depth in self.rq_depth_list):
            raise ValueError(
                "adaptive RQ currently requires rq_depth=2 at every scale"
            )

    @staticmethod
    def _adaptive_threshold_list(thresholds, num_scales):
        if thresholds is None:
            return [None] * num_scales
        if torch.is_tensor(thresholds) and thresholds.numel() == 1:
            return [thresholds] * num_scales
        if isinstance(thresholds, (int, float)):
            return [thresholds] * num_scales
        if not isinstance(thresholds, (list, tuple)):
            raise TypeError("thresholds must be a scalar or one value per scale")
        if len(thresholds) != num_scales:
            raise ValueError(
                f"expected {num_scales} adaptive thresholds, got {len(thresholds)}"
            )
        return list(thresholds)

    @staticmethod
    def _adaptive_mask_list(need_second_masks, num_scales):
        if need_second_masks is None:
            return [None] * num_scales
        if num_scales == 1 and torch.is_tensor(need_second_masks):
            return [need_second_masks]
        if not isinstance(need_second_masks, (list, tuple)):
            raise TypeError("need_second_masks must contain one mask per scale")
        if len(need_second_masks) != num_scales:
            raise ValueError(
                f"expected {num_scales} need_second masks, "
                f"got {len(need_second_masks)}"
            )
        return list(need_second_masks)

    @torch.no_grad()
    def forward_test_adaptive(self, x, thresholds=None, need_second_masks=None):
        """Encode eval features with token-wise one- or two-code RQ.

        ``thresholds`` may be one scalar shared by all scales or one threshold
        per scale.  Explicit per-scale boolean masks override threshold-based
        selection.  The legacy :meth:`forward_test` path remains unchanged.
        """
        self._require_adaptive_rq_eval()
        x = self._to_encoder_device(x)
        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])
        num_scales = len(self.vector_quantizers)
        threshold_list = self._adaptive_threshold_list(thresholds, num_scales)
        mask_list = self._adaptive_mask_list(need_second_masks, num_scales)

        indices_list = []
        feature_shapes = []
        normalized_masks = []
        stop_masks = []
        first_stage_errors = []
        final_stage_errors = []
        commitment_losses = []
        adaptive_metadata = []
        for i, feat in enumerate(encoder_features):
            result = self.vector_quantizers[i].forward_adaptive(
                feat,
                threshold_list[i],
                need_second_mask=mask_list[i],
            )
            indices_list.append(result["indices"])
            feature_shapes.append(tuple(feat.shape[-2:]))
            normalized_masks.append(result["need_second_mask"])
            stop_masks.append(result["stop_mask"])
            first_stage_errors.append(result["first_stage_error"])
            final_stage_errors.append(result["final_stage_error"])
            commitment_losses.append(result["commitment_loss"])
            adaptive_metadata.append(
                {
                    "threshold": result["threshold"],
                    "selection_mode": result["selection_mode"],
                    "second_token_count": result["second_token_count"],
                    "stop_token_count": result["stop_token_count"],
                    "second_token_ratio": result["second_token_ratio"],
                    "stop_token_ratio": result["stop_token_ratio"],
                }
            )

        return {
            "indices": indices_list,
            "codes": indices_list,
            "feature_shapes": feature_shapes,
            "need_second_masks": normalized_masks,
            "stop_masks": stop_masks,
            "first_stage_errors": first_stage_errors,
            "final_stage_errors": final_stage_errors,
            "commitment_losses": commitment_losses,
            "thresholds": [item["threshold"] for item in adaptive_metadata],
            "adaptive_metadata": adaptive_metadata,
        }

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

    @torch.no_grad()
    def reconstruct_from_adaptive_indices(
        self,
        all_encoding_indices,
        feature_shapes=None,
        need_second_masks=None,
    ):
        """Decode adaptive RQ codes, inferring masks from ``-1`` if omitted.

        The dictionary returned by :meth:`forward_test_adaptive` can be passed
        directly as the first argument.  Keeping this decoder separate avoids
        changing the fixed-depth reconstruction contract.
        """
        self._require_adaptive_rq_eval()
        if isinstance(all_encoding_indices, dict):
            payload = all_encoding_indices
            all_encoding_indices = payload["indices"]
            if feature_shapes is None:
                feature_shapes = payload.get("feature_shapes")
            if need_second_masks is None:
                need_second_masks = payload.get("need_second_masks")

        num_scales = len(self.vector_quantizers)
        if len(all_encoding_indices) != num_scales:
            raise ValueError(
                f"expected {num_scales} adaptive index tensors, "
                f"got {len(all_encoding_indices)}"
            )
        mask_list = self._adaptive_mask_list(need_second_masks, num_scales)
        if feature_shapes is not None and len(feature_shapes) != num_scales:
            raise ValueError(
                f"expected {num_scales} feature shapes, got {len(feature_shapes)}"
            )

        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices):
            encoding_indices = encoding_indices.to(
                self.encoder_device, non_blocking=True
            )
            output_spatial_size = (
                feature_shapes[i] if feature_shapes is not None else None
            )
            quantized = self.vector_quantizers[
                i
            ].get_adaptive_quantized_features(
                encoding_indices,
                need_second_mask=mask_list[i],
                output_spatial_size=output_spatial_size,
            )
            quantized_features.append(quantized)
        quantized_features = self._to_decoder_device(quantized_features)
        reconstructed_image = self.semantic_decoder(quantized_features)
        reconstructed_image = self.swinir_enhance(reconstructed_image)
        return reconstructed_image

    def set_channel_prob(self, channel_prob):
        self.channel_prob = float(max(0.0, min(1.0, channel_prob)))
