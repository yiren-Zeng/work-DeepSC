import torch
import torch.nn as nn
import os
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import ChannelwiseVectorQuantizer, VanillaVectorQuantizer, VectorQuantizer
from .channel import FiniteBlocklengthChannel
from .attention import BottleneckAttentionStack
from .raq import RAQ
from .swinir_enhance import SwinIREnhance
from utils.math_utils import sample_trg
from utils.raq_rvq import resolve_rvq_stage_k_lists


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
                 use_raq=False,
                 raq_target_list=None,
                 raq_min_trg=None,
                 raq_max_trg=None,
                 raq_recon_grad_mode="ste",
                 raq_generator_type="encoder_decoder",
                 raq_routed_src_enabled=False,
                 raq_routed_src_small_list=None,
                 raq_routed_src_large_list=None,
                 raq_routed_src_threshold=16,
                 use_dynamic_raq_rvq=False,
                 dynamic_raq_rvq_zero_codeword=True,
                 test_use_raq_rvq=False,
                 test_raq_rvq_depth=2,
                 test_raq_rvq_k_lists=None,
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
        if quantizer_axis_list is None:
            quantizer_axis_list = ["patch"] * num_downsample_blocks
        if len(quantizer_axis_list) != num_downsample_blocks:
            raise ValueError("quantizer_axis_list length must match num_downsample_blocks")
        if cvq_codeword_shapes is None:
            cvq_codeword_shapes = [None] * num_downsample_blocks
        if len(cvq_codeword_shapes) != num_downsample_blocks:
            raise ValueError("cvq_codeword_shapes length must match num_downsample_blocks")


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

        def build_quantizer(layer_index, num_embeddings):
            if quantizer_type == "none":
                return None
            if quantizer_type == "simvq" and self.quantizer_axis_list[layer_index] == "channel":
                return ChannelwiseVectorQuantizer(
                    num_embeddings, cvq_codeword_shapes[layer_index], commitment_cost
                )
            if quantizer_type == "simvq":
                return VectorQuantizer(
                    num_embeddings, embedding_dim_list[layer_index], commitment_cost
                )
            if quantizer_type == "vq":
                return VanillaVectorQuantizer(
                    num_embeddings, embedding_dim_list[layer_index], commitment_cost
                )
            if quantizer_type == "vitvq_nocompress":
                from .vector_quantizer_vitvq import ViTvqNoCompressVectorQuantizer

                return ViTvqNoCompressVectorQuantizer(
                    num_embeddings,
                    embedding_dim_list[layer_index],
                    commitment_cost,
                    qbridge_type=vitvq_qbridge_type,
                    emb_nograd=vitvq_emb_nograd,
                )
            raise ValueError(f"Unknown quantizer_type={quantizer_type!r}")

        for i in range(num_downsample_blocks):
            quantizer = build_quantizer(i, num_embeddings_list[i])
            if quantizer is None:
                continue
            self.vector_quantizers.append(quantizer)

        self.raq_routed_src_enabled = bool(raq_routed_src_enabled)
        self.raq_routed_src_threshold = int(raq_routed_src_threshold)
        self.raq_routed_src_small_list = (
            list(raq_routed_src_small_list) if raq_routed_src_small_list is not None else None
        )
        self.raq_routed_src_large_list = (
            list(raq_routed_src_large_list) if raq_routed_src_large_list is not None else list(num_embeddings_list)
        )
        self.vector_quantizers_small = nn.ModuleList()
        if self.raq_routed_src_enabled:
            if not use_raq:
                raise ValueError("Routed source codebooks require RAQ to be enabled.")
            if quantizer_type != "simvq" or any(axis != "patch" for axis in self.quantizer_axis_list):
                raise ValueError("Routed source codebooks currently support patch-wise SimVQ only.")
            if self.raq_routed_src_small_list is None:
                raise ValueError("Routed source codebooks require a small source codebook list.")
            if len(self.raq_routed_src_small_list) != num_downsample_blocks:
                raise ValueError("raq_routed_src_small_list length must match num_downsample_blocks")
            if list(self.raq_routed_src_large_list) != list(num_embeddings_list):
                raise ValueError("raq_routed_src_large_list must match num_embeddings_list")
            for i, small_k in enumerate(self.raq_routed_src_small_list):
                self.vector_quantizers_small.append(build_quantizer(i, int(small_k)))

        self.use_raq = bool(use_raq)
        self.use_dynamic_raq_rvq = bool(use_dynamic_raq_rvq)
        self.dynamic_raq_rvq_zero_codeword = bool(dynamic_raq_rvq_zero_codeword)
        if self.use_dynamic_raq_rvq and not self.use_raq:
            raise ValueError("dynamic RAQ-RVQ requires use_raq=True")
        # Plain inference controls only: these do not add parameters, buffers,
        # or modules and therefore do not change the checkpoint state_dict.
        self.test_use_raq_rvq = bool(test_use_raq_rvq)
        self.test_raq_rvq_depth = int(test_raq_rvq_depth)
        self.test_raq_rvq_k_lists = (
            [list(stage_sizes) for stage_sizes in test_raq_rvq_k_lists]
            if test_raq_rvq_k_lists is not None else None
        )
        if self.test_use_raq_rvq and not self.use_raq:
            raise ValueError("test-time RAQ-RVQ requires use_raq=True")
        if self.test_use_raq_rvq and self.test_raq_rvq_depth != 2:
            raise ValueError("test-time RAQ-RVQ currently supports depth=2 only")
        self.raq_recon_grad_mode = str(raq_recon_grad_mode).lower()
        if self.raq_recon_grad_mode not in {"ste", "dual"}:
            raise ValueError("raq_recon_grad_mode must be 'ste' or 'dual'")
        self.raq_generator_type = str(raq_generator_type).replace("-", "_").lower()
        if self.raq_generator_type not in {"encoder_decoder", "decoder_only"}:
            raise ValueError("raq_generator_type must be 'encoder_decoder' or 'decoder_only'")
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
        self.raq_target_list = list(raq_target_list) if raq_target_list is not None else list(num_embeddings_list)
        self.raqs = nn.ModuleList()
        self.raqs_rvq_stage2 = nn.ModuleList()
        if self.use_raq:
            # if raq_target_list is None:
            #     raise ValueError("RAQ enabled but raq_target_list is not configured.")
            if self.raq_min_trg_list is None or self.raq_max_trg_list is None:
                raise ValueError(
                    "RAQ enabled but target min/max bounds are not configured."
                )
            if quantizer_type != "simvq":
                raise ValueError("RAQ integration requires SIMVQ quantizers.")
            if any(axis != "patch" for axis in self.quantizer_axis_list):
                raise ValueError("RAQ integration currently supports patch-wise quantizers only.")
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
            if len(self.raq_target_list) != num_downsample_blocks:
                raise ValueError(
                    "raq_target_list length must match num_downsample_blocks"
                )
            if raq_target_list is not None:
                for layer_index, (target, min_k, max_k) in enumerate(
                    zip(
                        self.raq_target_list,
                        self.raq_min_trg_list,
                        self.raq_max_trg_list,
                    )
                ):
                    if target < min_k or target > max_k:
                        raise ValueError(
                            f"RAQ target layer {layer_index} K={target} is "
                            f"outside [{min_k},{max_k}]"
                        )
            if self.test_use_raq_rvq:
                # Resolve once at construction so invalid custom splits fail
                # before any dataset or checkpoint evaluation begins.
                self.test_raq_rvq_k_lists = resolve_rvq_stage_k_lists(
                    self.raq_target_list,
                    rvq_depth=self.test_raq_rvq_depth,
                    stage_k_lists=self.test_raq_rvq_k_lists,
                    min_k=self.raq_min_trg_list,
                    max_k=self.raq_max_trg_list,
                )
            # self.raq_target_list = list(raq_target_list)
            # if len(self.raq_target_list) != num_downsample_blocks:
            #     raise ValueError("raq_target_list length must match num_downsample_blocks")
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
                        generator_type=self.raq_generator_type,
                    )
                )
                if self.use_dynamic_raq_rvq:
                    self.raqs_rvq_stage2.append(
                        RAQ(
                            embedding_dim=Di,
                            n_embed_src=Ki,
                            n_embed_min_trg=min_k,
                            n_embed_max_trg=max_k,
                            device=device,
                            generator_type=self.raq_generator_type,
                            allocation_conditioned=True,
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
        self.vector_quantizers_small.to(self.encoder_device)
        self.raqs.to(self.encoder_device)
        self.raqs_rvq_stage2.to(self.encoder_device)
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

    def _sample_raq_target_list(self):
        return [
            sample_trg(min_k, max_k)
            for min_k, max_k in zip(
                self.raq_min_trg_list, self.raq_max_trg_list
            )
        ]

    def _select_source_quantizer(self, layer_index, k_trg=None):
        if self.raq_routed_src_enabled and k_trg is not None and int(k_trg) <= self.raq_routed_src_threshold:
            return (
                self.vector_quantizers_small[layer_index],
                int(self.raq_routed_src_small_list[layer_index]),
                "small",
            )
        route = "large" if self.raq_routed_src_enabled else "src"
        return self.vector_quantizers[layer_index], int(self.num_embeddings_list[layer_index]), route

    def _decode_features(self, quantized_features):
        quantized_features = self._to_decoder_device(quantized_features)
        reconstructed_images = self.semantic_decoder(quantized_features)
        return self.swinir_enhance(reconstructed_images)

    def _generate_raq_codebook(self, layer_index, k_trg, source_codebook=None):
        if source_codebook is None:
            source_quantizer, _, _ = self._select_source_quantizer(layer_index, k_trg)
            source_codebook = source_quantizer.transformed_weight()
        return self.raqs[layer_index].generate_codebook_transformer(k_trg, source_codebook)

    def _generate_rvq_stage_codebook(
        self,
        layer_index,
        stage_index,
        stage_k,
        k_total,
        stage_k_list,
        source_codebook,
    ):
        if stage_index == 0 or not self.use_dynamic_raq_rvq:
            return self._generate_raq_codebook(
                layer_index,
                stage_k,
                source_codebook=source_codebook,
            )

        k_first = int(stage_k_list[0])
        k_second = int(stage_k_list[1])
        codebook = self.raqs_rvq_stage2[layer_index].generate_codebook_transformer(
            int(stage_k),
            source_codebook,
            allocation=(int(k_total), k_first, k_second),
        )
        if self.dynamic_raq_rvq_zero_codeword:
            codebook = torch.cat([torch.zeros_like(codebook[:1]), codebook[1:]], dim=0)
        return codebook

    def _forward_dynamic_raq_rvq(
        self,
        encoder_features,
        target_list,
        rvq_k_lists,
        use_channel,
        snr_tensor,
        current_rc,
        current_mod_bits,
        ste_channel,
    ):
        rvq_k_lists = resolve_rvq_stage_k_lists(
            target_list,
            rvq_depth=2,
            stage_k_lists=rvq_k_lists,
            min_k=self.raq_min_trg_list,
            max_k=self.raq_max_trg_list,
        )
        quantized_raq = []
        z_q_raq_list = []
        vq_losses_raq = []
        codebooks_flat = []
        codebooks_nested = []
        indices_nested = []
        source_codebooks_list = []
        residual_mse_list = []

        for i, feat in enumerate(encoder_features):
            k_total = int(target_list[i])
            stage_k_list = list(rvq_k_lists[i])
            source_quantizer, _, _ = self._select_source_quantizer(i, k_total)
            source_codebook = source_quantizer.transformed_weight()
            source_codebooks_list.append(source_codebook)

            residual = feat
            clean_sum_raw = torch.zeros_like(feat)
            noisy_sum_raw = torch.zeros_like(feat)
            scale_vq_loss = feat.new_zeros(())
            scale_codebooks = []
            scale_indices = []
            scale_residual_mse = []

            for stage_index, stage_k in enumerate(stage_k_list):
                stage_codebook = self._generate_rvq_stage_codebook(
                    i,
                    stage_index,
                    int(stage_k),
                    k_total,
                    stage_k_list,
                    source_codebook,
                )
                stage_input = residual if stage_index == 0 else residual.detach()
                stage_vq_loss, _, stage_indices, stage_quantized_raw = (
                    source_quantizer.forward_raq(
                        stage_input,
                        stage_codebook,
                        return_raw=True,
                        recon_grad_mode="ste",
                    )
                )
                scale_vq_loss = scale_vq_loss + stage_vq_loss
                clean_sum_raw = clean_sum_raw + stage_quantized_raw
                residual = residual - stage_quantized_raw.detach()

                if use_channel:
                    corrupted_indices, _ = self.channel.apply_channel_noise(
                        stage_indices,
                        int(stage_k),
                        snr_tensor,
                        current_rc,
                        mod_bits=current_mod_bits,
                    )
                    noisy_stage_raw = source_quantizer.get_quantized_features(
                        corrupted_indices,
                        output_spatial_size=feat.shape[-2:],
                        codebook_weight=stage_codebook,
                    )
                else:
                    noisy_stage_raw = stage_quantized_raw
                noisy_sum_raw = noisy_sum_raw + noisy_stage_raw

                codebooks_flat.append(stage_codebook)
                scale_codebooks.append(stage_codebook)
                scale_indices.append(stage_indices)
                scale_residual_mse.append(residual.detach().pow(2).mean())

            # Apply one gradient bridge to the final RVQ sum. Summing the
            # per-stage STE tensors would duplicate the encoder identity path.
            quantized_clean = feat + (clean_sum_raw - feat).detach()
            if self.training and self.raq_recon_grad_mode == "dual":
                quantized_clean = quantized_clean + (
                    clean_sum_raw - clean_sum_raw.detach()
                )
            quantized_final = (
                quantized_clean + (noisy_sum_raw - clean_sum_raw).detach()
                if use_channel and ste_channel else
                (noisy_sum_raw if use_channel else quantized_clean)
            )

            quantized_raq.append(quantized_final)
            z_q_raq_list.append(clean_sum_raw)
            vq_losses_raq.append(scale_vq_loss)
            codebooks_nested.append(scale_codebooks)
            indices_nested.append(scale_indices)
            residual_mse_list.append(scale_residual_mse)

        reconstructed_images_raq = self._decode_features(quantized_raq)
        return {
            "reconstructed_images": reconstructed_images_raq,
            "vq_losses": vq_losses_raq,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "source_codebooks_list": source_codebooks_list,
            "W_trg_list": codebooks_flat,
            "rvq_codebooks_list": codebooks_nested,
            "rvq_indices_list": indices_nested,
            "z_q_raq_list": z_q_raq_list,
            "raq_target_list": list(target_list),
            "rvq_k_lists": [list(values) for values in rvq_k_lists],
            "rvq_residual_mse_list": residual_mse_list,
        }

    def _forward_impl(
        self,
        x,
        channel_coding_rate,
        ste_channel=False,
        raq_trg_list=None,
        raq_rvq_k_lists=None,
    ):
        x = self._to_encoder_device(x)
        snr_db = random.uniform(self.snr_range_db[0], self.snr_range_db[1])
        snr_tensor = torch.tensor(snr_db, device=self.encoder_device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = channel_coding_rate

        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])
        target_list = list(raq_trg_list or self._sample_raq_target_list()) if self.use_raq else None

        quantized_src = []
        z_q_src_list = [] if self.use_raq else None
        vq_losses_src = []
        source_route_list = [] if self.use_raq else None
        use_channel = self.quantizer_type != "none" and random.random() < self.channel_prob

        for i, feat in enumerate(encoder_features):
            if self.quantizer_type == "none":
                quantized_src.append(feat)
                vq_losses_src.append(feat.new_zeros(()))
                continue
            source_quantizer, source_k, route_name = self._select_source_quantizer(
                i, target_list[i] if target_list is not None else None
            )
            if source_route_list is not None:
                source_route_list.append(route_name)
            feat_for_quant = self._maybe_apply_nested_channel_dropout(i, feat)
            if self.use_raq:
                vq_loss, quantized_clean, encoding_idx, z_q_src = source_quantizer(
                    feat_for_quant,
                    return_raw=True,
                )
                z_q_src_list.append(z_q_src)
            else:
                vq_loss, quantized_clean, encoding_idx = source_quantizer(feat_for_quant)
            vq_losses_src.append(vq_loss)

            if use_channel:
                corrupted_idx, _ = self.channel.apply_channel_noise(
                    encoding_idx,
                    source_k,
                    snr_tensor,
                    current_rc,
                    mod_bits=current_mod_bits
                )
                quantized_noisy = source_quantizer.get_quantized_features(
                    corrupted_idx, output_spatial_size=feat_for_quant.shape[-2:]
                )
                quantized_final = (
                    quantized_clean + (quantized_noisy - quantized_clean).detach()
                    if ste_channel else quantized_noisy
                )
            else:
                quantized_final = quantized_clean
            quantized_src.append(quantized_final)

        reconstructed_images_src = self._decode_features(quantized_src)

        result = {
            "reconstructed_images": reconstructed_images_src,
            "vq_losses": vq_losses_src,
            "reconstructed_images_src": reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "current_snr": snr_db if use_channel else None,
            "channel_used": use_channel,
            "channel_prob": self.channel_prob,
        }
        if not self.use_raq:
            return result

        if self.use_dynamic_raq_rvq:
            rvq_result = self._forward_dynamic_raq_rvq(
                encoder_features,
                target_list,
                raq_rvq_k_lists,
                use_channel,
                snr_tensor,
                current_rc,
                current_mod_bits,
                ste_channel,
            )
            rvq_result["z_q_src_list"] = z_q_src_list
            rvq_result["source_route_list"] = source_route_list
            result.update(rvq_result)
            return result

        quantized_raq = []
        z_q_raq_list = []
        vq_losses_raq = []
        codebooks_trg_list = []
        source_codebooks_list = []
        for i, feat in enumerate(encoder_features):
            k_trg = int(target_list[i])
            source_quantizer, _, _ = self._select_source_quantizer(i, k_trg)
            source_codebook = source_quantizer.transformed_weight()
            source_codebooks_list.append(source_codebook)
            w_trg = self._generate_raq_codebook(i, k_trg, source_codebook=source_codebook)
            codebooks_trg_list.append(w_trg)
            vq_loss_raq, quantized_clean_raq, encoding_idx_raq, z_q_raq = source_quantizer.forward_raq(
                feat,
                w_trg,
                return_raw=True,
                recon_grad_mode=self.raq_recon_grad_mode if self.training else "ste",
            )
            z_q_raq_list.append(z_q_raq)
            vq_losses_raq.append(vq_loss_raq)

            if use_channel:
                corrupted_idx, _ = self.channel.apply_channel_noise(
                    encoding_idx_raq,
                    k_trg,
                    snr_tensor,
                    current_rc,
                    mod_bits=current_mod_bits
                )
                quantized_noisy_raq = source_quantizer.get_quantized_features(
                    corrupted_idx,
                    output_spatial_size=feat.shape[-2:],
                    codebook_weight=w_trg,
                )
                quantized_final_raq = (
                    quantized_clean_raq + (quantized_noisy_raq - quantized_clean_raq).detach()
                    if ste_channel else quantized_noisy_raq
                )
            else:
                quantized_final_raq = quantized_clean_raq
            quantized_raq.append(quantized_final_raq)

        reconstructed_images_raq = self._decode_features(quantized_raq)

        result.update({
            "reconstructed_images": reconstructed_images_raq,
            "vq_losses": vq_losses_raq,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "source_codebooks_list": source_codebooks_list,
            "W_trg_list": codebooks_trg_list,
            "z_q_src_list": z_q_src_list,
            "z_q_raq_list": z_q_raq_list,
            "raq_target_list": target_list,
            "source_route_list": source_route_list,
        })
        return result

    def forward_train(self, x, raq_trg_list=None, raq_rvq_k_lists=None):
        return self._forward_impl(
            x,
            channel_coding_rate=self.channel_coding_rate_train,
            ste_channel=True,
            raq_trg_list=raq_trg_list,
            raq_rvq_k_lists=raq_rvq_k_lists,
        )

    def forward_val(self, x, raq_trg_list=None, raq_rvq_k_lists=None):
        return self._forward_impl(
            x,
            channel_coding_rate=self.channel_coding_rate_val,
            ste_channel=False,
            raq_trg_list=raq_trg_list,
            raq_rvq_k_lists=raq_rvq_k_lists,
        )

    def _forward_test_raq_rvq(self, encoder_features):
        """Build nested two-stage RAQ indices/codebooks for each U-Net scale.

        This is a test-time zero-shot residual RAQ experiment.  In particular,
        the second-stage generator was not trained on first-stage residuals:
        every stage deliberately reuses the scale's existing, single-stage RAQ
        generator and the same learned source codebook.  Consequently, a quality
        regression here does not by itself rule out a fully trained RVQ model.
        """
        indices_by_scale = []
        codebooks_by_scale = []
        feature_shapes = []
        rvq_k_lists = []
        diagnostics = []

        for i, feat in enumerate(encoder_features):
            k_total = int(self.raq_target_list[i])
            stage_k_list = self.test_raq_rvq_k_lists[i]
            for stage_k in stage_k_list:
                min_k = self.raq_min_trg_list[i]
                max_k = self.raq_max_trg_list[i]
                if not min_k <= stage_k <= max_k:
                    raise ValueError(
                        f"RAQ-RVQ scale {i} stage K={stage_k} is outside the trained "
                        f"RAQ target range [{min_k}, {max_k}]"
                    )

            # Source routing is decided once from K_total. Trained dynamic RVQ
            # uses its independent allocation-conditioned stage-2 generator;
            # the legacy zero-shot branch continues to reuse self.raqs[i].
            source_quantizer, _, source_route = self._select_source_quantizer(i, k_total)
            source_codebook = source_quantizer.transformed_weight()
            residual = feat
            quantized_sum = torch.zeros_like(feat)
            scale_indices = []
            scale_codebooks = []
            stage_diagnostics = []
            residual_mse_energies = []

            input_mse_energy = float(feat.detach().pow(2).mean().item())
            for stage_index, stage_k in enumerate(stage_k_list):
                stage_codebook = self._generate_rvq_stage_codebook(
                    i,
                    stage_index,
                    stage_k,
                    k_total,
                    stage_k_list,
                    source_codebook,
                )
                _, _, stage_indices, quantized_raw = source_quantizer.forward_raq(
                    residual,
                    stage_codebook,
                    return_raw=True,
                )
                quantized_sum = quantized_sum + quantized_raw
                residual = residual - quantized_raw

                bits_per_index = stage_k.bit_length() - 1
                residual_mse = float(residual.detach().pow(2).mean().item())
                residual_mse_energies.append(residual_mse)
                stage_diagnostics.append({
                    "stage_index": stage_index,
                    "k": stage_k,
                    "bits_per_index": bits_per_index,
                    "index_min": int(stage_indices.detach().min().item()),
                    "index_max": int(stage_indices.detach().max().item()),
                    "codebook_size": int(stage_codebook.shape[0]),
                    "payload_bits": int(stage_indices.numel() * bits_per_index),
                    "residual_mse_energy": residual_mse,
                })
                scale_indices.append(stage_indices)
                scale_codebooks.append(stage_codebook)

            token_count = int(feat.shape[0] * feat.shape[-2] * feat.shape[-1])
            baseline_payload_bits = token_count * (k_total.bit_length() - 1)
            payload_bits = sum(stage["payload_bits"] for stage in stage_diagnostics)
            diagnostics.append({
                "scale_index": i,
                "source_route": source_route,
                "k_total": k_total,
                "stage_k_list": list(stage_k_list),
                "input_mse_energy": input_mse_energy,
                "residual_mse_energies": residual_mse_energies,
                "final_residual_mse_energy": residual_mse_energies[-1],
                "quantized_sum_mse_energy": float(
                    quantized_sum.detach().pow(2).mean().item()
                ),
                "stage_diagnostics": stage_diagnostics,
                "payload_bits": payload_bits,
                "baseline_payload_bits": baseline_payload_bits,
                "bit_budget_matches": payload_bits == baseline_payload_bits,
            })
            indices_by_scale.append(scale_indices)
            codebooks_by_scale.append(scale_codebooks)
            feature_shapes.append(tuple(feat.shape[-2:]))
            rvq_k_lists.append(list(stage_k_list))

        return {
            "indices": indices_by_scale,
            "feature_shapes": feature_shapes,
            "num_embeddings_list": rvq_k_lists,
            "branch": "raq_rvq",
            "codebooks": codebooks_by_scale,
            "raq_target_list": list(self.raq_target_list),
            "rvq_k_lists": rvq_k_lists,
            "test_raq_rvq_enabled": True,
            "rvq_depth": self.test_raq_rvq_depth,
            "rvq_diagnostics": diagnostics,
        }

    def forward_test(self, x):
        x = self._to_encoder_device(x)
        encoder_features = self.semantic_encoder(x)
        encoder_features[-1] = self.bottleneck_attention(encoder_features[-1])
        if self.quantizer_type == "none":
            return {"indices": encoder_features}
        if self.use_raq and self.test_use_raq_rvq:
            return self._forward_test_raq_rvq(encoder_features)
        indices_list = []
        feature_shapes = []
        codebooks = []
        num_embeddings_for_bits = self.num_embeddings_list
        for i, feat in enumerate(encoder_features):
            if self.use_raq:
                k_trg = int(self.raq_target_list[i])
                source_quantizer, _, _ = self._select_source_quantizer(i, k_trg)
                source_codebook = source_quantizer.transformed_weight()
                w_trg = self._generate_raq_codebook(i, k_trg, source_codebook=source_codebook)
                _, _, encoding_idx = source_quantizer.forward_raq(feat, w_trg)
                codebooks.append(w_trg)
                num_embeddings_for_bits = self.raq_target_list
            else:
                _, _, encoding_idx = self.vector_quantizers[i](feat)
            indices_list.append(encoding_idx)
            feature_shapes.append(tuple(feat.shape[-2:]))
        result = {
            "indices": indices_list,
            "feature_shapes": feature_shapes,
            "num_embeddings_list": list(num_embeddings_for_bits),
            "branch": "raq" if self.use_raq else "src",
        }
        if self.use_raq:
            result["codebooks"] = codebooks
        return result

    def reconstruct_from_indices(self, all_encoding_indices, feature_shapes=None, codebooks=None):
        if self.quantizer_type == "none":
            all_encoding_indices = self._to_decoder_device(all_encoding_indices)
            reconstructed_image = self.semantic_decoder(all_encoding_indices)
            return self.swinir_enhance(reconstructed_image)
        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices):
            output_spatial_size = feature_shapes[i] if feature_shapes is not None else None
            if isinstance(encoding_indices, (list, tuple)):
                if not encoding_indices:
                    raise ValueError(f"RVQ scale {i} has no stage indices")
                if codebooks is None or not isinstance(codebooks[i], (list, tuple)):
                    raise ValueError(
                        "Nested RVQ indices require matching nested codebooks"
                    )
                if len(encoding_indices) != len(codebooks[i]):
                    raise ValueError(
                        f"RVQ scale {i} has {len(encoding_indices)} index stages but "
                        f"{len(codebooks[i])} codebook stages"
                    )
                quantized = None
                for stage_indices, stage_codebook in zip(encoding_indices, codebooks[i]):
                    stage_indices = stage_indices.to(self.encoder_device, non_blocking=True)
                    quantized_stage = self.vector_quantizers[i].get_quantized_features(
                        stage_indices,
                        output_spatial_size=output_spatial_size,
                        codebook_weight=stage_codebook,
                    )
                    quantized = (
                        quantized_stage if quantized is None else quantized + quantized_stage
                    )
            else:
                encoding_indices = encoding_indices.to(self.encoder_device, non_blocking=True)
                codebook_weight = codebooks[i] if codebooks is not None else None
                quantized = self.vector_quantizers[i].get_quantized_features(
                    encoding_indices,
                    output_spatial_size=output_spatial_size,
                    codebook_weight=codebook_weight,
                )
            quantized_features.append(quantized)
        quantized_features = self._to_decoder_device(quantized_features)
        reconstructed_image = self.semantic_decoder(quantized_features)
        reconstructed_image = self.swinir_enhance(reconstructed_image)
        return reconstructed_image

    def set_channel_prob(self, channel_prob):
        self.channel_prob = float(max(0.0, min(1.0, channel_prob)))
