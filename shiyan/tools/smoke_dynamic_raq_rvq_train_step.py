#!/usr/bin/env python3
"""Run one synthetic full-model dynamic RAQ-RVQ training step."""

import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import Config
from losses.deepsc_loss import DeepSCLoss
from models.deepsc import DeepSC
from utils.raq_rvq import resolve_rvq_stage_k_lists


def build_model(cfg, device):
    return DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        device=device,
        strides=cfg.DOWNSAMPLE_STRIDES,
        skip_dropout_p=cfg.SKIP_DROPOUT_P_INIT,
        channel_coding_rate_train=cfg.CHANNEL_CODING_RATE_TRAIN,
        channel_coding_rate_val=cfg.CHANNEL_CODING_RATE_VAL,
        block_length=cfg.BLOCK_LENGTH,
        snr_range_db=cfg.SNR_RANGE_DB,
        norm_type=cfg.NORM_TYPE,
        norm_groups=cfg.GROUP_NORM_GROUPS,
        activation=cfg.ACTIVATION,
        encoder_res_blocks=cfg.ENCODER_RES_BLOCKS,
        decoder_res_blocks=cfg.DECODER_RES_BLOCKS,
        upsample_mode=cfg.UPSAMPLE_MODE,
        use_cascade_downsample=cfg.USE_CASCADE_DOWNSAMPLE,
        use_bottleneck_attention=cfg.USE_BOTTLENECK_ATTENTION,
        bottleneck_attention_blocks=cfg.BOTTLENECK_ATTENTION_BLOCKS,
        use_swinir_enhance=cfg.USE_SWINIR_ENHANCE,
        swinir_enhance_blocks=cfg.SWINIR_ENHANCE_BLOCKS,
        quantizer_type=cfg.QUANTIZER_TYPE,
        quantizer_axis_list=cfg.QUANTIZER_AXIS_LIST,
        cvq_codeword_shapes=cfg.CVQ_CODEWORD_SHAPES,
        nested_channel_dropout_alpha=cfg.NESTED_CHANNEL_DROPOUT_ALPHA,
        vitvq_qbridge_type=cfg.VITVQ_QBRIDGE_TYPE,
        vitvq_emb_nograd=cfg.VITVQ_EMB_NOGRAD,
        use_raq=cfg.USE_RAQ,
        raq_target_list=cfg.RAQ_TARGET_LIST,
        raq_min_trg=cfg.RAQ_MIN_TRG,
        raq_max_trg=cfg.RAQ_MAX_TRG,
        raq_min_trg_list=getattr(cfg, "RAQ_MIN_TRG_LIST", None),
        raq_max_trg_list=getattr(cfg, "RAQ_MAX_TRG_LIST", None),
        raq_recon_grad_mode=cfg.RAQ_RECON_GRAD_MODE,
        raq_generator_type=cfg.RAQ_GENERATOR_TYPE,
        raq_routed_src_enabled=cfg.RAQ_ROUTED_SRC_ENABLED,
        raq_routed_src_small_list=cfg.RAQ_ROUTED_SRC_SMALL_LIST,
        raq_routed_src_large_list=cfg.RAQ_ROUTED_SRC_LARGE_LIST,
        raq_routed_src_threshold=cfg.RAQ_ROUTED_SRC_THRESHOLD,
        use_dynamic_raq_rvq=cfg.USE_DYNAMIC_RAQ_RVQ,
        dynamic_raq_rvq_zero_codeword=cfg.DYNAMIC_RAQ_RVQ_ZERO_CODEWORD,
    ).to(device)


def main():
    cfg = Config()
    cfg.validate()
    if not cfg.USE_DYNAMIC_RAQ_RVQ:
        raise ValueError("smoke test requires SIMVQ_USE_DYNAMIC_RAQ_RVQ=1")
    device = torch.device(cfg.DEVICE)
    model = build_model(cfg, device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE_G, betas=cfg.BETAS)
    model.set_channel_prob(0.0)
    loss_fn = DeepSCLoss(
        layer_weights=cfg.LAYER_LOSS_WEIGHTS_INIT,
        mse_weight=cfg.MSE_LOSS_WEIGHT,
        ms_ssim_weight=cfg.MS_SSIM_LOSS_WEIGHT,
        lpips_weight=cfg.LPIPS_LOSS_WEIGHT,
        raq_repulsion_weight=cfg.RAQ_REPULSION_WEIGHT,
        raq_latent_distill_weight=cfg.RAQ_LATENT_DISTILL_WEIGHT,
    ).to(device)

    image_size = int(os.environ.get("SIMVQ_SMOKE_IMAGE_SIZE", "64"))
    batch_size = int(os.environ.get("SIMVQ_SMOKE_BATCH_SIZE", "1"))
    images = torch.randn(batch_size, 3, image_size, image_size, device=device)
    target_list = list(cfg.RAQ_TARGET_LIST)
    rvq_k_lists = resolve_rvq_stage_k_lists(
        target_list,
        stage_k_lists=cfg.TEST_RAQ_RVQ_K_LISTS,
    )
    out = model.forward_train(
        images,
        raq_trg_list=target_list,
        raq_rvq_k_lists=rvq_k_lists,
    )
    recon, vq, _ = loss_fn(
        images,
        out["reconstructed_images_src"],
        out["vq_losses_src"],
        out["reconstructed_images_raq"],
        out["vq_losses_raq"],
        out["W_trg_list"],
        out["z_q_src_list"],
        out["z_q_raq_list"],
        out["source_codebooks_list"],
        return_details=True,
    )
    (recon + vq).backward()
    stage2_has_grad = any(
        parameter.grad is not None
        for parameter in model.raqs_rvq_stage2.parameters()
        if parameter.requires_grad
    )
    optimizer.step()
    print(f"target_list={target_list} rvq_k_lists={out['rvq_k_lists']}")
    print(f"batch_size={batch_size} image_size={image_size}")
    print(f"recon={recon.item():.6f} vq={vq.item():.6f}")
    print(f"stage2_has_grad={stage2_has_grad}")
    print(f"max_memory_mib={torch.cuda.max_memory_allocated(device) / 1024 ** 2:.1f}")


if __name__ == "__main__":
    main()
