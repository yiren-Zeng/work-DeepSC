import torch

from models.deepsc import DeepSC


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_model_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return extract_state_dict(checkpoint)


def infer_codebook_config(state_dict, cfg=None):
    if any(".qbridge." in key for key in state_dict):
        quantizer_type = "vitvq_nocompress"
        weight_suffix = "embedding.weight"
    elif any(
        key.startswith("vector_quantizers.") and key.endswith("embedding.weight")
        for key in state_dict
    ):
        quantizer_type = "vq"
        weight_suffix = "embedding.weight"
    else:
        quantizer_type = "simvq"
        weight_suffix = "codebook.embed.weight"
    codebook_weights = [
        state_dict[key] for key in sorted(state_dict)
        if key.startswith("vector_quantizers.") and key.endswith(weight_suffix)
    ]
    if not codebook_weights and cfg is not None and cfg.QUANTIZER_TYPE == "none":
        return {
            "num_downsample_blocks": cfg.NUM_DOWNSAMPLE_BLOCKS,
            "num_embeddings_list": list(cfg.NUM_EMBEDDINGS_LIST),
            "embedding_dim_list": list(cfg.EMBEDDING_DIM_LIST),
            "quantizer_type": "none",
        }
    if not codebook_weights:
        raise ValueError("No vector quantizer codebook weights found in checkpoint.")
    embedding_dim_list = [weight.shape[1] for weight in codebook_weights]
    if cfg is not None and hasattr(cfg, "QUANTIZER_AXIS_LIST"):
        embedding_dim_list = [
            cfg.EMBEDDING_DIM_LIST[idx] if cfg.QUANTIZER_AXIS_LIST[idx] == "channel" else weight.shape[1]
            for idx, weight in enumerate(codebook_weights)
        ]

    return {
        "num_downsample_blocks": len(codebook_weights),
        "num_embeddings_list": [weight.shape[0] for weight in codebook_weights],
        "embedding_dim_list": embedding_dim_list,
        "quantizer_type": quantizer_type,
    }


def build_model_from_checkpoint(checkpoint_path, cfg, device):
    cfg.validate()
    state_dict = load_model_state_dict(checkpoint_path, device)
    inferred = infer_codebook_config(state_dict, cfg)
    if inferred["num_downsample_blocks"] != cfg.NUM_DOWNSAMPLE_BLOCKS:
        raise ValueError(
            "Checkpoint layer count differs from Config; provide compatible "
            "NUM_DOWNSAMPLE_BLOCKS and DOWNSAMPLE_STRIDES before evaluation."
        )

    model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=inferred["num_downsample_blocks"],
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=inferred["num_embeddings_list"],
        embedding_dim_list=inferred["embedding_dim_list"],
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
        quantizer_type=inferred["quantizer_type"],
        quantizer_axis_list=cfg.QUANTIZER_AXIS_LIST,
        cvq_codeword_shapes=cfg.CVQ_CODEWORD_SHAPES,
        nested_channel_dropout_alpha=cfg.NESTED_CHANNEL_DROPOUT_ALPHA,
        vitvq_qbridge_type=cfg.VITVQ_QBRIDGE_TYPE,
        vitvq_emb_nograd=cfg.VITVQ_EMB_NOGRAD,
    ).to(device)
    model.load_state_dict(state_dict)
    if getattr(cfg, "MODEL_PARALLEL", False):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("SIMVQ_MODEL_PARALLEL=1 requires at least two visible CUDA devices.")
        model.enable_model_parallel(cfg.ENCODER_DEVICE, cfg.DECODER_DEVICE)
    model.eval()
    return model, inferred
