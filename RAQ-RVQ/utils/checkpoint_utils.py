import torch

from models.deepsc import DeepSC


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_model_state_dict(checkpoint_path, device):
    # Both copied checkpoints are trusted local artifacts. The resume file
    # also contains optimizer and RNG objects, which require a full load on
    # PyTorch 2.6+.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    return extract_state_dict(checkpoint)


def make_state_dict_compatible(model, state_dict):
    model_state = model.state_dict()
    compatible_state = dict(state_dict)
    adjusted = []
    for key, value in state_dict.items():
        if not key.endswith("pos_encoder.pe"):
            continue
        if key not in model_state:
            continue
        current_value = model_state[key]
        if tuple(current_value.shape) != tuple(value.shape):
            compatible_state[key] = current_value
            adjusted.append((key, tuple(value.shape), tuple(current_value.shape)))
    return compatible_state, adjusted


def load_state_dict_compatible(model, state_dict, strict=True):
    compatible_state, adjusted = make_state_dict_compatible(model, state_dict)
    for key, ckpt_shape, model_shape in adjusted:
        print(
            "[Info] Compatible checkpoint load: regenerate "
            f"{key} ({ckpt_shape} -> {model_shape})"
        )
    return model.load_state_dict(compatible_state, strict=strict)


def infer_codebook_config(state_dict, cfg=None):
    codebook_weights = [
        state_dict[key] for key in sorted(state_dict)
        if key.startswith("vector_quantizers.")
        and key.endswith("codebook.embed.weight")
    ]
    if not codebook_weights:
        raise ValueError("No SimVQ codebook weights found in checkpoint.")
    embedding_dim_list = [weight.shape[1] for weight in codebook_weights]

    return {
        "num_downsample_blocks": len(codebook_weights),
        "num_embeddings_list": [weight.shape[0] for weight in codebook_weights],
        "embedding_dim_list": embedding_dim_list,
        "quantizer_type": "simvq",
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
    if inferred["quantizer_type"] != "simvq":
        raise ValueError(
            "This project only supports the copied SimVQ independent "
            "RAQ-RVQ checkpoint."
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
        raq_min_trg=cfg.RAQ_MIN_TRG,
        raq_max_trg=cfg.RAQ_MAX_TRG,
        raq_min_trg_list=cfg.RAQ_MIN_TRG_LIST,
        raq_max_trg_list=cfg.RAQ_MAX_TRG_LIST,
        independent_raq_rvq_depth=cfg.INDEPENDENT_RAQ_RVQ_DEPTH,
        independent_raq_rvq_k_lists=cfg.INDEPENDENT_RAQ_RVQ_K_LISTS,
    ).to(device)
    load_state_dict_compatible(model, state_dict)
    model.eval()
    return model, inferred
