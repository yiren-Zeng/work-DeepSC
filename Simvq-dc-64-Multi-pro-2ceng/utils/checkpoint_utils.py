import torch

from models.deepsc import DeepSC


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def load_model_state_dict(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return extract_state_dict(checkpoint)


def infer_codebook_config(state_dict):
    codebook_weights = [
        state_dict[key] for key in sorted(state_dict)
        if key.endswith("codebook.embed.weight")
    ]
    if not codebook_weights:
        raise ValueError("No vector quantizer codebook weights found in checkpoint.")
    return {
        "num_downsample_blocks": len(codebook_weights),
        "num_embeddings_list": [weight.shape[0] for weight in codebook_weights],
        "embedding_dim_list": [weight.shape[1] for weight in codebook_weights],
    }


def build_model_from_checkpoint(checkpoint_path, cfg, device):
    cfg.validate()
    state_dict = load_model_state_dict(checkpoint_path, device)
    inferred = infer_codebook_config(state_dict)
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
        use_bottleneck_attention=cfg.USE_BOTTLENECK_ATTENTION,
        bottleneck_attention_blocks=cfg.BOTTLENECK_ATTENTION_BLOCKS,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, inferred
