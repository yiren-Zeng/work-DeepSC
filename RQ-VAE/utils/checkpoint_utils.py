import inspect
import re
from collections.abc import Mapping

import torch

from models.deepsc import DeepSC


CHECKPOINT_FORMAT_VERSION = 2
_SCALE_RE = re.compile(r"(?:^|\.)vector_quantizers\.(\d+)\.")
_CODEBOOK_DEPTH_RE = re.compile(r"(?:^|\.)codebooks\.(\d+)\.")


def _plain_list(value):
    if value is None:
        return None
    return list(value)


def _plain_shapes(value):
    if value is None:
        return None
    return [None if shape is None else list(shape) for shape in value]


def build_checkpoint_metadata(cfg):
    """Build a self-describing, pickle/JSON-friendly model configuration."""
    source_bits = getattr(cfg, "ESTIMATED_SOURCE_BITS_PER_IMAGE", None)
    if source_bits is not None:
        rounded = round(float(source_bits))
        source_bits = int(rounded) if abs(float(source_bits) - rounded) < 1e-9 else float(source_bits)

    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_format_version": CHECKPOINT_FORMAT_VERSION,
        "experiment_name": getattr(cfg, "EXPERIMENT_NAME", None),
        "experiment_family": getattr(cfg, "EXPERIMENT_FAMILY", None),
        "experiment_stage": getattr(cfg, "EXPERIMENT_STAGE", None),
        "quantizer_type": str(getattr(cfg, "QUANTIZER_TYPE", "simvq")).lower(),
        "num_downsample_blocks": int(getattr(cfg, "NUM_DOWNSAMPLE_BLOCKS", cfg.UNET_DEPTH)),
        "unet_depth": int(cfg.UNET_DEPTH),
        "num_embeddings_list": _plain_list(cfg.NUM_EMBEDDINGS_LIST),
        "embedding_dim_list": _plain_list(cfg.EMBEDDING_DIM_LIST),
        "rq_depth_list": _plain_list(getattr(cfg, "RQ_DEPTH_LIST", [1] * cfg.UNET_DEPTH)),
        "rq_ema_decay": float(getattr(cfg, "RQ_EMA_DECAY", 0.99)),
        "rq_restart_unused_codes": bool(getattr(cfg, "RQ_RESTART_UNUSED_CODES", True)),
        "rq_shared_codebook": bool(getattr(cfg, "RQ_SHARED_CODEBOOK", True)),
        "in_channels": int(cfg.IN_CHANNELS),
        "out_channels": int(cfg.OUT_CHANNELS),
        "base_channels": int(cfg.BASE_CHANNELS),
        "downsample_strides": _plain_list(cfg.DOWNSAMPLE_STRIDES),
        "quantizer_axis_list": _plain_list(getattr(cfg, "QUANTIZER_AXIS_LIST", None)),
        "cvq_codeword_shapes": _plain_shapes(getattr(cfg, "CVQ_CODEWORD_SHAPES", None)),
        "commitment_cost": float(cfg.COMMITMENT_COST),
        "norm_type": getattr(cfg, "NORM_TYPE", "batch"),
        "norm_groups": int(getattr(cfg, "GROUP_NORM_GROUPS", 32)),
        "activation": getattr(cfg, "ACTIVATION", "prelu"),
        "encoder_res_blocks": int(getattr(cfg, "ENCODER_RES_BLOCKS", 1)),
        "decoder_res_blocks": int(getattr(cfg, "DECODER_RES_BLOCKS", 1)),
        "upsample_mode": getattr(cfg, "UPSAMPLE_MODE", "nearest"),
        "use_cascade_downsample": bool(getattr(cfg, "USE_CASCADE_DOWNSAMPLE", True)),
        "use_bottleneck_attention": bool(getattr(cfg, "USE_BOTTLENECK_ATTENTION", False)),
        "bottleneck_attention_blocks": int(getattr(cfg, "BOTTLENECK_ATTENTION_BLOCKS", 1)),
        "use_swinir_enhance": bool(getattr(cfg, "USE_SWINIR_ENHANCE", False)),
        "swinir_enhance_blocks": int(getattr(cfg, "SWINIR_ENHANCE_BLOCKS", 4)),
        "skip_dropout_p_init": _plain_list(getattr(cfg, "SKIP_DROPOUT_P_INIT", None)),
        "skip_dropout_p_final": _plain_list(getattr(cfg, "SKIP_DROPOUT_P_FINAL", None)),
        "layer_loss_weights_init": _plain_list(
            getattr(cfg, "LAYER_LOSS_WEIGHTS_INIT", None)
        ),
        "layer_loss_weights_final": _plain_list(
            getattr(cfg, "LAYER_LOSS_WEIGHTS_FINAL", None)
        ),
        "mse_loss_weight": float(getattr(cfg, "MSE_LOSS_WEIGHT", 1.0)),
        "ms_ssim_loss_weight": float(getattr(cfg, "MS_SSIM_LOSS_WEIGHT", 0.0)),
        "lpips_loss_weight": float(getattr(cfg, "LPIPS_LOSS_WEIGHT", 0.0)),
        "phase1_end": float(getattr(cfg, "PHASE1_END", 0.1)),
        "phase2_end": float(getattr(cfg, "PHASE2_END", 0.4)),
        "channel_prob_start_epoch": int(
            getattr(cfg, "CHANNEL_PROB_START_EPOCH", 80)
        ),
        "channel_prob_end_epoch": int(
            getattr(cfg, "CHANNEL_PROB_END_EPOCH", 120)
        ),
        "channel_coding_rate_train": float(getattr(cfg, "CHANNEL_CODING_RATE_TRAIN", 0.5)),
        "channel_coding_rate_val": float(getattr(cfg, "CHANNEL_CODING_RATE_VAL", 0.5)),
        "block_length": int(getattr(cfg, "BLOCK_LENGTH", 256)),
        "snr_range_db": _plain_list(getattr(cfg, "SNR_RANGE_DB", [0, 15])),
        "learning_rate": float(getattr(cfg, "LEARNING_RATE_G", 5e-5)),
        "codebook_projection_learning_rate": float(
            getattr(cfg, "CODEBOOK_PROJ_LR", 2e-4)
        ),
        "adam_betas": _plain_list(getattr(cfg, "BETAS", (0.5, 0.999))),
        "lr_scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.5},
        "train_image_size": _plain_list(getattr(cfg, "TRAIN_IMAGE_SIZE", (256, 256))),
        "test_image_size": _plain_list(getattr(cfg, "TEST_IMAGE_SIZE", (768, 512))),
        "estimated_source_bits_per_image": source_bits,
        "estimated_source_bpp": float(getattr(cfg, "ESTIMATED_SOURCE_BPP", 0.0)),
        "estimated_test_source_bpp": float(getattr(cfg, "ESTIMATED_TEST_SOURCE_BPP", 0.0)),
        "estimated_test_transmission_ratio": float(
            getattr(cfg, "ESTIMATED_TEST_TRANSMISSION_RATIO", 0.0)
        ),
    }
    # Keep the rate values grouped as well as flat.  Flat keys make old and
    # lightweight consumers simple; the nested form makes their units clear.
    metadata["rate"] = {
        "source_bits_per_image": metadata["estimated_source_bits_per_image"],
        "source_bpp": metadata["estimated_source_bpp"],
        "test_source_bpp": metadata["estimated_test_source_bpp"],
        "test_transmission_ratio": metadata["estimated_test_transmission_ratio"],
    }
    metadata["projected_embedding"] = {
        "enabled": metadata["quantizer_type"]
        in {"simvq", "residual_simvq"},
        "base_embedding_frozen": metadata["quantizer_type"]
        in {"simvq", "residual_simvq"},
        "projection_type": "linear",
        "projection_bias": False,
        "shared_across_rq_depth": bool(metadata["rq_shared_codebook"]),
    }
    metadata["loss"] = {
        "mse_weight": metadata["mse_loss_weight"],
        "ms_ssim_weight": metadata["ms_ssim_loss_weight"],
        "lpips_weight": metadata["lpips_loss_weight"],
        "commitment_cost": metadata["commitment_cost"],
        "layer_weights_init": metadata["layer_loss_weights_init"],
        "layer_weights_final": metadata["layer_loss_weights_final"],
    }
    metadata["schedule"] = {
        "phase1_end": metadata["phase1_end"],
        "phase2_end": metadata["phase2_end"],
        "skip_dropout_init": metadata["skip_dropout_p_init"],
        "skip_dropout_final": metadata["skip_dropout_p_final"],
        "channel_prob_start_epoch": metadata["channel_prob_start_epoch"],
        "channel_prob_end_epoch": metadata["channel_prob_end_epoch"],
    }
    metadata["optimizer"] = {
        "type": "Adam",
        "learning_rate": metadata["learning_rate"],
        "codebook_projection_learning_rate": metadata[
            "codebook_projection_learning_rate"
        ],
        "betas": metadata["adam_betas"],
        "scheduler": metadata["lr_scheduler"],
    }
    return metadata


def build_checkpoint_payload(model, cfg, **training_state):
    """Package model state, v2 metadata, and optional resume-training state."""
    if "model_state_dict" in training_state or "model_metadata" in training_state:
        raise ValueError(
            "model_state_dict and model_metadata are managed by build_checkpoint_payload"
        )
    payload = dict(training_state)
    payload["model_state_dict"] = model.state_dict()
    payload["model_metadata"] = build_checkpoint_metadata(cfg)
    return payload


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def extract_checkpoint_metadata(checkpoint):
    if not isinstance(checkpoint, Mapping):
        return {}
    for key in ("model_metadata", "checkpoint_metadata", "metadata", "model_config"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    # Also accept a checkpoint that placed metadata beside model_state_dict.
    if "model_state_dict" in checkpoint and "quantizer_type" in checkpoint:
        return {
            key: value
            for key, value in checkpoint.items()
            if key not in {
                "model_state_dict",
                "optimizer_state_dict",
                "scheduler_state_dict",
                "rng_state",
                "cuda_rng_state",
            }
        }
    return {}


def load_checkpoint_payload(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(checkpoint)
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a valid model state_dict.")
    return checkpoint, state_dict, extract_checkpoint_metadata(checkpoint)


def load_model_state_dict(checkpoint_path, device):
    _, state_dict, _ = load_checkpoint_payload(checkpoint_path, device)
    return state_dict


def _state_by_scale(state_dict):
    groups = {}
    for key, value in state_dict.items():
        match = _SCALE_RE.search(key)
        if match is not None:
            groups.setdefault(int(match.group(1)), []).append((key, value))
    return {scale: groups[scale] for scale in sorted(groups)}


def _is_rq_ema_state(state_dict):
    return any(
        _SCALE_RE.search(key)
        and (key.endswith("cluster_size_ema") or key.endswith("embed_ema"))
        for key in state_dict
    )


def _rq_scale_shape(entries):
    cluster_sizes = [
        int(value.numel())
        for key, value in entries
        if key.endswith("cluster_size_ema") and isinstance(value, torch.Tensor)
    ]
    if cluster_sizes:
        if len(set(cluster_sizes)) != 1:
            raise ValueError(f"Inconsistent EMA codebook sizes in one RQ scale: {cluster_sizes}")
        num_embeddings = cluster_sizes[0]
    else:
        # Official VQEmbedding stores one extra padding row in `weight`.
        weight_rows = [
            int(value.shape[0]) - 1
            for key, value in entries
            if key.endswith("weight") and isinstance(value, torch.Tensor) and value.ndim == 2
        ]
        if not weight_rows:
            raise ValueError("Unable to infer an EMA-RQ codebook size from checkpoint state.")
        if len(set(weight_rows)) != 1:
            raise ValueError(f"Inconsistent padded RQ weight sizes in one scale: {weight_rows}")
        num_embeddings = weight_rows[0]

    embedding_dims = [
        int(value.shape[-1])
        for key, value in entries
        if isinstance(value, torch.Tensor)
        and value.ndim >= 2
        and (key.endswith("weight") or key.endswith("embed_ema"))
    ]
    if not embedding_dims:
        raise ValueError("Unable to infer EMA-RQ embedding dimension from checkpoint state.")
    if len(set(embedding_dims)) != 1:
        raise ValueError(f"Inconsistent RQ embedding dimensions in one scale: {embedding_dims}")

    depth_indices = []
    for key, _ in entries:
        match = _CODEBOOK_DEPTH_RE.search(key)
        if match is not None:
            depth_indices.append(int(match.group(1)))
    inferred_depth = max(depth_indices) + 1 if depth_indices else 1
    return num_embeddings, embedding_dims[0], inferred_depth


def _infer_rq_ema_config(state_dict, cfg=None):
    groups = _state_by_scale(state_dict)
    if not groups:
        raise ValueError("No vector_quantizers.<scale> EMA-RQ state found in checkpoint.")

    num_embeddings_list = []
    embedding_dim_list = []
    state_depth_list = []
    for entries in groups.values():
        num_embeddings, embedding_dim, depth = _rq_scale_shape(entries)
        num_embeddings_list.append(num_embeddings)
        embedding_dim_list.append(embedding_dim)
        state_depth_list.append(depth)

    cfg_depths = _plain_list(getattr(cfg, "RQ_DEPTH_LIST", None)) if cfg is not None else None
    if cfg_depths is not None and len(cfg_depths) == len(groups):
        # A shared RQ checkpoint has only codebooks.0 and therefore cannot
        # encode depth in its bare key layout.  The evaluation config is the
        # only available fallback when old checkpoints have no metadata.
        rq_depth_list = [
            max(state_depth, int(cfg_depth))
            for state_depth, cfg_depth in zip(state_depth_list, cfg_depths)
        ]
    else:
        rq_depth_list = state_depth_list

    return {
        "num_downsample_blocks": len(groups),
        "num_embeddings_list": num_embeddings_list,
        "embedding_dim_list": embedding_dim_list,
        "quantizer_type": "rq_ema",
        "rq_depth_list": rq_depth_list,
        "rq_ema_decay": float(getattr(cfg, "RQ_EMA_DECAY", 0.99)) if cfg is not None else 0.99,
        "rq_restart_unused_codes": bool(
            getattr(cfg, "RQ_RESTART_UNUSED_CODES", True)
        ) if cfg is not None else True,
        "rq_shared_codebook": bool(
            getattr(cfg, "RQ_SHARED_CODEBOOK", True)
        ) if cfg is not None else True,
    }


def _legacy_weight_entries(state_dict, suffix):
    by_scale = _state_by_scale(state_dict)
    entries = []
    for scale, values in by_scale.items():
        matches = [(key, value) for key, value in values if key.endswith(suffix)]
        if len(matches) > 1:
            raise ValueError(
                f"Checkpoint has multiple legacy codebook weights for scale {scale}: "
                f"{[key for key, _ in matches]}"
            )
        if matches:
            entries.append(matches[0][1])
    return entries


def _infer_legacy_config(state_dict, cfg=None, forced_quantizer_type=None):
    if forced_quantizer_type == "none":
        codebook_weights = []
        quantizer_type = "none"
    elif forced_quantizer_type == "vitvq_nocompress" or any(
        ".qbridge." in key for key in state_dict
    ):
        quantizer_type = "vitvq_nocompress"
        codebook_weights = _legacy_weight_entries(state_dict, "embedding.weight")
    elif forced_quantizer_type == "vq" or any(
        _SCALE_RE.search(key) and key.endswith("embedding.weight") for key in state_dict
    ):
        quantizer_type = "vq"
        codebook_weights = _legacy_weight_entries(state_dict, "embedding.weight")
    else:
        quantizer_type = "simvq"
        codebook_weights = _legacy_weight_entries(state_dict, "codebook.embed.weight")

    if not codebook_weights and (
        quantizer_type == "none"
        or (cfg is not None and str(getattr(cfg, "QUANTIZER_TYPE", "")).lower() == "none")
    ):
        return {
            "num_downsample_blocks": int(cfg.NUM_DOWNSAMPLE_BLOCKS),
            "num_embeddings_list": list(cfg.NUM_EMBEDDINGS_LIST),
            "embedding_dim_list": list(cfg.EMBEDDING_DIM_LIST),
            "quantizer_type": "none",
        }
    if not codebook_weights:
        raise ValueError("No vector quantizer codebook weights found in checkpoint.")

    embedding_dim_list = [int(weight.shape[1]) for weight in codebook_weights]
    if cfg is not None and hasattr(cfg, "QUANTIZER_AXIS_LIST"):
        embedding_dim_list = [
            int(cfg.EMBEDDING_DIM_LIST[idx])
            if cfg.QUANTIZER_AXIS_LIST[idx] == "channel"
            else int(weight.shape[1])
            for idx, weight in enumerate(codebook_weights)
        ]
    return {
        "num_downsample_blocks": len(codebook_weights),
        "num_embeddings_list": [int(weight.shape[0]) for weight in codebook_weights],
        "embedding_dim_list": embedding_dim_list,
        "quantizer_type": quantizer_type,
    }


def _metadata_list(metadata, key):
    value = metadata.get(key)
    return list(value) if value is not None else None


def infer_codebook_config(state_dict, cfg=None, metadata=None):
    """Infer quantizer layout, preferring explicit v2 checkpoint metadata."""
    metadata = dict(metadata or {})
    explicit_type = metadata.get("quantizer_type")
    if explicit_type is not None:
        explicit_type = str(explicit_type).lower()

    if (
        explicit_type in {"rq_ema", "residual_simvq"}
        and metadata.get("num_embeddings_list") is not None
        and metadata.get("embedding_dim_list") is not None
    ):
        # A v2 checkpoint is self-describing.  Do not make successful loading
        # depend on internal buffer names when all architectural values are
        # explicitly present.
        num_embeddings_list = _metadata_list(metadata, "num_embeddings_list")
        embedding_dim_list = _metadata_list(metadata, "embedding_dim_list")
        count = int(metadata.get("num_downsample_blocks", len(num_embeddings_list)))
        inferred = {
            "num_downsample_blocks": count,
            "num_embeddings_list": num_embeddings_list,
            "embedding_dim_list": embedding_dim_list,
            "quantizer_type": explicit_type,
            "rq_depth_list": _metadata_list(metadata, "rq_depth_list") or [1] * count,
            "rq_shared_codebook": bool(metadata.get("rq_shared_codebook", True)),
        }
        if explicit_type == "rq_ema":
            inferred.update(
                {
                    "rq_ema_decay": float(metadata.get("rq_ema_decay", 0.99)),
                    "rq_restart_unused_codes": bool(
                        metadata.get("rq_restart_unused_codes", True)
                    ),
                }
            )
    elif explicit_type == "rq_ema" or (explicit_type is None and _is_rq_ema_state(state_dict)):
        inferred = _infer_rq_ema_config(state_dict, cfg)
    elif explicit_type == "residual_simvq":
        # Residual-SimVQ intentionally reuses the same single shared SimVQ
        # codebook keys as legacy SimVQ, so a bare state_dict cannot be
        # distinguished safely.  An explicit type marker is authoritative;
        # infer only K/D from those familiar weights and take RQ layout from
        # metadata (or Config for an early transitional checkpoint).
        inferred = _infer_legacy_config(
            state_dict, cfg, forced_quantizer_type="simvq"
        )
        inferred["quantizer_type"] = "residual_simvq"
        count = inferred["num_downsample_blocks"]
        inferred["rq_depth_list"] = (
            _metadata_list(metadata, "rq_depth_list")
            or list(getattr(cfg, "RQ_DEPTH_LIST", [1] * count))
        )
        inferred["rq_shared_codebook"] = bool(
            metadata.get(
                "rq_shared_codebook",
                getattr(cfg, "RQ_SHARED_CODEBOOK", True),
            )
        )
    elif explicit_type == "none":
        if cfg is None and (
            metadata.get("num_embeddings_list") is None
            or metadata.get("embedding_dim_list") is None
        ):
            raise ValueError("A no-quantization checkpoint requires explicit dimensions or Config.")
        default_count = cfg.NUM_DOWNSAMPLE_BLOCKS if cfg is not None else len(
            metadata["num_embeddings_list"]
        )
        inferred = {
            "num_downsample_blocks": int(metadata.get("num_downsample_blocks", default_count)),
            "num_embeddings_list": _metadata_list(metadata, "num_embeddings_list")
            or list(cfg.NUM_EMBEDDINGS_LIST),
            "embedding_dim_list": _metadata_list(metadata, "embedding_dim_list")
            or list(cfg.EMBEDDING_DIM_LIST),
            "quantizer_type": "none",
        }
    else:
        inferred = _infer_legacy_config(state_dict, cfg, explicit_type)

    # Explicit metadata is authoritative.  State-based inference above remains
    # a validation/fallback path for old bare and old resume checkpoints.
    for key in (
        "num_downsample_blocks",
        "num_embeddings_list",
        "embedding_dim_list",
        "rq_depth_list",
        "rq_ema_decay",
        "rq_restart_unused_codes",
        "rq_shared_codebook",
    ):
        if key in metadata and metadata[key] is not None:
            value = metadata[key]
            inferred[key] = list(value) if key.endswith("_list") else value
    if explicit_type is not None:
        inferred["quantizer_type"] = explicit_type

    inferred["num_downsample_blocks"] = int(inferred["num_downsample_blocks"])
    for key in ("num_embeddings_list", "embedding_dim_list", "rq_depth_list"):
        if key in inferred:
            inferred[key] = [int(value) for value in inferred[key]]
    if inferred["quantizer_type"] in {"rq_ema", "residual_simvq"}:
        count = inferred["num_downsample_blocks"]
        inferred.setdefault("rq_depth_list", [1] * count)
        inferred.setdefault("rq_shared_codebook", True)
        if inferred["quantizer_type"] == "rq_ema":
            inferred.setdefault("rq_ema_decay", 0.99)
            inferred.setdefault("rq_restart_unused_codes", True)

    count = inferred["num_downsample_blocks"]
    for key in ("num_embeddings_list", "embedding_dim_list"):
        if len(inferred[key]) != count:
            raise ValueError(f"Checkpoint {key} length does not match its U-Net scale count.")
    if inferred.get("rq_depth_list") is not None and len(inferred["rq_depth_list"]) != count:
        raise ValueError("Checkpoint rq_depth_list length does not match its U-Net scale count.")
    inferred["checkpoint_metadata"] = metadata
    return inferred


def _meta_or_cfg(metadata, metadata_key, cfg, cfg_name, default=None):
    value = metadata.get(metadata_key)
    if value is not None:
        return value
    return getattr(cfg, cfg_name, default)


def build_model_from_checkpoint(checkpoint_path, cfg, device):
    cfg.validate()
    _, state_dict, metadata = load_checkpoint_payload(checkpoint_path, device)
    inferred = infer_codebook_config(state_dict, cfg, metadata=metadata)

    num_scales = inferred["num_downsample_blocks"]
    strides = list(_meta_or_cfg(metadata, "downsample_strides", cfg, "DOWNSAMPLE_STRIDES"))
    if len(strides) != num_scales:
        raise ValueError(
            "Checkpoint layer count differs from its configured downsample strides. "
            "Use checkpoint metadata or provide a compatible Config."
        )
    skip_dropout = list(
        _meta_or_cfg(metadata, "skip_dropout_p_init", cfg, "SKIP_DROPOUT_P_INIT", [])
    )
    quantizer_axes = list(
        _meta_or_cfg(
            metadata,
            "quantizer_axis_list",
            cfg,
            "QUANTIZER_AXIS_LIST",
            ["patch"] * num_scales,
        )
    )
    cvq_shapes = _meta_or_cfg(
        metadata,
        "cvq_codeword_shapes",
        cfg,
        "CVQ_CODEWORD_SHAPES",
        [None] * num_scales,
    )
    cvq_shapes = [None if shape is None else tuple(shape) for shape in cvq_shapes]

    model_kwargs = {
        "in_channels": int(_meta_or_cfg(metadata, "in_channels", cfg, "IN_CHANNELS", 3)),
        "out_channels": int(_meta_or_cfg(metadata, "out_channels", cfg, "OUT_CHANNELS", 3)),
        "num_downsample_blocks": num_scales,
        "base_channels": int(_meta_or_cfg(metadata, "base_channels", cfg, "BASE_CHANNELS")),
        "num_embeddings_list": inferred["num_embeddings_list"],
        "embedding_dim_list": inferred["embedding_dim_list"],
        "commitment_cost": float(
            _meta_or_cfg(metadata, "commitment_cost", cfg, "COMMITMENT_COST", 0.25)
        ),
        "device": device,
        "strides": strides,
        "skip_dropout_p": skip_dropout,
        "channel_coding_rate_train": float(
            _meta_or_cfg(
                metadata, "channel_coding_rate_train", cfg, "CHANNEL_CODING_RATE_TRAIN", 0.5
            )
        ),
        "channel_coding_rate_val": float(
            _meta_or_cfg(
                metadata, "channel_coding_rate_val", cfg, "CHANNEL_CODING_RATE_VAL", 0.5
            )
        ),
        "block_length": int(_meta_or_cfg(metadata, "block_length", cfg, "BLOCK_LENGTH", 256)),
        "snr_range_db": list(
            _meta_or_cfg(metadata, "snr_range_db", cfg, "SNR_RANGE_DB", [0, 15])
        ),
        "norm_type": _meta_or_cfg(metadata, "norm_type", cfg, "NORM_TYPE", "batch"),
        "norm_groups": int(
            _meta_or_cfg(metadata, "norm_groups", cfg, "GROUP_NORM_GROUPS", 32)
        ),
        "activation": _meta_or_cfg(metadata, "activation", cfg, "ACTIVATION", "prelu"),
        "encoder_res_blocks": int(
            _meta_or_cfg(metadata, "encoder_res_blocks", cfg, "ENCODER_RES_BLOCKS", 1)
        ),
        "decoder_res_blocks": int(
            _meta_or_cfg(metadata, "decoder_res_blocks", cfg, "DECODER_RES_BLOCKS", 1)
        ),
        "upsample_mode": _meta_or_cfg(
            metadata, "upsample_mode", cfg, "UPSAMPLE_MODE", "nearest"
        ),
        "use_cascade_downsample": bool(
            _meta_or_cfg(
                metadata, "use_cascade_downsample", cfg, "USE_CASCADE_DOWNSAMPLE", True
            )
        ),
        "use_bottleneck_attention": bool(
            _meta_or_cfg(
                metadata,
                "use_bottleneck_attention",
                cfg,
                "USE_BOTTLENECK_ATTENTION",
                False,
            )
        ),
        "bottleneck_attention_blocks": int(
            _meta_or_cfg(
                metadata,
                "bottleneck_attention_blocks",
                cfg,
                "BOTTLENECK_ATTENTION_BLOCKS",
                1,
            )
        ),
        "use_swinir_enhance": bool(
            _meta_or_cfg(metadata, "use_swinir_enhance", cfg, "USE_SWINIR_ENHANCE", False)
        ),
        "swinir_enhance_blocks": int(
            _meta_or_cfg(
                metadata, "swinir_enhance_blocks", cfg, "SWINIR_ENHANCE_BLOCKS", 4
            )
        ),
        "quantizer_type": inferred["quantizer_type"],
        "quantizer_axis_list": quantizer_axes,
        "cvq_codeword_shapes": cvq_shapes,
        "nested_channel_dropout_alpha": float(
            getattr(cfg, "NESTED_CHANNEL_DROPOUT_ALPHA", 0.0)
        ),
        "vitvq_qbridge_type": getattr(cfg, "VITVQ_QBRIDGE_TYPE", "QBridgeNoCompress-S"),
        "vitvq_emb_nograd": bool(getattr(cfg, "VITVQ_EMB_NOGRAD", False)),
        "rq_depth_list": inferred.get(
            "rq_depth_list", list(getattr(cfg, "RQ_DEPTH_LIST", [1] * num_scales))
        ),
        "rq_ema_decay": float(
            inferred.get("rq_ema_decay", getattr(cfg, "RQ_EMA_DECAY", 0.99))
        ),
        "rq_restart_unused_codes": bool(
            inferred.get(
                "rq_restart_unused_codes", getattr(cfg, "RQ_RESTART_UNUSED_CODES", True)
            )
        ),
        "rq_shared_codebook": bool(
            inferred.get("rq_shared_codebook", getattr(cfg, "RQ_SHARED_CODEBOOK", True))
        ),
    }

    # The guard keeps old DeepSC revisions loadable while the new constructor
    # receives every RQ argument as soon as those parameters are available.
    accepted = inspect.signature(DeepSC.__init__).parameters
    model_kwargs = {key: value for key, value in model_kwargs.items() if key in accepted}
    model = DeepSC(**model_kwargs).to(device)
    model.load_state_dict(state_dict)
    if getattr(cfg, "MODEL_PARALLEL", False):
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("SIMVQ_MODEL_PARALLEL=1 requires at least two visible CUDA devices.")
        model.enable_model_parallel(cfg.ENCODER_DEVICE, cfg.DECODER_DEVICE)
    model.eval()
    return model, inferred
