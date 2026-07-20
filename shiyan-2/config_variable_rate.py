"""Configuration for the isolated single-teacher variable-rate RAQ pipeline.

This module intentionally does not mutate :mod:`config` or inherit its legacy RAQ
defaults.  Every profile is a two-layer atomic unit ``(K0, K1)`` and both layers
use the same supported power-of-two set from 2 through 2048.
"""

from __future__ import annotations

import math
import os
from itertools import product
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


SUPPORTED_K_VALUES = (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
ALL_PROFILES = tuple(product(SUPPORTED_K_VALUES, repeat=2))
DEFAULT_VAL_PROFILES = (
    (2048, 2048),
    (2048, 16),
    (16, 2),
    (1024, 256),
    (512, 64),
    (64, 16),
)


def _first_env(names: Sequence[str], default: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip() != "":
            return value.strip()
    return default


def _env_str(name: str, default: str = "", aliases: Sequence[str] = ()) -> str:
    return str(_first_env((name, *aliases), default))


def _env_int(name: str, default: int, aliases: Sequence[str] = ()) -> int:
    value = _first_env((name, *aliases))
    return default if value is None else int(value)


def _env_float(name: str, default: float, aliases: Sequence[str] = ()) -> float:
    value = _first_env((name, *aliases))
    return default if value is None else float(value)


def _env_bool(name: str, default: bool, aliases: Sequence[str] = ()) -> bool:
    value = _first_env((name, *aliases))
    if value is None:
        return bool(default)
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean (0/1, true/false), got {value!r}")


def _env_int_list(name: str, default: Sequence[int]) -> list[int]:
    value = _first_env((name,))
    if value is None:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _env_float_list(name: str, default: Sequence[float]) -> list[float]:
    value = _first_env((name,))
    if value is None:
        return list(default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _env_size(name: str, default: tuple[int, int]) -> tuple[int, int]:
    value = _first_env((name,))
    if value is None:
        return default
    parts = [part.strip() for part in value.lower().replace("x", ",").split(",")]
    parts = [part for part in parts if part]
    if len(parts) != 2:
        raise ValueError(f"{name} must be HxW or H,W, got {value!r}")
    return int(parts[0]), int(parts[1])


def parse_profile(
    value: str | Sequence[int],
    *,
    supported_k: Sequence[int] = SUPPORTED_K_VALUES,
    name: str = "profile",
) -> tuple[int, int]:
    """Parse and strictly validate one ``K0xK1`` profile."""

    if isinstance(value, str):
        parts = [part.strip() for part in value.lower().split("x") if part.strip()]
    else:
        parts = list(value)
    if len(parts) != 2:
        raise ValueError(f"{name} must contain exactly two K values, got {value!r}")
    try:
        profile = int(parts[0]), int(parts[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be formatted as K0xK1, got {value!r}") from exc
    allowed = set(int(k) for k in supported_k)
    unsupported = [k for k in profile if k not in allowed]
    if unsupported:
        raise ValueError(
            f"{name} contains unsupported K={unsupported}; supported values are "
            f"{sorted(allowed)}"
        )
    return profile


def parse_profiles(
    value: str | Iterable[Sequence[int]],
    *,
    supported_k: Sequence[int] = SUPPORTED_K_VALUES,
    name: str = "profiles",
) -> tuple[tuple[int, int], ...]:
    """Parse ``K0xK1;...`` or the special value ``all``.

    Duplicate profiles are rejected because they would otherwise silently bias
    both the sampler and validation aggregate.
    """

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "all":
            return tuple(product(tuple(supported_k), repeat=2))
        if not normalized:
            raise ValueError(f"{name} cannot be empty")
        raw_profiles: Iterable[str | Sequence[int]] = [
            item.strip() for item in normalized.split(";") if item.strip()
        ]
    else:
        raw_profiles = value

    profiles = tuple(
        parse_profile(item, supported_k=supported_k, name=name)
        for item in raw_profiles
    )
    if not profiles:
        raise ValueError(f"{name} must contain at least one profile")
    if len(set(profiles)) != len(profiles):
        raise ValueError(f"{name} contains duplicate profiles")
    return profiles


def _parse_profile_weights(
    value: str,
    *,
    supported_k: Sequence[int],
) -> dict[tuple[int, int], float]:
    """Parse optional ``K0xK1=weight;...`` validation weights."""

    if not value.strip():
        return {}
    result: dict[tuple[int, int], float] = {}
    for entry in value.split(";"):
        if not entry.strip():
            continue
        try:
            profile_text, weight_text = entry.split("=", maxsplit=1)
        except ValueError as exc:
            raise ValueError(
                "SIMVQ_RAQ_VAL_PROFILE_WEIGHTS entries must be K0xK1=weight"
            ) from exc
        profile = parse_profile(
            profile_text,
            supported_k=supported_k,
            name="SIMVQ_RAQ_VAL_PROFILE_WEIGHTS",
        )
        if profile in result:
            raise ValueError(f"duplicate validation weight for profile {profile}")
        result[profile] = float(weight_text)
    return result


def _normalize_stage(value: str) -> str:
    aliases = {
        "1": "src_teacher",
        "stage1": "src_teacher",
        "stage1_src_teacher": "src_teacher",
        "teacher": "src_teacher",
        "src": "src_teacher",
        "src_teacher": "src_teacher",
        "2": "identity_warmup",
        "stage2": "identity_warmup",
        "stage2_identity_warmup": "identity_warmup",
        "identity": "identity_warmup",
        "identity_warmup": "identity_warmup",
        "3": "variable_rate",
        "stage3": "variable_rate",
        "stage3_variable_rate": "variable_rate",
        "variable_rate": "variable_rate",
        "4": "joint_lite",
        "stage4": "joint_lite",
        "stage4_joint_lite": "joint_lite",
        "jointlite": "joint_lite",
        "joint_lite": "joint_lite",
        "5": "channel_finetune",
        "stage5": "channel_finetune",
        "stage5_channel_finetune": "channel_finetune",
        "channel": "channel_finetune",
        "channel_finetune": "channel_finetune",
    }
    key = value.strip().lower().replace("-", "_")
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown variable-rate stage {value!r}; expected stage1..stage5"
        ) from exc


_STAGE = _normalize_stage(
    _env_str(
        "SIMVQ_RAQ_STAGE",
        "src_teacher",
        aliases=("SIMVQ_VARIABLE_RATE_STAGE", "SIMVQ_RAQ_TRAIN_STAGE"),
    )
)
_TARGET_PROFILES = parse_profiles(
    _env_str("SIMVQ_RAQ_TARGET_PROFILES", "all"),
    name="SIMVQ_RAQ_TARGET_PROFILES",
)
_VAL_PROFILES = parse_profiles(
    _env_str(
        "SIMVQ_RAQ_VAL_PROFILES",
        ";".join(f"{k0}x{k1}" for k0, k1 in DEFAULT_VAL_PROFILES),
    ),
    name="SIMVQ_RAQ_VAL_PROFILES",
)
_MIN_PROFILE = parse_profile(
    _env_str("SIMVQ_RAQ_MIN_PROFILE", "2x2"),
    name="SIMVQ_RAQ_MIN_PROFILE",
)


class VariableRateConfig:
    """Import-time environment configuration used by the new five-stage runner."""

    # Profile space and stage.
    SUPPORTED_K_VALUES = SUPPORTED_K_VALUES
    ALL_PROFILES = ALL_PROFILES
    MAX_PROFILE = (2048, 2048)
    MIN_PROFILE = _MIN_PROFILE
    RAQ_MIN_PROFILE = MIN_PROFILE
    TARGET_PROFILES = _TARGET_PROFILES
    RAQ_TARGET_PROFILES = TARGET_PROFILES
    VAL_PROFILES = _VAL_PROFILES
    RAQ_VAL_PROFILES = VAL_PROFILES
    TRAIN_STAGE = _STAGE
    RAQ_STAGE = TRAIN_STAGE
    STAGE_INDEX = {
        "src_teacher": 1,
        "identity_warmup": 2,
        "variable_rate": 3,
        "joint_lite": 4,
        "channel_finetune": 5,
    }[TRAIN_STAGE]
    SANDWICH_NUM_RANDOM = _env_int("SIMVQ_RAQ_SANDWICH_NUM_RANDOM", 1)
    RAQ_SANDWICH_NUM_RANDOM = SANDWICH_NUM_RANDOM
    PROFILE_SAMPLER_SEED = _env_int("SIMVQ_RAQ_PROFILE_SAMPLER_SEED", 3407)
    SEED = _env_int("SIMVQ_SEED", 42)

    # Existing model-facing names, fixed to the two-layer 2048 source teacher.
    IN_CHANNELS = _env_int("SIMVQ_IN_CHANNELS", 3)
    OUT_CHANNELS = _env_int("SIMVQ_OUT_CHANNELS", 3)
    UNET_DEPTH = _env_int("SIMVQ_UNET_DEPTH", 2)
    NUM_DOWNSAMPLE_BLOCKS = UNET_DEPTH
    BASE_CHANNELS = _env_int("SIMVQ_BASE_CHANNELS", 256)
    EMBEDDING_DIM_LIST = _env_int_list(
        "SIMVQ_EMBEDDING_DIM_LIST", [BASE_CHANNELS * 2, BASE_CHANNELS * 4]
    )
    NUM_EMBEDDINGS_PER_LAYER = None
    NUM_EMBEDDINGS_LIST = [2048, 2048]
    SOURCE_NUM_EMBEDDINGS = tuple(NUM_EMBEDDINGS_LIST)
    MAX_CODEBOOK_SIZES = tuple(NUM_EMBEDDINGS_LIST)
    COMMITMENT_COST = _env_float("SIMVQ_COMMITMENT_COST", 0.25)
    DOWNSAMPLE_STRIDES = _env_int_list("SIMVQ_DOWNSAMPLE_STRIDES", [8, 2])
    QUANTIZER_TYPE = _env_str("SIMVQ_QUANTIZER_TYPE", "simvq").lower()
    QUANTIZER_AXIS_LIST = ["patch", "patch"]
    NORM_TYPE = _env_str("SIMVQ_NORM_TYPE", "group").lower()
    GROUP_NORM_GROUPS = _env_int("SIMVQ_GROUP_NORM_GROUPS", 32)
    ACTIVATION = _env_str("SIMVQ_ACTIVATION", "silu").lower()
    ENCODER_RES_BLOCKS = _env_int("SIMVQ_ENCODER_RES_BLOCKS", 4)
    DECODER_RES_BLOCKS = _env_int("SIMVQ_DECODER_RES_BLOCKS", 4)
    UPSAMPLE_MODE = _env_str("SIMVQ_UPSAMPLE_MODE", "bilinear").lower()
    USE_CASCADE_DOWNSAMPLE = _env_bool("SIMVQ_USE_CASCADE_DOWNSAMPLE", False)
    USE_BOTTLENECK_ATTENTION = _env_bool("SIMVQ_USE_BOTTLENECK_ATTENTION", True)
    BOTTLENECK_ATTENTION_BLOCKS = _env_int("SIMVQ_BOTTLENECK_ATTENTION_BLOCKS", 1)
    SKIP_DROPOUT_P_INIT = [0.0]
    SKIP_DROPOUT_P_FINAL = [0.0]
    USE_SWINIR_ENHANCE = False
    SWINIR_ENHANCE_BLOCKS = _env_int("SIMVQ_SWINIR_ENHANCE_BLOCKS", 4)
    USE_SWIN_BACKBONE = False

    # New RAQ architecture.  A full profile conditions one unified generator.
    USE_RAQ = TRAIN_STAGE != "src_teacher"
    RAQ_RECON_GRAD_MODE = "dual"
    RAQ_GENERATOR_TYPE = "variable_rate_residual"
    USE_DYNAMIC_RAQ_RVQ = False
    RAQ_ROUTED_SRC_ENABLED = False
    RATE_CONDITIONING = _env_bool("SIMVQ_RAQ_RATE_CONDITIONING", True)
    USE_RATE_CONDITIONING = RATE_CONDITIONING
    RATE_EMBED_DIM = _env_int("SIMVQ_RAQ_RATE_EMBED_DIM", 64)
    RATE_HIDDEN_DIM = _env_int("SIMVQ_RAQ_RATE_HIDDEN_DIM", 128)
    FILM_INIT_IDENTITY = _env_bool("SIMVQ_RAQ_FILM_INIT_IDENTITY", True)
    RAQ_TRANSFORMER_DIM = _env_int("SIMVQ_RAQ_TRANSFORMER_DIM", 256)
    RAQ_GENERATOR_MODEL_DIMS = _env_int_list(
        "SIMVQ_RAQ_GENERATOR_MODEL_DIMS", [RAQ_TRANSFORMER_DIM, RAQ_TRANSFORMER_DIM]
    )
    RAQ_GENERATOR_ATTENTION_DIM = _env_int("SIMVQ_RAQ_GENERATOR_ATTENTION_DIM", 64)
    RAQ_TRANSFORMER_HEADS = _env_int("SIMVQ_RAQ_TRANSFORMER_HEADS", 8)
    RAQ_TRANSFORMER_LAYERS = _env_int("SIMVQ_RAQ_TRANSFORMER_LAYERS", 2)
    RAQ_TRANSFORMER_DROPOUT = _env_float("SIMVQ_RAQ_TRANSFORMER_DROPOUT", 0.0)
    RAQ_GENERATOR_FEEDFORWARD_MULTIPLIER = _env_float(
        "SIMVQ_RAQ_GENERATOR_FEEDFORWARD_MULTIPLIER", 4.0
    )

    # Reconstruction and auxiliary loss defaults.
    MSE_LOSS_WEIGHT = _env_float("SIMVQ_RAQ_RECON_MSE_WEIGHT", 1.0)
    MS_SSIM_LOSS_WEIGHT = _env_float("SIMVQ_RAQ_RECON_MS_SSIM_WEIGHT", 0.0)
    LPIPS_LOSS_WEIGHT = _env_float("SIMVQ_RAQ_RECON_LPIPS_WEIGHT", 0.0)
    RAQ_VQ_WEIGHT = _env_float("SIMVQ_RAQ_VQ_WEIGHT", 1.0)
    VQ_WEIGHT = RAQ_VQ_WEIGHT
    RAQ_LAYER_VQ_WEIGHTS = _env_float_list("SIMVQ_RAQ_LAYER_VQ_WEIGHTS", [0.25, 0.5])
    LAYER_VQ_WEIGHTS = RAQ_LAYER_VQ_WEIGHTS
    LAYER_LOSS_WEIGHTS_INIT = list(RAQ_LAYER_VQ_WEIGHTS)
    LAYER_LOSS_WEIGHTS_FINAL = _env_float_list(
        "SIMVQ_RAQ_LAYER_VQ_WEIGHTS_FINAL", RAQ_LAYER_VQ_WEIGHTS
    )

    OUTPUT_DISTILL_WEIGHT_LOW = _env_float(
        "SIMVQ_RAQ_OUTPUT_DISTILL_WEIGHT_LOW",
        0.02,
        aliases=("SIMVQ_RAQ_OUTPUT_DISTILL_LOW", "SIMVQ_RAQ_OUTPUT_DISTILL_LOW_WEIGHT"),
    )
    OUTPUT_DISTILL_WEIGHT_HIGH = _env_float(
        "SIMVQ_RAQ_OUTPUT_DISTILL_WEIGHT_HIGH",
        0.20,
        aliases=("SIMVQ_RAQ_OUTPUT_DISTILL_HIGH", "SIMVQ_RAQ_OUTPUT_DISTILL_HIGH_WEIGHT"),
    )
    OUTPUT_DISTILL_GAMMA = _env_float("SIMVQ_RAQ_OUTPUT_DISTILL_GAMMA", 2.0)
    FEATURE_DISTILL_WEIGHT_LOW = _env_float(
        "SIMVQ_RAQ_FEATURE_DISTILL_WEIGHT_LOW",
        0.01,
        aliases=("SIMVQ_RAQ_FEATURE_DISTILL_LOW", "SIMVQ_RAQ_FEATURE_DISTILL_LOW_WEIGHT"),
    )
    FEATURE_DISTILL_WEIGHT_HIGH = _env_float(
        "SIMVQ_RAQ_FEATURE_DISTILL_WEIGHT_HIGH",
        0.10,
        aliases=("SIMVQ_RAQ_FEATURE_DISTILL_HIGH", "SIMVQ_RAQ_FEATURE_DISTILL_HIGH_WEIGHT"),
    )
    FEATURE_DISTILL_GAMMA = _env_float("SIMVQ_RAQ_FEATURE_DISTILL_GAMMA", 2.0)
    FEATURE_LAYER_WEIGHTS = _env_float_list(
        "SIMVQ_RAQ_FEATURE_LAYER_WEIGHTS", [1.0, 1.0]
    )
    IDENTITY_WEIGHT = _env_float("SIMVQ_RAQ_IDENTITY_WEIGHT", 1.0)
    HIERARCHY_WEIGHT = _env_float("SIMVQ_RAQ_HIERARCHY_WEIGHT", 0.05)
    DIVERSITY_WEIGHT = _env_float("SIMVQ_RAQ_DIVERSITY_WEIGHT", 0.01)
    DIVERSITY_MARGIN = _env_float("SIMVQ_RAQ_DIVERSITY_MARGIN", 0.5)
    DIVERSITY_NUM_PAIRS = _env_int("SIMVQ_RAQ_DIVERSITY_NUM_PAIRS", 4096)

    # Optimizer and five-stage learning rates.
    LEARNING_RATE_G = _env_float("SIMVQ_LEARNING_RATE_G", 5e-5)
    LEARNING_RATE = LEARNING_RATE_G
    WEIGHT_DECAY = _env_float("SIMVQ_RAQ_WEIGHT_DECAY", 0.0)
    BETAS = (
        _env_float("SIMVQ_RAQ_ADAM_BETA1", 0.5),
        _env_float("SIMVQ_RAQ_ADAM_BETA2", 0.999),
    )
    GENERATOR_LR = _env_float("SIMVQ_RAQ_GENERATOR_LR", 2e-4)
    RATE_MODULE_LR = _env_float("SIMVQ_RAQ_RATE_MODULE_LR", 1e-4)
    FILM_LR = _env_float("SIMVQ_RAQ_FILM_LR", 1e-4)
    DECODER_LR = _env_float("SIMVQ_RAQ_DECODER_LR", 1e-5)
    ENCODER_LR = _env_float("SIMVQ_RAQ_ENCODER_LR", 1e-6)
    STAGE1_SRC_LR = _env_float("SIMVQ_RAQ_STAGE1_SRC_LR", 5e-5)
    STAGE2_RAQ_LR = _env_float("SIMVQ_RAQ_STAGE2_RAQ_LR", 2e-4)
    STAGE3_RAQ_LR = _env_float("SIMVQ_RAQ_STAGE3_RAQ_LR", 1e-4)
    STAGE4_RAQ_LR = _env_float("SIMVQ_RAQ_STAGE4_RAQ_LR", 5e-5)
    STAGE4_DECODER_LR = _env_float("SIMVQ_RAQ_STAGE4_DECODER_LR", 1e-5)
    STAGE4_ENCODER_LR = _env_float("SIMVQ_RAQ_STAGE4_ENCODER_LR", 1e-6)
    STAGE5_RAQ_LR = _env_float("SIMVQ_RAQ_STAGE5_RAQ_LR", 2e-5)
    STAGE5_DECODER_LR = _env_float("SIMVQ_RAQ_STAGE5_DECODER_LR", 5e-6)
    STAGE5_ENCODER_LR = _env_float("SIMVQ_RAQ_STAGE5_ENCODER_LR", 5e-7)
    TRAIN_ENCODER_STAGE4 = _env_bool("SIMVQ_RAQ_STAGE4_TRAIN_ENCODER", False)
    TRAIN_ENCODER_STAGE5 = _env_bool("SIMVQ_RAQ_STAGE5_TRAIN_ENCODER", False)
    DECODER_TAIL_BLOCKS = _env_int("SIMVQ_RAQ_DECODER_TAIL_BLOCKS", 1)
    ENCODER_TAIL_BLOCKS = _env_int("SIMVQ_RAQ_ENCODER_TAIL_BLOCKS", 1)
    GRAD_CLIP_NORM = _env_float("SIMVQ_RAQ_GRAD_CLIP_NORM", 1.0)
    AMP_ENABLED = _env_bool("SIMVQ_RAQ_AMP", True)

    # Batch/data parameters.  Multiple profiles are run sequentially and their
    # losses must be divided by profiles-per-step and accumulation steps.
    TOTAL_BATCH_SIZE = _env_int("SIMVQ_TOTAL_BATCH_SIZE", 24)
    MICRO_BATCH_SIZE = _env_int("SIMVQ_MICRO_BATCH_SIZE", 8)
    GRADIENT_ACCUMULATION_STEPS = TOTAL_BATCH_SIZE // MICRO_BATCH_SIZE
    NUM_WORKERS = _env_int("SIMVQ_NUM_WORKERS", 8)
    PIN_MEMORY = _env_bool("SIMVQ_PIN_MEMORY", True)
    LOG_INTERVAL = _env_int("SIMVQ_RAQ_LOG_INTERVAL", 25)
    VAL_INTERVAL = _env_int("SIMVQ_RAQ_VAL_INTERVAL", 1)
    VAL_MAX_BATCHES = _env_int("SIMVQ_RAQ_VAL_MAX_BATCHES", 32)
    # Non-zero is intended for bounded integration/smoke runs.  Formal scripts
    # leave it at zero and therefore consume the full training dataloader.
    TRAIN_MAX_BATCHES = _env_int("SIMVQ_RAQ_TRAIN_MAX_BATCHES", 0)
    SAVE_EVERY = _env_int("SIMVQ_RAQ_SAVE_EVERY", 10)
    TRAIN_RESIZE = _env_size("SIMVQ_TRAIN_RESIZE", (256, 256))
    VAL_RESIZE = _env_size("SIMVQ_VAL_RESIZE", (256, 256))
    TRAIN_DATASET_PATH = _env_str(
        "SIMVQ_TRAIN_DATASET_PATH", "/workspace/yi/work/Cars196/train_data"
    )
    VAL_DATASET_PATH = _env_str(
        "SIMVQ_VAL_DATASET_PATH", "/workspace/yi/work/Cars196/val_data"
    )
    TEST_DATASET_PATH = _env_str(
        "SIMVQ_TEST_DATASET_PATH", "/workspace/yi/work/Kodak-256-transform-resize"
    )
    STAGE_EPOCHS = {
        1: _env_int("SIMVQ_RAQ_STAGE1_EPOCHS", 200),
        2: _env_int("SIMVQ_RAQ_STAGE2_EPOCHS", 20),
        3: _env_int("SIMVQ_RAQ_STAGE3_EPOCHS", 120),
        4: _env_int("SIMVQ_RAQ_STAGE4_EPOCHS", 40),
        5: _env_int("SIMVQ_RAQ_STAGE5_EPOCHS", 40),
    }
    NUM_EPOCHS = _env_int("NUM_EPOCHS", STAGE_EPOCHS[STAGE_INDEX])

    # Checkpoints, output paths, and resume metadata.
    SRC_TEACHER_CHECKPOINT = _env_str("SIMVQ_SRC_TEACHER_CHECKPOINT", "")
    TEACHER_CHECKPOINT = SRC_TEACHER_CHECKPOINT
    STUDENT_CHECKPOINT = _env_str(
        "SIMVQ_RAQ_STUDENT_CHECKPOINT",
        "",
        aliases=("SIMVQ_STUDENT_CHECKPOINT",),
    )
    RESUME = _env_bool("SIMVQ_RESUME", False)
    RESUME_PATH = _env_str("SIMVQ_RESUME_PATH", STUDENT_CHECKPOINT)
    EXPERIMENT_FAMILY = _env_str(
        "SIMVQ_EXP_FAMILY", "single_teacher_variable_rate_raq"
    )
    EXPERIMENT_NAME = _env_str(
        "SIMVQ_EXPERIMENT_NAME", f"{EXPERIMENT_FAMILY}_{TRAIN_STAGE}"
    )
    CHECKPOINT_DIR = _env_str(
        "SIMVQ_CHECKPOINT_DIR", f"./checkpoints/{EXPERIMENT_NAME}"
    )
    LOG_DIR = _env_str(
        "SIMVQ_LOG_DIR", f"./experiments/tensorboard/{EXPERIMENT_NAME}"
    )
    METRICS_PATH = _env_str(
        "SIMVQ_METRICS_PATH", f"./experiments/{EXPERIMENT_NAME}_epoch_metrics.csv"
    )
    PROFILE_METRICS_PATH = _env_str(
        "SIMVQ_RAQ_PROFILE_METRICS_PATH",
        f"./experiments/{EXPERIMENT_NAME}_profile_metrics.csv",
    )
    CODEBOOK_METRICS_PATH = _env_str(
        "SIMVQ_CODEBOOK_METRICS_PATH",
        f"./experiments/{EXPERIMENT_NAME}_codebook_metrics.csv",
    )
    SNAPSHOT_DIR = _env_str(
        "SIMVQ_SNAPSHOT_DIR", f"./experiments/snapshots/{EXPERIMENT_NAME}"
    )

    # Validation aggregation and the hard maximum-rate protection gate.
    VAL_AVERAGE_WEIGHT = _env_float("SIMVQ_RAQ_VAL_AVERAGE_WEIGHT", 0.8)
    VAL_WORST_WEIGHT = _env_float("SIMVQ_RAQ_VAL_WORST_WEIGHT", 0.2)
    VAL_PROFILE_WEIGHTS = _parse_profile_weights(
        _env_str("SIMVQ_RAQ_VAL_PROFILE_WEIGHTS", ""),
        supported_k=SUPPORTED_K_VALUES,
    )
    VAL_REQUIRE_MAX_PROTECTION = _env_bool(
        "SIMVQ_RAQ_VAL_REQUIRE_MAX_PROTECTION", True
    )
    VAL_MAX_PSNR_DROP_DB = _env_float("SIMVQ_RAQ_VAL_MAX_PSNR_DROP_DB", 0.30)
    VAL_SCORE_METRIC = _env_str("SIMVQ_RAQ_VAL_SCORE_METRIC", "psnr").lower()

    # Channel is deliberately disabled before stage 5.
    CHANNEL_TYPE = _env_str("SIMVQ_CHANNEL_TYPE", "AWGN").upper()
    CHANNEL_ENABLED = TRAIN_STAGE == "channel_finetune"
    SNR_RANGE_DB = _env_float_list("SIMVQ_SNR_RANGE_DB", [0.0, 15.0])
    CHANNEL_CODING_RATE_TRAIN = _env_float("SIMVQ_CHANNEL_CODING_RATE_TRAIN", 0.5)
    CHANNEL_CODING_RATE_VAL = _env_float("SIMVQ_CHANNEL_CODING_RATE_VAL", 0.5)
    BLOCK_LENGTH = _env_int("SIMVQ_BLOCK_LENGTH", 256)
    RICIAN_K_FACTOR = _env_float("SIMVQ_RICIAN_K_FACTOR", 10.0)
    CHANNEL_RAMP_EPOCHS = _env_int("SIMVQ_RAQ_CHANNEL_RAMP_EPOCHS", 10)
    CHANNEL_PROB_START = _env_float("SIMVQ_RAQ_CHANNEL_PROB_START", 0.0)
    CHANNEL_PROB_END = _env_float("SIMVQ_RAQ_CHANNEL_PROB_END", 1.0)

    # Exact lowercase constructor names consumed by VariableRateDeepSC.from_config.
    # Keeping both spellings also makes checkpoint metadata self-describing.
    rate_embedding_dim = RATE_EMBED_DIM
    rate_hidden_dim = RATE_HIDDEN_DIM
    generator_model_dims = list(RAQ_GENERATOR_MODEL_DIMS)
    generator_attention_dim = RAQ_GENERATOR_ATTENTION_DIM
    generator_transformer_depth = RAQ_TRANSFORMER_LAYERS
    generator_transformer_heads = RAQ_TRANSFORMER_HEADS
    generator_feedforward_multiplier = RAQ_GENERATOR_FEEDFORWARD_MULTIPLIER
    generator_dropout = RAQ_TRANSFORMER_DROPOUT
    minimum_target_size = min(SUPPORTED_K_VALUES)
    allowed_target_sizes = [list(SUPPORTED_K_VALUES), list(SUPPORTED_K_VALUES)]
    freeze_source_codebooks = STAGE_INDEX >= 2
    channel_prob = 1.0 if CHANNEL_ENABLED else 0.0

    DEVICE = _env_str(
        "SIMVQ_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    MODEL_PARALLEL = False
    ENCODER_DEVICE = DEVICE
    DECODER_DEVICE = DEVICE

    @classmethod
    def rate_score(cls, profile: Sequence[int]) -> float:
        """Return normalized mean log-rate: 0 at 2x2 and 1 at 2048x2048."""

        k0, k1 = parse_profile(profile, supported_k=cls.SUPPORTED_K_VALUES)
        min_bits = math.log2(min(cls.SUPPORTED_K_VALUES))
        max_bits = math.log2(max(cls.SUPPORTED_K_VALUES))
        return (
            ((math.log2(k0) - min_bits) + (math.log2(k1) - min_bits))
            / (2.0 * (max_bits - min_bits))
        )

    @classmethod
    def distillation_weight(cls, profile: Sequence[int], *, feature: bool = False) -> float:
        score = cls.rate_score(profile)
        if feature:
            low, high, gamma = (
                cls.FEATURE_DISTILL_WEIGHT_LOW,
                cls.FEATURE_DISTILL_WEIGHT_HIGH,
                cls.FEATURE_DISTILL_GAMMA,
            )
        else:
            low, high, gamma = (
                cls.OUTPUT_DISTILL_WEIGHT_LOW,
                cls.OUTPUT_DISTILL_WEIGHT_HIGH,
                cls.OUTPUT_DISTILL_GAMMA,
            )
        return low + (high - low) * (score ** gamma)

    @classmethod
    def validate(
        cls,
        *,
        require_checkpoint_paths: bool = False,
        require_dataset_paths: bool = False,
    ) -> None:
        """Fail fast on unsafe or internally inconsistent experiment settings."""

        if cls.UNET_DEPTH != 2 or cls.NUM_DOWNSAMPLE_BLOCKS != 2:
            raise ValueError("single-teacher variable-rate RAQ requires exactly two layers")
        if cls.NUM_EMBEDDINGS_LIST != [2048, 2048]:
            raise ValueError("the only complete SRC teacher must use [2048, 2048]")
        for name, values in {
            "EMBEDDING_DIM_LIST": cls.EMBEDDING_DIM_LIST,
            "DOWNSAMPLE_STRIDES": cls.DOWNSAMPLE_STRIDES,
            "RAQ_LAYER_VQ_WEIGHTS": cls.RAQ_LAYER_VQ_WEIGHTS,
            "LAYER_LOSS_WEIGHTS_FINAL": cls.LAYER_LOSS_WEIGHTS_FINAL,
            "FEATURE_LAYER_WEIGHTS": cls.FEATURE_LAYER_WEIGHTS,
            "RAQ_GENERATOR_MODEL_DIMS": cls.RAQ_GENERATOR_MODEL_DIMS,
        }.items():
            if len(values) != 2:
                raise ValueError(f"{name} must contain exactly two entries")
        if any(dim <= 0 for dim in cls.EMBEDDING_DIM_LIST):
            raise ValueError("EMBEDDING_DIM_LIST entries must be positive")
        if any(stride <= 0 for stride in cls.DOWNSAMPLE_STRIDES):
            raise ValueError("DOWNSAMPLE_STRIDES entries must be positive")
        if cls.RAQ_RECON_GRAD_MODE != "dual":
            raise ValueError("variable-rate RAQ training requires dual reconstruction gradients")
        if cls.SANDWICH_NUM_RANDOM < 0:
            raise ValueError("SIMVQ_RAQ_SANDWICH_NUM_RANDOM must be non-negative")
        if cls.STAGE_INDEX == 2 and cls.MIN_PROFILE == cls.MAX_PROFILE:
            raise ValueError(
                "Stage 2 cannot train only the [2048,2048] hard bypass; "
                "SIMVQ_RAQ_MIN_PROFILE must be a non-maximum near-max profile"
            )
        if cls.TOTAL_BATCH_SIZE <= 0 or cls.MICRO_BATCH_SIZE <= 0:
            raise ValueError("batch sizes must be positive")
        if cls.TOTAL_BATCH_SIZE % cls.MICRO_BATCH_SIZE != 0:
            raise ValueError("SIMVQ_TOTAL_BATCH_SIZE must be divisible by SIMVQ_MICRO_BATCH_SIZE")
        if cls.NUM_EPOCHS <= 0 or cls.NUM_WORKERS < 0:
            raise ValueError("NUM_EPOCHS must be positive and NUM_WORKERS non-negative")
        if cls.LOG_INTERVAL <= 0 or cls.VAL_INTERVAL <= 0 or cls.VAL_MAX_BATCHES <= 0:
            raise ValueError("logging/validation intervals and max batches must be positive")
        if cls.TRAIN_MAX_BATCHES < 0:
            raise ValueError("SIMVQ_RAQ_TRAIN_MAX_BATCHES must be non-negative")
        if cls.SAVE_EVERY < 0:
            raise ValueError("SIMVQ_RAQ_SAVE_EVERY must be non-negative")
        if cls.DECODER_TAIL_BLOCKS < 0 or cls.ENCODER_TAIL_BLOCKS < 0:
            raise ValueError("tail block counts must be non-negative")
        if cls.GRAD_CLIP_NORM < 0:
            raise ValueError("SIMVQ_RAQ_GRAD_CLIP_NORM must be non-negative")
        if not cls.RATE_CONDITIONING:
            raise ValueError("the variable-rate student requires full-profile rate conditioning")
        if not cls.FILM_INIT_IDENTITY:
            raise ValueError("rate FiLM adapters must use identity initialization")
        if cls.QUANTIZER_TYPE != "simvq":
            raise ValueError("single-teacher variable-rate RAQ currently requires SimVQ")
        if cls.RATE_EMBED_DIM <= 0 or cls.RATE_HIDDEN_DIM <= 0:
            raise ValueError("rate embedding dimensions must be positive")
        if cls.RAQ_TRANSFORMER_HEADS <= 0 or cls.RAQ_TRANSFORMER_LAYERS <= 0:
            raise ValueError("Transformer heads and layers must be positive")
        if any(dim <= 0 or dim % cls.RAQ_TRANSFORMER_HEADS != 0 for dim in cls.RAQ_GENERATOR_MODEL_DIMS):
            raise ValueError("each generator model dimension must be positive and divisible by heads")
        if cls.RAQ_GENERATOR_ATTENTION_DIM <= 0:
            raise ValueError("SIMVQ_RAQ_GENERATOR_ATTENTION_DIM must be positive")
        if cls.OUTPUT_DISTILL_GAMMA <= 0 or cls.FEATURE_DISTILL_GAMMA <= 0:
            raise ValueError("distillation gamma values must be positive")
        if cls.OUTPUT_DISTILL_WEIGHT_LOW > cls.OUTPUT_DISTILL_WEIGHT_HIGH:
            raise ValueError("output distillation low weight cannot exceed high weight")
        if cls.FEATURE_DISTILL_WEIGHT_LOW > cls.FEATURE_DISTILL_WEIGHT_HIGH:
            raise ValueError("feature distillation low weight cannot exceed high weight")
        nonnegative = {
            "RAQ_VQ_WEIGHT": cls.RAQ_VQ_WEIGHT,
            "OUTPUT_DISTILL_WEIGHT_LOW": cls.OUTPUT_DISTILL_WEIGHT_LOW,
            "FEATURE_DISTILL_WEIGHT_LOW": cls.FEATURE_DISTILL_WEIGHT_LOW,
            "IDENTITY_WEIGHT": cls.IDENTITY_WEIGHT,
            "HIERARCHY_WEIGHT": cls.HIERARCHY_WEIGHT,
            "DIVERSITY_WEIGHT": cls.DIVERSITY_WEIGHT,
            "DIVERSITY_MARGIN": cls.DIVERSITY_MARGIN,
            "VAL_MAX_PSNR_DROP_DB": cls.VAL_MAX_PSNR_DROP_DB,
            "MSE_LOSS_WEIGHT": cls.MSE_LOSS_WEIGHT,
            "MS_SSIM_LOSS_WEIGHT": cls.MS_SSIM_LOSS_WEIGHT,
            "LPIPS_LOSS_WEIGHT": cls.LPIPS_LOSS_WEIGHT,
            "WEIGHT_DECAY": cls.WEIGHT_DECAY,
        }
        nonnegative.update(
            {f"RAQ_LAYER_VQ_WEIGHTS[{i}]": value for i, value in enumerate(cls.RAQ_LAYER_VQ_WEIGHTS)}
        )
        nonnegative.update(
            {f"LAYER_LOSS_WEIGHTS_FINAL[{i}]": value for i, value in enumerate(cls.LAYER_LOSS_WEIGHTS_FINAL)}
        )
        nonnegative.update(
            {f"FEATURE_LAYER_WEIGHTS[{i}]": value for i, value in enumerate(cls.FEATURE_LAYER_WEIGHTS)}
        )
        for name, value in nonnegative.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        learning_rates = {
            name: value
            for name, value in vars(cls).items()
            if name.endswith("_LR") and isinstance(value, (int, float))
        }
        if any(value <= 0 for value in learning_rates.values()):
            raise ValueError("all configured learning rates must be positive")
        if cls.DIVERSITY_NUM_PAIRS < 0:
            raise ValueError("SIMVQ_RAQ_DIVERSITY_NUM_PAIRS must be non-negative")
        if cls.VAL_AVERAGE_WEIGHT < 0 or cls.VAL_WORST_WEIGHT < 0:
            raise ValueError("validation aggregate weights must be non-negative")
        if cls.VAL_AVERAGE_WEIGHT + cls.VAL_WORST_WEIGHT <= 0:
            raise ValueError("at least one validation aggregate weight must be positive")
        if cls.VAL_SCORE_METRIC != "psnr":
            raise ValueError("the implemented multi-profile checkpoint score is PSNR-based")
        if cls.MAX_PROFILE not in cls.VAL_PROFILES:
            raise ValueError("fixed validation profiles must include [2048,2048]")
        if cls.CHANNEL_RAMP_EPOCHS <= 0:
            raise ValueError("SIMVQ_RAQ_CHANNEL_RAMP_EPOCHS must be positive")
        if not 0.0 <= cls.CHANNEL_PROB_START <= cls.CHANNEL_PROB_END <= 1.0:
            raise ValueError("channel probability ramp must satisfy 0 <= start <= end <= 1")
        if len(cls.SNR_RANGE_DB) != 2 or cls.SNR_RANGE_DB[0] > cls.SNR_RANGE_DB[1]:
            raise ValueError("SIMVQ_SNR_RANGE_DB must contain an ordered [low, high] pair")
        if cls.CHANNEL_TYPE != "AWGN":
            raise ValueError("the current finite-blocklength channel stage supports AWGN only")
        invalid_val_weights = set(cls.VAL_PROFILE_WEIGHTS) - set(cls.VAL_PROFILES)
        if invalid_val_weights:
            raise ValueError(
                f"validation weights reference profiles not in VAL_PROFILES: {invalid_val_weights}"
            )
        if any(weight <= 0 for weight in cls.VAL_PROFILE_WEIGHTS.values()):
            raise ValueError("explicit validation profile weights must be positive")

        if require_checkpoint_paths:
            if cls.STAGE_INDEX >= 2:
                if not cls.SRC_TEACHER_CHECKPOINT:
                    raise ValueError("SIMVQ_SRC_TEACHER_CHECKPOINT is required for stages 2-5")
                if not Path(cls.SRC_TEACHER_CHECKPOINT).is_file():
                    raise FileNotFoundError(cls.SRC_TEACHER_CHECKPOINT)
            if cls.STAGE_INDEX >= 3:
                if not cls.STUDENT_CHECKPOINT:
                    raise ValueError("SIMVQ_RAQ_STUDENT_CHECKPOINT is required for stages 3-5")
                if not Path(cls.STUDENT_CHECKPOINT).is_file():
                    raise FileNotFoundError(cls.STUDENT_CHECKPOINT)
        if require_dataset_paths:
            for name in ("TRAIN_DATASET_PATH", "VAL_DATASET_PATH"):
                path = Path(getattr(cls, name))
                if not path.is_dir():
                    raise FileNotFoundError(f"{name}: {path}")

    @classmethod
    def as_metadata(cls) -> Mapping[str, object]:
        """Stable checkpoint metadata for architecture/profile compatibility checks."""

        return {
            "schema": "single_teacher_variable_rate_raq_v1",
            "stage": cls.TRAIN_STAGE,
            "supported_k_values": list(cls.SUPPORTED_K_VALUES),
            "target_profiles": [list(profile) for profile in cls.TARGET_PROFILES],
            "validation_profiles": [list(profile) for profile in cls.VAL_PROFILES],
            "min_profile": list(cls.MIN_PROFILE),
            "max_profile": list(cls.MAX_PROFILE),
            "source_codebooks": list(cls.NUM_EMBEDDINGS_LIST),
            "embedding_dims": list(cls.EMBEDDING_DIM_LIST),
            "rate_conditioning": cls.RATE_CONDITIONING,
            "reconstruction_gradient_mode": cls.RAQ_RECON_GRAD_MODE,
        }


# Semantic validation is safe at import time.  Filesystem requirements are
# checked explicitly by the stage runner immediately before training.
VariableRateConfig.validate()


__all__ = [
    "SUPPORTED_K_VALUES",
    "ALL_PROFILES",
    "DEFAULT_VAL_PROFILES",
    "VariableRateConfig",
    "parse_profile",
    "parse_profiles",
]
