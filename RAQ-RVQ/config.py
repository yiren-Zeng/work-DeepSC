"""Configuration for the copied independent RAQ-RVQ experiment."""

import json
import math
import os

import torch

from utils.raq_rvq import validate_independent_rvq_k_lists


def _env_int(name, default):
    value = os.environ.get(name)
    return int(value) if value else default


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


def _env_str(name, default):
    return os.environ.get(name) or default


def _env_int_list(name, default):
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _env_nested_int_lists_optional(name):
    value = os.environ.get(name)
    if not value:
        return None
    value = value.strip()
    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list) or any(
            not isinstance(row, list) for row in parsed
        ):
            raise ValueError(f"{name} must be a nested integer list")
        return [[int(item) for item in row] for row in parsed]
    return [
        [int(item.strip()) for item in row.split(",") if item.strip()]
        for row in value.split(";")
        if row.strip()
    ]


def _source_bpp(strides, codebook_sizes):
    bpp = 0.0
    cumulative_stride = 1
    for stride, codebook_size in zip(strides, codebook_sizes):
        cumulative_stride *= stride
        bpp += math.log2(codebook_size) / (cumulative_stride ** 2)
    return bpp


def _embedding_dims(base_channels, depth):
    return [base_channels * (2 ** (index + 1)) for index in range(depth)]


def _initial_loss_weights(depth):
    return [0.25 * (index + 1) for index in range(depth)]


def _broadcast(values, depth):
    return [list(values) for _ in range(depth)]


def _products(nested_values):
    return [math.prod(values) for values in nested_values]


def _format_k_list(values):
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return f"k{unique_values[0]}"
    return "k" + "-".join(str(value) for value in values)


class Config:
    # Fixed model family; the remaining environment variables are the actual
    # per-run choices used by the two shell entry points.
    EXPERIMENT_STAGE = "B"
    EXPERIMENT_FAMILY = _env_str(
        "SIMVQ_EXP_FAMILY",
        "shiyan_independent_raq_rvq_src64-64_trg2-64_d2_curriculum_"
        "rate094_A_patch_ch256-512_res6-6",
    )

    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    UNET_DEPTH = _env_int("SIMVQ_UNET_DEPTH", 2)
    NUM_DOWNSAMPLE_BLOCKS = UNET_DEPTH
    BASE_CHANNELS = _env_int("SIMVQ_BASE_CHANNELS", 128)
    EMBEDDING_DIM_LIST = _embedding_dims(BASE_CHANNELS, UNET_DEPTH)
    NUM_EMBEDDINGS_LIST = _env_int_list(
        "SIMVQ_NUM_EMBEDDINGS_LIST", [64, 64]
    )
    DOWNSAMPLE_STRIDES = _env_int_list(
        "SIMVQ_DOWNSAMPLE_STRIDES", [8, 2]
    )

    RAQ_MIN_TRG = _env_int("SIMVQ_RAQ_MIN_TRG", 2)
    RAQ_MAX_TRG = _env_int("SIMVQ_RAQ_MAX_TRG", 64)
    RAQ_MIN_TRG_LIST = [RAQ_MIN_TRG] * UNET_DEPTH
    RAQ_MAX_TRG_LIST = [RAQ_MAX_TRG] * UNET_DEPTH

    INDEPENDENT_RAQ_RVQ_DEPTH = _env_int(
        "SIMVQ_INDEPENDENT_RAQ_RVQ_DEPTH", 2
    )
    INDEPENDENT_RAQ_RVQ_K_LISTS = _env_nested_int_lists_optional(
        "SIMVQ_INDEPENDENT_RAQ_RVQ_K_LISTS"
    )

    RAQ_USE_CURRICULUM = _env_int("SIMVQ_RAQ_USE_CURRICULUM", 0) == 1
    _CURRICULUM_DEFAULTS = {
        "EARLY": [32, 64],
        "MIDDLE": [8, 16, 32, 64],
        "LATE": [2, 4, 8, 16, 32, 64],
    }
    for _phase, _default in _CURRICULUM_DEFAULTS.items():
        _shared = _env_int_list(
            f"SIMVQ_RAQ_CURRICULUM_{_phase}_LIST", _default
        )
        locals()[f"RAQ_CURRICULUM_{_phase}_LISTS"] = _broadcast(
            _shared, UNET_DEPTH
        )
    del _phase, _default, _shared

    NORM_TYPE = "group"
    GROUP_NORM_GROUPS = 32
    ACTIVATION = "silu"
    ENCODER_RES_BLOCKS = _env_int("SIMVQ_ENCODER_RES_BLOCKS", 6)
    DECODER_RES_BLOCKS = _env_int("SIMVQ_DECODER_RES_BLOCKS", 6)
    UPSAMPLE_MODE = "bilinear"
    USE_CASCADE_DOWNSAMPLE = False
    COMMITMENT_COST = 0.25

    LAYER_LOSS_WEIGHTS_INIT = _initial_loss_weights(UNET_DEPTH)
    LAYER_LOSS_WEIGHTS_FINAL = [0.25] * UNET_DEPTH
    SKIP_DROPOUT_P_INIT = [0.1] * max(UNET_DEPTH - 1, 0)
    SKIP_DROPOUT_P_FINAL = [0.0] * max(UNET_DEPTH - 1, 0)
    MSE_LOSS_WEIGHT = 1.0
    PHASE1_END = 0.1
    PHASE2_END = 0.4

    LEARNING_RATE_G = _env_float("SIMVQ_LEARNING_RATE_G", 5e-5)
    CODEBOOK_PROJ_LR = _env_float("SIMVQ_CODEBOOK_PROJ_LR", 2e-4)
    LR_STEP_SIZE = _env_int("SIMVQ_LR_STEP_SIZE", 100)
    BETAS = (0.5, 0.999)

    CHANNEL_CODING_RATE_TRAIN = 0.5
    CHANNEL_CODING_RATE_VAL = 0.5
    BLOCK_LENGTH = 256
    SNR_RANGE_DB = [0, 15]
    CHANNEL_PROB_START_EPOCH = _env_int(
        "SIMVQ_CHANNEL_PROB_START_EPOCH", 80
    )
    CHANNEL_PROB_END_EPOCH = _env_int(
        "SIMVQ_CHANNEL_PROB_END_EPOCH", 120
    )

    TOTAL_BATCH_SIZE = _env_int("SIMVQ_TOTAL_BATCH_SIZE", 24)
    MICRO_BATCH_SIZE = _env_int("SIMVQ_MICRO_BATCH_SIZE", 24)
    NUM_WORKERS = _env_int("SIMVQ_NUM_WORKERS", 8)
    PIN_MEMORY = True
    TRAIN_DATASET_PATH = _env_str(
        "SIMVQ_TRAIN_DATASET_PATH", "/workspace/yi/work/Cars196/train_data"
    )
    VAL_DATASET_PATH = _env_str(
        "SIMVQ_VAL_DATASET_PATH", "/workspace/yi/work/Cars196/val_data"
    )
    TEST_DATASET_PATH = _env_str(
        "SIMVQ_TEST_DATASET_PATH",
        "/workspace/yi/work/Kodak-256-transform-resize",
    )

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    _STRIDE_NAME = "x".join(map(str, DOWNSAMPLE_STRIDES))
    EXPERIMENT_NAME = (
        f"{EXPERIMENT_FAMILY}_unet{UNET_DEPTH}_ds{_STRIDE_NAME}_"
        f"{_format_k_list(NUM_EMBEDDINGS_LIST)}"
    )
    ESTIMATED_SOURCE_BPP = _source_bpp(
        DOWNSAMPLE_STRIDES,
        NUM_EMBEDDINGS_LIST,
    )
    ESTIMATED_TEST_SOURCE_BPP = _source_bpp(
        DOWNSAMPLE_STRIDES,
        NUM_EMBEDDINGS_LIST,
    )
    _RAQ_EQUIVALENT_K = _products(
        INDEPENDENT_RAQ_RVQ_K_LISTS or [[1, 1]] * UNET_DEPTH
    )
    ESTIMATED_RAQ_TARGET_BPP = _source_bpp(
        DOWNSAMPLE_STRIDES,
        _RAQ_EQUIVALENT_K,
    )
    ESTIMATED_TEST_TRANSMISSION_RATIO = (
        ESTIMATED_RAQ_TARGET_BPP / (CHANNEL_CODING_RATE_VAL * 3)
    )

    CHECKPOINT_DIR = os.path.join("./checkpoints", EXPERIMENT_NAME)
    LOG_DIR = os.path.join("./experiments/tensorboard", EXPERIMENT_NAME)
    METRICS_PATH = os.path.join(
        "./experiments", f"{EXPERIMENT_NAME}_epoch_metrics.csv"
    )
    CODEBOOK_METRICS_PATH = os.path.join(
        "./experiments", f"{EXPERIMENT_NAME}_codebook_metrics.csv"
    )
    NUM_EPOCHS = _env_int("NUM_EPOCHS", 200)
    RESUME = _env_int("SIMVQ_RESUME", 0) == 1
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

    @classmethod
    def validate(cls):
        layer_lists = {
            "DOWNSAMPLE_STRIDES": cls.DOWNSAMPLE_STRIDES,
            "EMBEDDING_DIM_LIST": cls.EMBEDDING_DIM_LIST,
            "NUM_EMBEDDINGS_LIST": cls.NUM_EMBEDDINGS_LIST,
            "LAYER_LOSS_WEIGHTS_INIT": cls.LAYER_LOSS_WEIGHTS_INIT,
            "LAYER_LOSS_WEIGHTS_FINAL": cls.LAYER_LOSS_WEIGHTS_FINAL,
            "RAQ_MIN_TRG_LIST": cls.RAQ_MIN_TRG_LIST,
            "RAQ_MAX_TRG_LIST": cls.RAQ_MAX_TRG_LIST,
        }
        for name, values in layer_lists.items():
            if len(values) != cls.UNET_DEPTH:
                raise ValueError(
                    f"{name} length ({len(values)}) must equal "
                    f"UNET_DEPTH ({cls.UNET_DEPTH})"
                )
        if cls.INDEPENDENT_RAQ_RVQ_DEPTH != 2:
            raise ValueError("independent RAQ-RVQ depth must be 2")
        cls.INDEPENDENT_RAQ_RVQ_K_LISTS = validate_independent_rvq_k_lists(
            cls.INDEPENDENT_RAQ_RVQ_K_LISTS,
            num_scales=cls.UNET_DEPTH,
            rvq_depth=cls.INDEPENDENT_RAQ_RVQ_DEPTH,
            min_k=cls.RAQ_MIN_TRG_LIST,
            max_k=cls.RAQ_MAX_TRG_LIST,
        )
        for name in (
            "RAQ_CURRICULUM_EARLY_LISTS",
            "RAQ_CURRICULUM_MIDDLE_LISTS",
            "RAQ_CURRICULUM_LATE_LISTS",
        ):
            values_by_layer = getattr(cls, name)
            if len(values_by_layer) != cls.UNET_DEPTH:
                raise ValueError(
                    f"{name} length must equal UNET_DEPTH ({cls.UNET_DEPTH})"
                )
            for layer_index, values in enumerate(values_by_layer):
                if not values:
                    raise ValueError(f"{name} layer {layer_index} is empty")
                min_k = cls.RAQ_MIN_TRG_LIST[layer_index]
                max_k = cls.RAQ_MAX_TRG_LIST[layer_index]
                for value in values:
                    if (
                        value < min_k
                        or value > max_k
                        or value & (value - 1)
                    ):
                        raise ValueError(
                            f"{name} layer {layer_index} contains invalid "
                            f"K={value}; expected a power of two in "
                            f"[{min_k}, {max_k}]"
                        )
        if cls.TOTAL_BATCH_SIZE % cls.MICRO_BATCH_SIZE:
            raise ValueError(
                "SIMVQ_TOTAL_BATCH_SIZE must be divisible by "
                "SIMVQ_MICRO_BATCH_SIZE"
            )
        if cls.CHANNEL_PROB_END_EPOCH < cls.CHANNEL_PROB_START_EPOCH:
            raise ValueError(
                "channel probability end epoch must not precede start epoch"
            )
        if cls.LR_STEP_SIZE <= 0:
            raise ValueError("SIMVQ_LR_STEP_SIZE must be positive")

    @classmethod
    def architecture_summary(cls):
        return {
            "experiment_name": cls.EXPERIMENT_NAME,
            "unet_depth": cls.UNET_DEPTH,
            "downsample_strides": list(cls.DOWNSAMPLE_STRIDES),
            "total_downsample": math.prod(cls.DOWNSAMPLE_STRIDES),
            "embedding_dim_list": list(cls.EMBEDDING_DIM_LIST),
            "num_embeddings_list": list(cls.NUM_EMBEDDINGS_LIST),
            "independent_raq_rvq_depth": cls.INDEPENDENT_RAQ_RVQ_DEPTH,
            "independent_raq_rvq_k_lists": [
                list(stage_sizes)
                for stage_sizes in cls.INDEPENDENT_RAQ_RVQ_K_LISTS
            ],
        }
