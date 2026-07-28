# 大部分实验参数都从这里读取，并支持环境变量覆盖


import torch
import os
import math


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


# def _default_downsample_strides(depth):
#     """Return a conservative default downsampling plan for a U-Net depth."""
#     if depth < 1:
#         raise ValueError("UNET_DEPTH must be >= 1")
#     if depth == 1:
#         return [8]
#     # Keep the existing 2-layer behavior and append extra 2x stages for
#     # deeper variants. This keeps the source rate close to the 2-layer setup.
#     return [4] + [2] * (depth - 1)


def _default_embedding_dims(base_channels, depth):
    return [base_channels * (2 ** (i + 1)) for i in range(depth)]


def _expand_to_depth(value, depth, name):
    if isinstance(value, (int, float)):
        return [value for _ in range(depth)]
    expanded = list(value)
    if len(expanded) != depth:
        raise ValueError(f"{name} length ({len(expanded)}) must equal UNET_DEPTH ({depth})")
    return expanded


def _default_loss_weights_init(depth):
    return [0.25 * (i + 1) for i in range(depth)] # [0.25, 0.50, 0.75, ...]


def _default_loss_weights_final(depth):
    return [0.25 for _ in range(depth)] # [0.25, 0.25, 0.25, ...]


def _default_skip_dropout_init(depth):
    return [0.1 for _ in range(max(depth - 1, 0))] # [0.1, 0.1, ...]  (depth-1 个)


def _default_skip_dropout_final(depth):
    return [0.0 for _ in range(max(depth - 1, 0))] # [0.0, 0.0, ...]  (depth-1 个)


def _format_k_list(num_embeddings_list): 
    unique_values = sorted(set(num_embeddings_list))
    if len(unique_values) == 1:
        return f"k{unique_values[0]}"
    return "k" + "-".join(str(v) for v in num_embeddings_list) # [65536, 8192] → "k65536-8192"


def _format_depth_list(depth_list):
    return "d" + "-".join(str(v) for v in depth_list)


def _experiment_name(
    family,
    depth,
    strides,
    num_embeddings_list,
    quantizer_type=None,
    rq_depth_list=None,
):
    stride_part = "x".join(str(v) for v in strides) # [8,2] → "8x2"
    legacy_name = f"{family}_unet{depth}_ds{stride_part}_{_format_k_list(num_embeddings_list)}"
    if quantizer_type not in {"rq_ema", "residual_simvq"}:
        return legacy_name
    rq_depth_list = rq_depth_list or [1] * depth
    quantizer_part = (
        "rq_ema" if quantizer_type == "rq_ema" else "residual_simvq"
    )
    return (
        f"{family}_{quantizer_part}_unet{depth}_ds{stride_part}_"
        f"{_format_k_list(num_embeddings_list)}_{_format_depth_list(rq_depth_list)}"
    )


def _source_bpp(strides, num_embeddings_list, quantizer_axis_list=None,
                embedding_dim_list=None, image_size=(256, 256),
                rq_depth_list=None):
    bpp = 0.0
    cumulative_downsample = 1
    quantizer_axis_list = quantizer_axis_list or ["patch"] * len(num_embeddings_list)
    embedding_dim_list = embedding_dim_list or [None] * len(num_embeddings_list)
    rq_depth_list = rq_depth_list or [1] * len(num_embeddings_list)
    image_h, image_w = image_size
    for stride, codebook_size, axis, embedding_dim, rq_depth in zip(
        strides,
        num_embeddings_list,
        quantizer_axis_list,
        embedding_dim_list,
        rq_depth_list,
    ):
        cumulative_downsample *= stride
        bits = math.log2(codebook_size) * rq_depth
        if axis == "channel":
            token_count = embedding_dim
            bpp += token_count * bits / (image_h * image_w)
        else:
            bpp += bits / (cumulative_downsample ** 2)
    return bpp

# 环境变量读取
def _env_int(name, default):
    value = os.environ.get(name) # os.environ 是 dict，存所有环境变量
    return int(value) if value else default


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


def _env_str(name, default):
    value = os.environ.get(name)
    return str(value) if value else default


def _env_str_list(name, default):
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _env_int_list(name, default):
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()] # .split(",")，按照逗号切分字符串


def _env_float_list(name, default):
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _resize_tuple_from_env(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()] # .replace("x", ",") 把字符串里的小写字母 "x" 替换成逗号 ","
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as H,W, for example 256,256")
    return int(parts[0]), int(parts[1]) # tuple[int, int], 如(H, W)


def _default_cvq_codeword_shapes(strides): # CVQ 码字形状：每层下采样后特征图的空间维度，除非通过环境变量覆盖。对于 patch-wise CVQ，使用 None。 
    train_h, train_w = _resize_tuple_from_env("SIMVQ_TRAIN_RESIZE", (256, 256))
    shapes = []
    cumulative = 1
    for stride in strides:
        cumulative *= stride
        shapes.append((train_h // cumulative, train_w // cumulative))
    return shapes


def _env_shape_list(name, default): # 这个也是生成形状的
    value = os.environ.get(name)
    if not value:
        return list(default)
    shapes = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item or item in {"none", "patch"}:
            shapes.append(None)
            continue
        parts = [part.strip() for part in item.replace("x", " ").split() if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"{name} entries must be HxW, none, or patch")
        shapes.append((int(parts[0]), int(parts[1])))
    return shapes


def _stage_settings(stage):
    stage = (stage or "c").lower() # 默认阶段为 C，使用 lower() 以实现大写变小写
    settings = {
        "a": {
            "family": "quality_v2_A_curriculum",
            "norm_type": "batch",
            "activation": "prelu",
            "encoder_res_blocks": 1,
            "decoder_res_blocks": 1,
            "upsample_mode": "nearest",
            "use_cascade_downsample": False,
            "use_attention": False,
            "attention_blocks": 0,
            "mse_loss_weight": 1.0,
            "ms_ssim_loss_weight": 0.0,
            "phase1_end": 0.1,
            "phase2_end": 0.4,
        },
        "b": {
            "family": "quality_v2_B_backbone",
            "norm_type": "group",
            "activation": "silu",
            "encoder_res_blocks": 2,
            "decoder_res_blocks": 2,
            "upsample_mode": "bilinear",
            "use_cascade_downsample": False,
            "use_attention": False,
            "attention_blocks": 0,
            "mse_loss_weight": 1.0,
            "ms_ssim_loss_weight": 0.0,
            "phase1_end": 0.1,
            "phase2_end": 0.4,
        },
        "c": {
            "family": "quality_v2_C_full",
            "norm_type": "group",
            "activation": "silu",
            "encoder_res_blocks": 2,
            "decoder_res_blocks": 2,
            "upsample_mode": "bilinear",
            "use_cascade_downsample": False,
            "use_attention": True,
            "attention_blocks": 1,
            "mse_loss_weight": 1.0,
            "ms_ssim_loss_weight": 0.0,
            "phase1_end": 0.1,
            "phase2_end": 0.4,
        },
    }
    if stage not in settings:
        raise ValueError(f"Unknown SIMVQ_EXPERIMENT_STAGE={stage!r}; use A, B, or C")
    return settings[stage]


_STAGE = os.environ.get("SIMVQ_EXPERIMENT_STAGE", "C") # 读环境变量，默认 "C"
_STAGE_SETTINGS = _stage_settings(_STAGE) # 查字典，拿到对应配置


class Config:
    EXPERIMENT_STAGE = _STAGE
    # Allow env override of experiment family (for variant experiments)
    _FAMILY_OVERRIDE = _env_str("SIMVQ_EXP_FAMILY", "")
    EXPERIMENT_FAMILY = _FAMILY_OVERRIDE if _FAMILY_OVERRIDE else _STAGE_SETTINGS["family"]
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    # Change this number to switch the model between 2/3/4-layer U-Net variants.
    UNET_DEPTH = _env_int("SIMVQ_UNET_DEPTH", 2)
    NUM_DOWNSAMPLE_BLOCKS = UNET_DEPTH
    BASE_CHANNELS = _env_int("SIMVQ_BASE_CHANNELS", 64)
    EMBEDDING_DIM_LIST = _default_embedding_dims(BASE_CHANNELS, UNET_DEPTH)
    NUM_EMBEDDINGS_PER_LAYER = None
    NUM_EMBEDDINGS_LIST = _env_int_list("SIMVQ_NUM_EMBEDDINGS_LIST", [64, 256])
    COMMITMENT_COST = 0.25
    QUANTIZER_TYPE = _env_str("SIMVQ_QUANTIZER_TYPE", "simvq").lower()
    RQ_DEPTH_LIST = _env_int_list("SIMVQ_RQ_DEPTH_LIST", [1] * UNET_DEPTH)
    RQ_EMA_DECAY = _env_float("SIMVQ_RQ_EMA_DECAY", 0.99)
    RQ_RESTART_UNUSED_CODES = _env_int("SIMVQ_RQ_RESTART_UNUSED_CODES", 1) == 1
    RQ_SHARED_CODEBOOK = _env_int("SIMVQ_RQ_SHARED_CODEBOOK", 1) == 1
    VITVQ_QBRIDGE_TYPE = _env_str("SIMVQ_VITVQ_QBRIDGE_TYPE", "QBridgeNoCompress-S")
    VITVQ_EMB_NOGRAD = _env_int("SIMVQ_VITVQ_EMB_NOGRAD", 0) == 1
    DOWNSAMPLE_STRIDES = _env_int_list("SIMVQ_DOWNSAMPLE_STRIDES", [8, 2])
    QUANTIZER_AXIS_LIST = _env_str_list("SIMVQ_QUANTIZER_AXIS_LIST", ["patch"] * UNET_DEPTH) # 量化轴
    CVQ_CODEWORD_SHAPES = _env_shape_list(
        "SIMVQ_CVQ_CODEWORD_SHAPES", _default_cvq_codeword_shapes(DOWNSAMPLE_STRIDES)
    )
    NESTED_CHANNEL_DROPOUT_ALPHA = _env_float("SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA", 0.0)
    # The baseline's VQ term dominated its reconstruction loss during early training.
    _RQ_LAYER_LOSS_DEFAULT = [1.0] * UNET_DEPTH
    LAYER_LOSS_WEIGHTS_INIT = _env_float_list(
        "SIMVQ_LAYER_LOSS_WEIGHTS_INIT",
        _RQ_LAYER_LOSS_DEFAULT if QUANTIZER_TYPE == "rq_ema" else _default_loss_weights_init(UNET_DEPTH),
    )
    LAYER_LOSS_WEIGHTS_FINAL = _env_float_list(
        "SIMVQ_LAYER_LOSS_WEIGHTS_FINAL",
        _RQ_LAYER_LOSS_DEFAULT if QUANTIZER_TYPE == "rq_ema" else _default_loss_weights_final(UNET_DEPTH),
    )
    # Retain robustness training without discarding half of the high-resolution path.
    SKIP_DROPOUT_P_INIT = _default_skip_dropout_init(UNET_DEPTH)
    SKIP_DROPOUT_P_FINAL = _default_skip_dropout_final(UNET_DEPTH)
    PHASE1_END = _STAGE_SETTINGS["phase1_end"]
    PHASE2_END = _STAGE_SETTINGS["phase2_end"]
    LEARNING_RATE_G = 5e-5
    CODEBOOK_PROJ_LR = 2e-4
    BETAS = (0.5, 0.999)
    CHANNEL_CODING_RATE_TRAIN = 0.5
    CHANNEL_CODING_RATE_VAL = 0.5
    BLOCK_LENGTH = 256
    SNR_RANGE_DB = [0, 15]
    CHANNEL_TYPE = "AWGN"
    RICIAN_K_FACTOR = 10
    CHANNEL_PROB_START_EPOCH = _env_int("SIMVQ_CHANNEL_PROB_START_EPOCH", 80)
    CHANNEL_PROB_END_EPOCH = _env_int("SIMVQ_CHANNEL_PROB_END_EPOCH", 120)
    NORM_TYPE = _STAGE_SETTINGS["norm_type"]
    GROUP_NORM_GROUPS = 32
    ACTIVATION = _STAGE_SETTINGS["activation"]
    # Allow env override of res_blocks for variant experiments
    ENCODER_RES_BLOCKS = _env_int("SIMVQ_ENCODER_RES_BLOCKS", _STAGE_SETTINGS["encoder_res_blocks"])
    DECODER_RES_BLOCKS = _env_int("SIMVQ_DECODER_RES_BLOCKS", _STAGE_SETTINGS["decoder_res_blocks"])
    UPSAMPLE_MODE = _STAGE_SETTINGS["upsample_mode"]
    USE_CASCADE_DOWNSAMPLE = _STAGE_SETTINGS["use_cascade_downsample"]
    USE_BOTTLENECK_ATTENTION = _STAGE_SETTINGS["use_attention"]
    BOTTLENECK_ATTENTION_BLOCKS = _STAGE_SETTINGS["attention_blocks"]
    MSE_LOSS_WEIGHT = _STAGE_SETTINGS["mse_loss_weight"]
    MS_SSIM_LOSS_WEIGHT = _STAGE_SETTINGS["ms_ssim_loss_weight"]
    # LPIPS (VGG perceptual loss) weight - set via env var for variant experiments
    LPIPS_LOSS_WEIGHT = _env_float("SIMVQ_LPIPS_WEIGHT", 0.0)
    # SwinIR quality enhancement - set via env var
    USE_SWINIR_ENHANCE = _env_int("SIMVQ_USE_SWINIR_ENHANCE", 0) == 1
    SWINIR_ENHANCE_BLOCKS = _env_int("SIMVQ_SWINIR_ENHANCE_BLOCKS", 4)
    # Swin Transformer backbone - set via env var
    USE_SWIN_BACKBONE = _env_int("SIMVQ_USE_SWIN_BACKBONE", 0) == 1
    TOTAL_BATCH_SIZE = _env_int("SIMVQ_TOTAL_BATCH_SIZE", 24)
    MICRO_BATCH_SIZE = _env_int("SIMVQ_MICRO_BATCH_SIZE", 24)
    NUM_WORKERS = _env_int("SIMVQ_NUM_WORKERS", 8)
    PIN_MEMORY = True
    TRAIN_DATASET_PATH = _env_str("SIMVQ_TRAIN_DATASET_PATH", "/workspace/yi/work/Cars196/train_data")
    VAL_DATASET_PATH = _env_str("SIMVQ_VAL_DATASET_PATH", "/workspace/yi/work/Cars196/val_data")
    TEST_DATASET_PATH = _env_str("SIMVQ_TEST_DATASET_PATH", "/workspace/yi/work/Kodak")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PARALLEL = _env_int("SIMVQ_MODEL_PARALLEL", 0) == 1
    ENCODER_DEVICE = _env_str("SIMVQ_ENCODER_DEVICE", DEVICE)
    DECODER_DEVICE = _env_str("SIMVQ_DECODER_DEVICE", "cuda:1" if torch.cuda.device_count() > 1 else DEVICE)
    TRAIN_IMAGE_SIZE = _resize_tuple_from_env("SIMVQ_TRAIN_RESIZE", (256, 256))
    TEST_IMAGE_SIZE = _resize_tuple_from_env("SIMVQ_TEST_RESIZE", (768, 512))
    EXPERIMENT_NAME = _experiment_name(
        EXPERIMENT_FAMILY,
        UNET_DEPTH,
        DOWNSAMPLE_STRIDES,
        NUM_EMBEDDINGS_LIST,
        quantizer_type=QUANTIZER_TYPE,
        rq_depth_list=RQ_DEPTH_LIST,
    )
    ESTIMATED_SOURCE_BPP = _source_bpp(
        DOWNSAMPLE_STRIDES,
        NUM_EMBEDDINGS_LIST,
        QUANTIZER_AXIS_LIST,
        EMBEDDING_DIM_LIST,
        TRAIN_IMAGE_SIZE,
        RQ_DEPTH_LIST
        if QUANTIZER_TYPE in {"rq_ema", "residual_simvq"}
        else None,
    )
    ESTIMATED_SOURCE_BITS_PER_IMAGE = (
        ESTIMATED_SOURCE_BPP * TRAIN_IMAGE_SIZE[0] * TRAIN_IMAGE_SIZE[1]
    )
    ESTIMATED_TEST_SOURCE_BPP = _source_bpp(
        DOWNSAMPLE_STRIDES,
        NUM_EMBEDDINGS_LIST,
        QUANTIZER_AXIS_LIST,
        EMBEDDING_DIM_LIST,
        TEST_IMAGE_SIZE,
        RQ_DEPTH_LIST
        if QUANTIZER_TYPE in {"rq_ema", "residual_simvq"}
        else None,
    )
    ESTIMATED_TEST_TRANSMISSION_RATIO = (
        ESTIMATED_TEST_SOURCE_BPP / (CHANNEL_CODING_RATE_VAL * 1 * 3)
    )
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints", EXPERIMENT_NAME)
    LOG_DIR = os.path.join(PROJECT_ROOT, "experiments/tensorboard", EXPERIMENT_NAME)
    METRICS_PATH = os.path.join(PROJECT_ROOT, "experiments", f"{EXPERIMENT_NAME}_epoch_metrics.csv")
    CODEBOOK_METRICS_PATH = os.path.join(
        PROJECT_ROOT, "experiments", f"{EXPERIMENT_NAME}_codebook_metrics.csv"
    )
    SCREENING_PATH = os.path.join(PROJECT_ROOT, "experiments", f"{EXPERIMENT_NAME}_screening.csv")
    SNAPSHOT_DIR = os.path.join(PROJECT_ROOT, "experiments/snapshots", EXPERIMENT_NAME)
    NUM_EPOCHS = _env_int("SIMVQ_NUM_EPOCHS", 200)
    RESUME = _env_int("SIMVQ_RESUME", 1) == 1
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

    @classmethod
    def validate(cls):
        checks = {
            "DOWNSAMPLE_STRIDES": cls.DOWNSAMPLE_STRIDES,
            "EMBEDDING_DIM_LIST": cls.EMBEDDING_DIM_LIST,
            "NUM_EMBEDDINGS_LIST": cls.NUM_EMBEDDINGS_LIST,
            "LAYER_LOSS_WEIGHTS_INIT": cls.LAYER_LOSS_WEIGHTS_INIT,
            "LAYER_LOSS_WEIGHTS_FINAL": cls.LAYER_LOSS_WEIGHTS_FINAL,
            "QUANTIZER_AXIS_LIST": cls.QUANTIZER_AXIS_LIST,
            "CVQ_CODEWORD_SHAPES": cls.CVQ_CODEWORD_SHAPES,
            "RQ_DEPTH_LIST": cls.RQ_DEPTH_LIST,
        }
        for name, value in checks.items():
            if len(value) != cls.UNET_DEPTH:
                raise ValueError(f"{name} length ({len(value)}) must equal UNET_DEPTH ({cls.UNET_DEPTH})")

        expected_skip = max(cls.UNET_DEPTH - 1, 0)
        for name in ("SKIP_DROPOUT_P_INIT", "SKIP_DROPOUT_P_FINAL"):
            value = getattr(cls, name)
            if len(value) != expected_skip:
                raise ValueError(f"{name} length ({len(value)}) must equal UNET_DEPTH - 1 ({expected_skip})")

        if cls.NUM_DOWNSAMPLE_BLOCKS != cls.UNET_DEPTH:
            raise ValueError("NUM_DOWNSAMPLE_BLOCKS must equal UNET_DEPTH")
        if cls.QUANTIZER_TYPE not in {
            "simvq",
            "vq",
            "vitvq_nocompress",
            "rq_ema",
            "residual_simvq",
            "none",
        }:
            raise ValueError(
                "SIMVQ_QUANTIZER_TYPE must be simvq, vq, vitvq_nocompress, "
                "rq_ema, residual_simvq, or none"
            )
        for axis in cls.QUANTIZER_AXIS_LIST:
            if axis not in {"patch", "channel"}:
                raise ValueError("SIMVQ_QUANTIZER_AXIS_LIST entries must be patch or channel")
        if cls.QUANTIZER_TYPE in {"rq_ema", "residual_simvq"}:
            if any(axis != "patch" for axis in cls.QUANTIZER_AXIS_LIST):
                raise ValueError(
                    f"{cls.QUANTIZER_TYPE} only supports patch-wise quantization"
                )
            if not cls.RQ_SHARED_CODEBOOK:
                raise ValueError(
                    f"{cls.QUANTIZER_TYPE} requires SIMVQ_RQ_SHARED_CODEBOOK=1"
                )
            if any(depth <= 0 for depth in cls.RQ_DEPTH_LIST):
                raise ValueError("SIMVQ_RQ_DEPTH_LIST entries must be positive integers")
            if any(size < 2 or size & (size - 1) for size in cls.NUM_EMBEDDINGS_LIST):
                raise ValueError(
                    f"{cls.QUANTIZER_TYPE} codebook sizes must be powers of two "
                    "greater than or equal to 2"
                )
            if (
                cls.MSE_LOSS_WEIGHT != 1.0
                or cls.MS_SSIM_LOSS_WEIGHT != 0.0
                or cls.LPIPS_LOSS_WEIGHT != 0.0
            ):
                raise ValueError(
                    f"{cls.QUANTIZER_TYPE} requires MSE=1, MS-SSIM=0, and LPIPS=0"
                )
            if cls.USE_BOTTLENECK_ATTENTION:
                raise ValueError(
                    f"{cls.QUANTIZER_TYPE} requires bottleneck attention to be disabled"
                )
            if cls.USE_CASCADE_DOWNSAMPLE:
                raise ValueError(
                    f"{cls.QUANTIZER_TYPE} requires cascade downsampling to be disabled"
                )
        if cls.QUANTIZER_TYPE == "rq_ema":
            if not 0.0 <= cls.RQ_EMA_DECAY < 1.0:
                raise ValueError("SIMVQ_RQ_EMA_DECAY must be in [0, 1)")
            if any(weight != 1.0 for weight in cls.LAYER_LOSS_WEIGHTS_INIT):
                raise ValueError("rq_ema requires SIMVQ_LAYER_LOSS_WEIGHTS_INIT to be all 1")
            if any(weight != 1.0 for weight in cls.LAYER_LOSS_WEIGHTS_FINAL):
                raise ValueError("rq_ema requires SIMVQ_LAYER_LOSS_WEIGHTS_FINAL to be all 1")
        if cls.QUANTIZER_TYPE == "residual_simvq":
            expected_init = _default_loss_weights_init(cls.UNET_DEPTH)
            expected_final = _default_loss_weights_final(cls.UNET_DEPTH)
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(
                    cls.LAYER_LOSS_WEIGHTS_INIT, expected_init
                )
            ):
                raise ValueError(
                    "residual_simvq requires the original per-scale initial "
                    f"loss weights {expected_init}"
                )
            if any(
                not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(
                    cls.LAYER_LOSS_WEIGHTS_FINAL, expected_final
                )
            ):
                raise ValueError(
                    "residual_simvq requires the original per-scale final "
                    f"loss weights {expected_final}"
                )
        if cls.QUANTIZER_TYPE != "simvq" and any(axis == "channel" for axis in cls.QUANTIZER_AXIS_LIST):
            raise ValueError("channel-wise CVQ is currently implemented for SIMVQ_QUANTIZER_TYPE=simvq")

    @classmethod
    def architecture_summary(cls):
        return {
            "experiment_name": cls.EXPERIMENT_NAME,
            "experiment_stage": cls.EXPERIMENT_STAGE,
            "in_channels": cls.IN_CHANNELS,
            "out_channels": cls.OUT_CHANNELS,
            "unet_depth": cls.UNET_DEPTH,
            "num_downsample_blocks": cls.NUM_DOWNSAMPLE_BLOCKS,
            "base_channels": cls.BASE_CHANNELS,
            "downsample_strides": list(cls.DOWNSAMPLE_STRIDES),
            "total_downsample": math.prod(cls.DOWNSAMPLE_STRIDES),
            "train_image_size": tuple(cls.TRAIN_IMAGE_SIZE),
            "test_image_size": tuple(cls.TEST_IMAGE_SIZE),
            "estimated_source_bpp": cls.ESTIMATED_SOURCE_BPP,
            "estimated_source_bits_per_image": cls.ESTIMATED_SOURCE_BITS_PER_IMAGE,
            "estimated_test_source_bpp": cls.ESTIMATED_TEST_SOURCE_BPP,
            "estimated_test_transmission_ratio": cls.ESTIMATED_TEST_TRANSMISSION_RATIO,
            "embedding_dim_list": list(cls.EMBEDDING_DIM_LIST),
            "num_embeddings_list": list(cls.NUM_EMBEDDINGS_LIST),
            "quantizer_type": cls.QUANTIZER_TYPE,
            "rq_depth_list": list(cls.RQ_DEPTH_LIST),
            "rq_ema_decay": cls.RQ_EMA_DECAY,
            "rq_restart_unused_codes": cls.RQ_RESTART_UNUSED_CODES,
            "rq_shared_codebook": cls.RQ_SHARED_CODEBOOK,
            "quantizer_axis_list": list(cls.QUANTIZER_AXIS_LIST),
            "cvq_codeword_shapes": list(cls.CVQ_CODEWORD_SHAPES),
            "nested_channel_dropout_alpha": cls.NESTED_CHANNEL_DROPOUT_ALPHA,
            "vitvq_qbridge_type": cls.VITVQ_QBRIDGE_TYPE,
            "vitvq_emb_nograd": cls.VITVQ_EMB_NOGRAD,
            "loss_weights_init": list(cls.LAYER_LOSS_WEIGHTS_INIT),
            "loss_weights_final": list(cls.LAYER_LOSS_WEIGHTS_FINAL),
            "skip_dropout_init": list(cls.SKIP_DROPOUT_P_INIT),
            "skip_dropout_final": list(cls.SKIP_DROPOUT_P_FINAL),
            "channel_prob_start_epoch": cls.CHANNEL_PROB_START_EPOCH,
            "channel_prob_end_epoch": cls.CHANNEL_PROB_END_EPOCH,
            "norm_type": cls.NORM_TYPE,
            "activation": cls.ACTIVATION,
            "encoder_res_blocks": cls.ENCODER_RES_BLOCKS,
            "decoder_res_blocks": cls.DECODER_RES_BLOCKS,
            "upsample_mode": cls.UPSAMPLE_MODE,
            "use_cascade_downsample": cls.USE_CASCADE_DOWNSAMPLE,
            "use_bottleneck_attention": cls.USE_BOTTLENECK_ATTENTION,
            "bottleneck_attention_blocks": cls.BOTTLENECK_ATTENTION_BLOCKS,
            "mse_loss_weight": cls.MSE_LOSS_WEIGHT,
            "ms_ssim_loss_weight": cls.MS_SSIM_LOSS_WEIGHT,
            "lpips_loss_weight": cls.LPIPS_LOSS_WEIGHT,
            "resume": cls.RESUME,
            "checkpoint_dir": cls.CHECKPOINT_DIR,
        }
