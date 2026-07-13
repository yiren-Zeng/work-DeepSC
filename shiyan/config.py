# 大部分实验参数都从这里读取，并支持环境变量覆盖
import torch
import os
import math

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


def _experiment_name(family, depth, strides, num_embeddings_list):
    stride_part = "x".join(str(v) for v in strides) # [8,2] → "8x2"
    return f"{family}_unet{depth}_ds{stride_part}_{_format_k_list(num_embeddings_list)}"


def _source_bpp(strides, num_embeddings_list, quantizer_axis_list=None,
                embedding_dim_list=None, image_size=(256, 256)):
    bpp = 0.0
    cumulative_downsample = 1
    quantizer_axis_list = quantizer_axis_list or ["patch"] * len(num_embeddings_list)
    embedding_dim_list = embedding_dim_list or [None] * len(num_embeddings_list)
    image_h, image_w = image_size
    for i, (stride, codebook_size) in enumerate(zip(strides, num_embeddings_list)):
        cumulative_downsample *= stride
        bits = math.log2(codebook_size)
        if quantizer_axis_list[i] == "channel":
            token_count = embedding_dim_list[i]
            bpp += token_count * bits / (image_h * image_w)
        else:
            bpp += bits / (cumulative_downsample ** 2)
    return bpp

# 环境变量读取
def _env_int(name, default):
    value = os.environ.get(name) # os.environ 是 dict，存所有环境变量
    return int(value) if value else default


def _env_int_optional(name):
    value = os.environ.get(name)
    return int(value) if value else None


def _env_float(name, default):
    value = os.environ.get(name)
    return float(value) if value else default


def _env_float_optional(name):
    value = os.environ.get(name)
    return float(value) if value else None


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
    return [int(item.strip()) for item in value.split(",") if item.strip()] # .split(",")，按照逗号切分字符串，并且自动返回一个列表


def _env_int_list_optional(name):
    value = os.environ.get(name)
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _resize_tuple_from_env(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    parts = [part.strip() for part in value.replace("x", ",").split(",") if part.strip()] # .replace("x", ",") 把字符串里的小写字母 "x" 替换成逗号 ","
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as H,W, for example 256,256")
    return int(parts[0]), int(parts[1]) # tuple[int, int], 如(H, W)


# def _default_cvq_codeword_shapes(strides): # CVQ 码字形状：每层下采样后特征图的空间维度，除非通过环境变量覆盖。对于 patch-wise CVQ，使用 None。 
#     train_h, train_w = _resize_tuple_from_env("SIMVQ_TRAIN_RESIZE", (256, 256))
#     shapes = []
#     cumulative = 1
#     for stride in strides:
#         cumulative *= stride
#         shapes.append((train_h // cumulative, train_w // cumulative))
#     return shapes


def _env_shape_list(name, default=None): # 这个也是生成形状的
    value = os.environ.get(name)
    
    if not value:
        return None if default is None else list(default)
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


_STAGE = os.environ.get("SIMVQ_EXPERIMENT_STAGE", "B") # 读环境变量，默认 "B"
_STAGE_SETTINGS = _stage_settings(_STAGE) # 查字典，拿到对应配置


class Config:
    EXPERIMENT_STAGE = _STAGE
    # Allow env override of experiment family (for variant experiments)
    FAMILY_OVERRIDE = _env_str(
        "SIMVQ_EXP_FAMILY",
        "shiyan_raq_quality_v2_B_larger_rate044_A_patch_cb16-2_ch512-1024",
    )
    EXPERIMENT_FAMILY = FAMILY_OVERRIDE if FAMILY_OVERRIDE else _STAGE_SETTINGS["family"]
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    # Change this number to switch the model between 2/3/4-layer U-Net variants.
    UNET_DEPTH = _env_int("SIMVQ_UNET_DEPTH", 2)
    NUM_DOWNSAMPLE_BLOCKS = UNET_DEPTH
    BASE_CHANNELS = _env_int("SIMVQ_BASE_CHANNELS", 256)
    EMBEDDING_DIM_LIST = _default_embedding_dims(BASE_CHANNELS, UNET_DEPTH)
    NUM_EMBEDDINGS_PER_LAYER = None
    NUM_EMBEDDINGS_LIST = _env_int_list("SIMVQ_NUM_EMBEDDINGS_LIST", [16, 2])
    QUANTIZER_TYPE = _env_str("SIMVQ_QUANTIZER_TYPE", "simvq").lower()
    _USE_RAQ_VALUE = _env_int_optional("SIMVQ_USE_RAQ")
    USE_RAQ = (_USE_RAQ_VALUE == 1) if _USE_RAQ_VALUE is not None else False
    RAQ_TARGET_LIST = _env_int_list_optional("SIMVQ_RAQ_TARGET_LIST")
    RAQ_MIN_TRG = _env_int_optional("SIMVQ_RAQ_MIN_TRG")
    RAQ_MAX_TRG = _env_int_optional("SIMVQ_RAQ_MAX_TRG")
    RAQ_REPULSION_WEIGHT = _env_float_optional("SIMVQ_RAQ_REPULSION_WEIGHT")
    RAQ_LATENT_DISTILL_WEIGHT = _env_float("SIMVQ_RAQ_LATENT_DISTILL_WEIGHT", 0.0)
    RAQ_LATENT_DISTILL_FINAL_WEIGHT = _env_float_optional("SIMVQ_RAQ_LATENT_DISTILL_FINAL_WEIGHT")
    RAQ_LATENT_DISTILL_DECAY_START_EPOCH = _env_int("SIMVQ_RAQ_LATENT_DISTILL_DECAY_START_EPOCH", 0)
    RAQ_LATENT_DISTILL_DECAY_END_EPOCH = _env_int_optional("SIMVQ_RAQ_LATENT_DISTILL_DECAY_END_EPOCH")
    RAQ_USE_CURRICULUM = _env_int("SIMVQ_RAQ_USE_CURRICULUM", 0) == 1
    RAQ_CURRICULUM_EARLY_LIST = _env_int_list("SIMVQ_RAQ_CURRICULUM_EARLY_LIST", [32, 64])
    RAQ_CURRICULUM_MIDDLE_LIST = _env_int_list("SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST", [8, 16, 32, 64])
    RAQ_CURRICULUM_LATE_LIST = _env_int_list("SIMVQ_RAQ_CURRICULUM_LATE_LIST", [2, 4, 8, 16, 32, 64])
    RAQ_RECON_GRAD_MODE = _env_str("SIMVQ_RAQ_RECON_GRAD_MODE", "ste").lower()
    RAQ_GENERATOR_TYPE = _env_str("SIMVQ_RAQ_GENERATOR_TYPE", "encoder_decoder").replace("-", "_").lower()
    RAQ_ROUTED_SRC_ENABLED = _env_int("SIMVQ_RAQ_ROUTED_SRC_ENABLED", 0) == 1
    RAQ_ROUTED_SRC_THRESHOLD = _env_int("SIMVQ_RAQ_ROUTED_SRC_THRESHOLD", 16)
    RAQ_ROUTED_SRC_SMALL_LIST = _env_int_list_optional("SIMVQ_RAQ_ROUTED_SRC_SMALL_LIST")
    _RAQ_ROUTED_SRC_LARGE_LIST_VALUE = _env_int_list_optional("SIMVQ_RAQ_ROUTED_SRC_LARGE_LIST")
    RAQ_ROUTED_SRC_LARGE_LIST = (
        _RAQ_ROUTED_SRC_LARGE_LIST_VALUE
        if _RAQ_ROUTED_SRC_LARGE_LIST_VALUE is not None
        else list(NUM_EMBEDDINGS_LIST)
    )
    RAQ_TRAIN_ENCODER = _env_int("SIMVQ_RAQ_TRAIN_ENCODER", 0) == 1
    RAQ_SRC_RECON_WEIGHT = _env_float("SIMVQ_RAQ_SRC_RECON_WEIGHT", 0.0)
    RAQ_SRC_RECON_FINAL_WEIGHT = _env_float_optional("SIMVQ_RAQ_SRC_RECON_FINAL_WEIGHT")
    RAQ_SRC_VQ_WEIGHT = _env_float("SIMVQ_RAQ_SRC_VQ_WEIGHT", 0.0)
    RAQ_SRC_VQ_FINAL_WEIGHT = _env_float_optional("SIMVQ_RAQ_SRC_VQ_FINAL_WEIGHT")
    RAQ_CODEBOOK_ANCHOR_WEIGHT = _env_float("SIMVQ_RAQ_CODEBOOK_ANCHOR_WEIGHT", 0.0)
    RAQ_CODEBOOK_ANCHOR_FINAL_WEIGHT = _env_float_optional("SIMVQ_RAQ_CODEBOOK_ANCHOR_FINAL_WEIGHT")
    RAQ_JOINTLITE_DECAY_START_EPOCH = _env_int("SIMVQ_RAQ_JOINTLITE_DECAY_START_EPOCH", 0)
    RAQ_JOINTLITE_DECAY_END_EPOCH = _env_int_optional("SIMVQ_RAQ_JOINTLITE_DECAY_END_EPOCH")
    SRC_CODEBOOK_ANCHOR_CHECKPOINT = _env_str("SIMVQ_SRC_CODEBOOK_ANCHOR_CHECKPOINT", "")
    TRAIN_BRANCH = _env_str("SIMVQ_TRAIN_BRANCH", "joint").lower()
    SRC_CODEBOOK_REPULSION_WEIGHT = _env_float("SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT", 0.0)
    SRC_CODEBOOK_REPULSION_MARGIN = _env_float("SIMVQ_SRC_CODEBOOK_REPULSION_MARGIN", 0.5)
    SRC_CODEBOOK_REPULSION_NORMALIZE = _env_int("SIMVQ_SRC_CODEBOOK_REPULSION_NORMALIZE", 1) == 1
    SRC_CODEBOOK_REPULSION_WARMUP_START_EPOCH = _env_int("SIMVQ_SRC_CODEBOOK_REPULSION_WARMUP_START_EPOCH", 5)
    SRC_CODEBOOK_REPULSION_WARMUP_END_EPOCH = _env_int("SIMVQ_SRC_CODEBOOK_REPULSION_WARMUP_END_EPOCH", 20)
    VITVQ_QBRIDGE_TYPE = _env_str("SIMVQ_VITVQ_QBRIDGE_TYPE", "QBridgeNoCompress-S")
    VITVQ_EMB_NOGRAD = _env_int("SIMVQ_VITVQ_EMB_NOGRAD", 0) == 1
    DOWNSAMPLE_STRIDES = _env_int_list("SIMVQ_DOWNSAMPLE_STRIDES", [8, 2])
    QUANTIZER_AXIS_LIST = _env_str_list("SIMVQ_QUANTIZER_AXIS_LIST", ["patch"] * UNET_DEPTH) # 量化轴
    CVQ_CODEWORD_SHAPES = _env_shape_list(
        "SIMVQ_CVQ_CODEWORD_SHAPES", [None] * UNET_DEPTH
    )
    NESTED_CHANNEL_DROPOUT_ALPHA = _env_float("SIMVQ_NESTED_CHANNEL_DROPOUT_ALPHA", 0.0)
    
    LAYER_LOSS_WEIGHTS_INIT = _default_loss_weights_init(UNET_DEPTH)
    LAYER_LOSS_WEIGHTS_FINAL = _default_loss_weights_final(UNET_DEPTH)
    
    SKIP_DROPOUT_P_INIT = _default_skip_dropout_init(UNET_DEPTH)
    SKIP_DROPOUT_P_FINAL = _default_skip_dropout_final(UNET_DEPTH)
    PHASE1_END = _STAGE_SETTINGS["phase1_end"]
    PHASE2_END = _STAGE_SETTINGS["phase2_end"]
    LEARNING_RATE_G = _env_float("SIMVQ_LEARNING_RATE_G", 5e-5)
    CODEBOOK_PROJ_LR = _env_float("SIMVQ_CODEBOOK_PROJ_LR", 2e-4)
    BETAS = (0.5, 0.999)
    COMMITMENT_COST = 0.25
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
    ENCODER_RES_BLOCKS = _env_int("SIMVQ_ENCODER_RES_BLOCKS", 4)
    DECODER_RES_BLOCKS = _env_int("SIMVQ_DECODER_RES_BLOCKS", 4)
    UPSAMPLE_MODE = _STAGE_SETTINGS["upsample_mode"]
    USE_CASCADE_DOWNSAMPLE = _STAGE_SETTINGS["use_cascade_downsample"] # 是否采用级联方式进行下采样
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
    TEST_DATASET_PATH = _env_str("SIMVQ_TEST_DATASET_PATH", "/workspace/yi/work/Kodak-256-transform-resize")
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    MODEL_PARALLEL = _env_int("SIMVQ_MODEL_PARALLEL", 0) == 1
    ENCODER_DEVICE = _env_str("SIMVQ_ENCODER_DEVICE", DEVICE)
    DECODER_DEVICE = _env_str("SIMVQ_DECODER_DEVICE", "cuda:1" if torch.cuda.device_count() > 1 else DEVICE)
    EXPERIMENT_NAME = _experiment_name(
        EXPERIMENT_FAMILY, UNET_DEPTH, DOWNSAMPLE_STRIDES, NUM_EMBEDDINGS_LIST
    )
    ESTIMATED_SOURCE_BPP = _source_bpp(
        DOWNSAMPLE_STRIDES,
        NUM_EMBEDDINGS_LIST,
        QUANTIZER_AXIS_LIST,
        EMBEDDING_DIM_LIST,
        _resize_tuple_from_env("SIMVQ_TRAIN_RESIZE", (256, 256)),
    )
    ESTIMATED_TEST_SOURCE_BPP = _source_bpp(
        DOWNSAMPLE_STRIDES,
        NUM_EMBEDDINGS_LIST,
        QUANTIZER_AXIS_LIST,
        EMBEDDING_DIM_LIST,
        _resize_tuple_from_env("SIMVQ_TEST_RESIZE", (768, 512)),
    )
    _RAQ_BPP_K_LIST = RAQ_TARGET_LIST if (USE_RAQ and RAQ_TARGET_LIST is not None) else NUM_EMBEDDINGS_LIST
    ESTIMATED_RAQ_TARGET_BPP = _source_bpp(
        DOWNSAMPLE_STRIDES,
        _RAQ_BPP_K_LIST,
        QUANTIZER_AXIS_LIST,
        EMBEDDING_DIM_LIST,
        _resize_tuple_from_env("SIMVQ_TEST_RESIZE", (768, 512)),
    )
    ESTIMATED_TEST_TRANSMISSION_RATIO = (
        ESTIMATED_RAQ_TARGET_BPP / (CHANNEL_CODING_RATE_VAL * 1 * 3)
    )
    CHECKPOINT_DIR = os.path.join("./checkpoints", EXPERIMENT_NAME)
    LOG_DIR = os.path.join("./experiments/tensorboard", EXPERIMENT_NAME)
    METRICS_PATH = os.path.join("./experiments", f"{EXPERIMENT_NAME}_epoch_metrics.csv")
    CODEBOOK_METRICS_PATH = os.path.join("./experiments", f"{EXPERIMENT_NAME}_codebook_metrics.csv")
    SCREENING_PATH = os.path.join("./experiments", f"{EXPERIMENT_NAME}_screening.csv")
    SNAPSHOT_DIR = os.path.join("./experiments/snapshots", EXPERIMENT_NAME)
    NUM_EPOCHS = _env_int("NUM_EPOCHS", 200)
    RESUME = _env_int("SIMVQ_RESUME", 0) == 1
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

    @classmethod
    def validate(cls): # validate() 是配置合法性检查函数
        checks = {
            "DOWNSAMPLE_STRIDES": cls.DOWNSAMPLE_STRIDES,
            "EMBEDDING_DIM_LIST": cls.EMBEDDING_DIM_LIST,
            "NUM_EMBEDDINGS_LIST": cls.NUM_EMBEDDINGS_LIST,
            "LAYER_LOSS_WEIGHTS_INIT": cls.LAYER_LOSS_WEIGHTS_INIT,
            "LAYER_LOSS_WEIGHTS_FINAL": cls.LAYER_LOSS_WEIGHTS_FINAL,
            "QUANTIZER_AXIS_LIST": cls.QUANTIZER_AXIS_LIST,
            "CVQ_CODEWORD_SHAPES": cls.CVQ_CODEWORD_SHAPES,
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
        if cls.QUANTIZER_TYPE not in {"simvq", "vq", "vitvq_nocompress", "none"}:
            raise ValueError("SIMVQ_QUANTIZER_TYPE must be simvq, vq, vitvq_nocompress, or none")
        for axis in cls.QUANTIZER_AXIS_LIST:
            if axis not in {"patch", "channel"}:
                raise ValueError("SIMVQ_QUANTIZER_AXIS_LIST entries must be patch or channel")
        if cls.QUANTIZER_TYPE != "simvq" and any(axis == "channel" for axis in cls.QUANTIZER_AXIS_LIST):
            raise ValueError("channel-wise CVQ is currently implemented for SIMVQ_QUANTIZER_TYPE=simvq")
        valid_train_branches = {
            "joint", "src", "raq_warmup", "raq_finetune", "raq_channel",
            "raq_jointlite", "raq_jointlite_channel",
        }
        if cls.TRAIN_BRANCH not in valid_train_branches:
            raise ValueError(
                "SIMVQ_TRAIN_BRANCH must be one of: " + ", ".join(sorted(valid_train_branches))
            )
        if cls.TRAIN_BRANCH == "src" and cls.USE_RAQ:
            raise ValueError("SIMVQ_TRAIN_BRANCH=src requires SIMVQ_USE_RAQ=0")
        if cls.TRAIN_BRANCH.startswith("raq_") and not cls.USE_RAQ:
            raise ValueError(f"SIMVQ_TRAIN_BRANCH={cls.TRAIN_BRANCH} requires SIMVQ_USE_RAQ=1")
        if cls.RAQ_RECON_GRAD_MODE not in {"ste", "dual"}:
            raise ValueError("SIMVQ_RAQ_RECON_GRAD_MODE must be ste or dual")
        if cls.RAQ_GENERATOR_TYPE not in {"encoder_decoder", "decoder_only"}:
            raise ValueError("SIMVQ_RAQ_GENERATOR_TYPE must be encoder_decoder or decoder_only")
        if cls.RAQ_ROUTED_SRC_ENABLED:
            if not cls.USE_RAQ:
                raise ValueError("SIMVQ_RAQ_ROUTED_SRC_ENABLED requires SIMVQ_USE_RAQ=1")
            if cls.RAQ_ROUTED_SRC_SMALL_LIST is None:
                raise ValueError("SIMVQ_RAQ_ROUTED_SRC_ENABLED requires SIMVQ_RAQ_ROUTED_SRC_SMALL_LIST")
            routed_lists = {
                "SIMVQ_RAQ_ROUTED_SRC_SMALL_LIST": cls.RAQ_ROUTED_SRC_SMALL_LIST,
                "SIMVQ_RAQ_ROUTED_SRC_LARGE_LIST": cls.RAQ_ROUTED_SRC_LARGE_LIST,
            }
            for name, values in routed_lists.items():
                if len(values) != cls.UNET_DEPTH:
                    raise ValueError(f"{name} length ({len(values)}) must equal UNET_DEPTH ({cls.UNET_DEPTH})")
                for value in values:
                    if value < 2:
                        raise ValueError(f"{name} entries must be >= 2")
            if list(cls.RAQ_ROUTED_SRC_LARGE_LIST) != list(cls.NUM_EMBEDDINGS_LIST):
                raise ValueError(
                    "SIMVQ_RAQ_ROUTED_SRC_LARGE_LIST must match SIMVQ_NUM_EMBEDDINGS_LIST "
                    "so the existing source codebook bank remains the large bank"
                )
            if cls.RAQ_ROUTED_SRC_THRESHOLD < 2:
                raise ValueError("SIMVQ_RAQ_ROUTED_SRC_THRESHOLD must be >= 2")
        jointlite_weights = {
            "SIMVQ_RAQ_SRC_RECON_WEIGHT": cls.RAQ_SRC_RECON_WEIGHT,
            "SIMVQ_RAQ_SRC_VQ_WEIGHT": cls.RAQ_SRC_VQ_WEIGHT,
            "SIMVQ_RAQ_CODEBOOK_ANCHOR_WEIGHT": cls.RAQ_CODEBOOK_ANCHOR_WEIGHT,
        }
        jointlite_final_weights = {
            "SIMVQ_RAQ_SRC_RECON_FINAL_WEIGHT": cls.RAQ_SRC_RECON_FINAL_WEIGHT,
            "SIMVQ_RAQ_SRC_VQ_FINAL_WEIGHT": cls.RAQ_SRC_VQ_FINAL_WEIGHT,
            "SIMVQ_RAQ_CODEBOOK_ANCHOR_FINAL_WEIGHT": cls.RAQ_CODEBOOK_ANCHOR_FINAL_WEIGHT,
        }
        for name, weight in jointlite_weights.items():
            if weight < 0:
                raise ValueError(f"{name} must be >= 0")
        for name, weight in jointlite_final_weights.items():
            if weight is not None and weight < 0:
                raise ValueError(f"{name} must be >= 0")
        if cls.RAQ_JOINTLITE_DECAY_START_EPOCH < 0:
            raise ValueError("SIMVQ_RAQ_JOINTLITE_DECAY_START_EPOCH must be >= 0")
        if (
            cls.RAQ_JOINTLITE_DECAY_END_EPOCH is not None
            and cls.RAQ_JOINTLITE_DECAY_END_EPOCH < cls.RAQ_JOINTLITE_DECAY_START_EPOCH
        ):
            raise ValueError(
                "SIMVQ_RAQ_JOINTLITE_DECAY_END_EPOCH must be >= decay start"
            )
        if cls.RAQ_LATENT_DISTILL_FINAL_WEIGHT is not None and cls.RAQ_LATENT_DISTILL_FINAL_WEIGHT < 0:
            raise ValueError("SIMVQ_RAQ_LATENT_DISTILL_FINAL_WEIGHT must be >= 0")
        if cls.RAQ_LATENT_DISTILL_DECAY_START_EPOCH < 0:
            raise ValueError("SIMVQ_RAQ_LATENT_DISTILL_DECAY_START_EPOCH must be >= 0")
        if (
            cls.RAQ_LATENT_DISTILL_DECAY_END_EPOCH is not None
            and cls.RAQ_LATENT_DISTILL_DECAY_END_EPOCH < cls.RAQ_LATENT_DISTILL_DECAY_START_EPOCH
        ):
            raise ValueError(
                "SIMVQ_RAQ_LATENT_DISTILL_DECAY_END_EPOCH must be >= decay start"
            )
        if cls.CHANNEL_PROB_START_EPOCH < 0 or cls.CHANNEL_PROB_END_EPOCH < cls.CHANNEL_PROB_START_EPOCH:
            raise ValueError(
                "SIMVQ_CHANNEL_PROB_END_EPOCH must be >= SIMVQ_CHANNEL_PROB_START_EPOCH >= 0"
            )
        if cls.USE_RAQ:
            required_raq_envs = {
                # "SIMVQ_RAQ_TARGET_LIST": cls.RAQ_TARGET_LIST,
                "SIMVQ_RAQ_MIN_TRG": cls.RAQ_MIN_TRG,
                "SIMVQ_RAQ_MAX_TRG": cls.RAQ_MAX_TRG,
                "SIMVQ_RAQ_REPULSION_WEIGHT": cls.RAQ_REPULSION_WEIGHT,
            }
            missing = [name for name, value in required_raq_envs.items() if value is None]
            if missing:
                raise ValueError(
                    "SIMVQ_USE_RAQ=1 requires explicit env vars: " + ", ".join(missing)
                )
            if cls.QUANTIZER_TYPE != "simvq":
                raise ValueError("RAQ integration requires SIMVQ_QUANTIZER_TYPE=simvq")
            if any(axis != "patch" for axis in cls.QUANTIZER_AXIS_LIST):
                raise ValueError("RAQ integration currently supports patch-wise quantizers only")
            # if len(cls.RAQ_TARGET_LIST) != cls.UNET_DEPTH:
            #     raise ValueError("SIMVQ_RAQ_TARGET_LIST length must equal UNET_DEPTH")
            if cls.RAQ_MIN_TRG < 2 or cls.RAQ_MIN_TRG > cls.RAQ_MAX_TRG:
                raise ValueError("RAQ target range must satisfy 2 <= RAQ_MIN_TRG <= RAQ_MAX_TRG")
            if cls.RAQ_LATENT_DISTILL_WEIGHT < 0:
                raise ValueError("SIMVQ_RAQ_LATENT_DISTILL_WEIGHT must be >= 0")
            if (
                cls.TRAIN_BRANCH in {"raq_jointlite", "raq_jointlite_channel"}
                and cls.RAQ_CODEBOOK_ANCHOR_WEIGHT > 0
                and not cls.SRC_CODEBOOK_ANCHOR_CHECKPOINT
            ):
                raise ValueError(
                    "SIMVQ_SRC_CODEBOOK_ANCHOR_CHECKPOINT is required when "
                    "joint-lite codebook anchor weight is > 0"
                )
            if cls.RAQ_USE_CURRICULUM:
                curriculum_lists = {
                    "SIMVQ_RAQ_CURRICULUM_EARLY_LIST": cls.RAQ_CURRICULUM_EARLY_LIST,
                    "SIMVQ_RAQ_CURRICULUM_MIDDLE_LIST": cls.RAQ_CURRICULUM_MIDDLE_LIST,
                    "SIMVQ_RAQ_CURRICULUM_LATE_LIST": cls.RAQ_CURRICULUM_LATE_LIST,
                }
                for name, values in curriculum_lists.items():
                    if not values:
                        raise ValueError(f"{name} must not be empty")
                    for value in values:
                        if value < cls.RAQ_MIN_TRG or value > cls.RAQ_MAX_TRG:
                            raise ValueError(f"{name} contains {value}, outside RAQ target range")
                        if value & (value - 1) != 0:
                            raise ValueError(f"{name} contains {value}, which is not a power of two")
            # for target in cls.RAQ_TARGET_LIST:
            #     if target < cls.RAQ_MIN_TRG or target > cls.RAQ_MAX_TRG:
            #         raise ValueError("Every RAQ target K must be inside [RAQ_MIN_TRG, RAQ_MAX_TRG]")
        elif cls.RAQ_LATENT_DISTILL_WEIGHT > 0:
            raise ValueError("SIMVQ_RAQ_LATENT_DISTILL_WEIGHT requires SIMVQ_USE_RAQ=1")
        elif cls.RAQ_USE_CURRICULUM:
            raise ValueError("SIMVQ_RAQ_USE_CURRICULUM requires SIMVQ_USE_RAQ=1")
        if cls.SRC_CODEBOOK_REPULSION_WEIGHT < 0:
            raise ValueError("SIMVQ_SRC_CODEBOOK_REPULSION_WEIGHT must be >= 0")
        if cls.SRC_CODEBOOK_REPULSION_MARGIN <= 0:
            raise ValueError("SIMVQ_SRC_CODEBOOK_REPULSION_MARGIN must be > 0")
        if cls.SRC_CODEBOOK_REPULSION_WARMUP_START_EPOCH < 0:
            raise ValueError("SIMVQ_SRC_CODEBOOK_REPULSION_WARMUP_START_EPOCH must be >= 0")
        if cls.SRC_CODEBOOK_REPULSION_WARMUP_END_EPOCH < cls.SRC_CODEBOOK_REPULSION_WARMUP_START_EPOCH:
            raise ValueError(
                "SIMVQ_SRC_CODEBOOK_REPULSION_WARMUP_END_EPOCH must be >= warmup start"
            )

    @classmethod
    def architecture_summary(cls): # cls 代表类本身，即Config
        return {
            "experiment_name": cls.EXPERIMENT_NAME,
            "experiment_stage": cls.EXPERIMENT_STAGE,
            "unet_depth": cls.UNET_DEPTH,
            "downsample_strides": list(cls.DOWNSAMPLE_STRIDES),
            "total_downsample": math.prod(cls.DOWNSAMPLE_STRIDES),
            "estimated_source_bpp": cls.ESTIMATED_SOURCE_BPP,
            "estimated_test_source_bpp": cls.ESTIMATED_TEST_SOURCE_BPP,
            "estimated_raq_target_bpp": cls.ESTIMATED_RAQ_TARGET_BPP,
            "estimated_test_transmission_ratio": cls.ESTIMATED_TEST_TRANSMISSION_RATIO,
            "embedding_dim_list": list(cls.EMBEDDING_DIM_LIST),
            "num_embeddings_list": list(cls.NUM_EMBEDDINGS_LIST),
            "use_raq": cls.USE_RAQ,
            "raq_target_list": list(cls.RAQ_TARGET_LIST) if cls.RAQ_TARGET_LIST is not None else None,
            "raq_min_trg": cls.RAQ_MIN_TRG,
            "raq_max_trg": cls.RAQ_MAX_TRG,
            "raq_repulsion_weight": cls.RAQ_REPULSION_WEIGHT,
            "raq_latent_distill_weight": cls.RAQ_LATENT_DISTILL_WEIGHT,
            "raq_latent_distill_final_weight": cls.RAQ_LATENT_DISTILL_FINAL_WEIGHT,
            "raq_latent_distill_decay_start_epoch": cls.RAQ_LATENT_DISTILL_DECAY_START_EPOCH,
            "raq_latent_distill_decay_end_epoch": cls.RAQ_LATENT_DISTILL_DECAY_END_EPOCH,
            "raq_use_curriculum": cls.RAQ_USE_CURRICULUM,
            "raq_curriculum_early_list": list(cls.RAQ_CURRICULUM_EARLY_LIST),
            "raq_curriculum_middle_list": list(cls.RAQ_CURRICULUM_MIDDLE_LIST),
            "raq_curriculum_late_list": list(cls.RAQ_CURRICULUM_LATE_LIST),
            "raq_recon_grad_mode": cls.RAQ_RECON_GRAD_MODE,
            "raq_generator_type": cls.RAQ_GENERATOR_TYPE,
            "raq_routed_src_enabled": cls.RAQ_ROUTED_SRC_ENABLED,
            "raq_routed_src_threshold": cls.RAQ_ROUTED_SRC_THRESHOLD,
            "raq_routed_src_small_list": (
                list(cls.RAQ_ROUTED_SRC_SMALL_LIST)
                if cls.RAQ_ROUTED_SRC_SMALL_LIST is not None
                else None
            ),
            "raq_routed_src_large_list": list(cls.RAQ_ROUTED_SRC_LARGE_LIST),
            "raq_train_encoder": cls.RAQ_TRAIN_ENCODER,
            "raq_src_recon_weight": cls.RAQ_SRC_RECON_WEIGHT,
            "raq_src_recon_final_weight": cls.RAQ_SRC_RECON_FINAL_WEIGHT,
            "raq_src_vq_weight": cls.RAQ_SRC_VQ_WEIGHT,
            "raq_src_vq_final_weight": cls.RAQ_SRC_VQ_FINAL_WEIGHT,
            "raq_codebook_anchor_weight": cls.RAQ_CODEBOOK_ANCHOR_WEIGHT,
            "raq_codebook_anchor_final_weight": cls.RAQ_CODEBOOK_ANCHOR_FINAL_WEIGHT,
            "raq_jointlite_decay_start_epoch": cls.RAQ_JOINTLITE_DECAY_START_EPOCH,
            "raq_jointlite_decay_end_epoch": cls.RAQ_JOINTLITE_DECAY_END_EPOCH,
            "src_codebook_anchor_checkpoint": cls.SRC_CODEBOOK_ANCHOR_CHECKPOINT,
            "train_branch": cls.TRAIN_BRANCH,
            "src_codebook_repulsion_weight": cls.SRC_CODEBOOK_REPULSION_WEIGHT,
            "src_codebook_repulsion_margin": cls.SRC_CODEBOOK_REPULSION_MARGIN,
            "src_codebook_repulsion_normalize": cls.SRC_CODEBOOK_REPULSION_NORMALIZE,
            "src_codebook_repulsion_warmup_start_epoch": cls.SRC_CODEBOOK_REPULSION_WARMUP_START_EPOCH,
            "src_codebook_repulsion_warmup_end_epoch": cls.SRC_CODEBOOK_REPULSION_WARMUP_END_EPOCH,
            "quantizer_type": cls.QUANTIZER_TYPE,
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
            "checkpoint_dir": cls.CHECKPOINT_DIR,
        }
