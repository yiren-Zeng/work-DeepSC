import torch
import os
import math


def _default_downsample_strides(depth):
    """Return a conservative default downsampling plan for a U-Net depth."""
    if depth < 1:
        raise ValueError("UNET_DEPTH must be >= 1")
    if depth == 1:
        return [8]
    # Keep the existing 2-layer behavior and append extra 2x stages for
    # deeper variants. This keeps the source rate close to the 2-layer setup.
    return [4] + [2] * (depth - 1)


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
    return [0.25 * (i + 1) for i in range(depth)]


def _default_loss_weights_final(depth):
    return [0.25 for _ in range(depth)]


def _default_skip_dropout_init(depth):
    return [0.1 for _ in range(max(depth - 1, 0))]


def _default_skip_dropout_final(depth):
    return [0.0 for _ in range(max(depth - 1, 0))]


def _format_k_list(num_embeddings_list):
    unique_values = sorted(set(num_embeddings_list))
    if len(unique_values) == 1:
        return f"k{unique_values[0]}"
    return "k" + "-".join(str(v) for v in num_embeddings_list)


def _experiment_name(family, depth, strides, num_embeddings_list):
    stride_part = "x".join(str(v) for v in strides)
    return f"{family}_unet{depth}_ds{stride_part}_{_format_k_list(num_embeddings_list)}"


def _source_bpp(strides, num_embeddings_list):
    bpp = 0.0
    cumulative_downsample = 1
    for stride, codebook_size in zip(strides, num_embeddings_list):
        cumulative_downsample *= stride
        bpp += math.log2(codebook_size) / (cumulative_downsample ** 2)
    return bpp


def _stage_settings(stage):
    stage = (stage or "full").lower()
    settings = {
        "full": {
            "family": "quality_v2",
            "norm_type": "group",
            "activation": "silu",
            "encoder_res_blocks": 2,
            "decoder_res_blocks": 2,
            "upsample_mode": "bilinear",
            "use_attention": True,
            "attention_blocks": 1,
            "mse_loss_weight": 0.8,
            "ms_ssim_loss_weight": 0.2,
        },
        "a": {
            "family": "quality_v2_A_curriculum",
            "norm_type": "batch",
            "activation": "prelu",
            "encoder_res_blocks": 1,
            "decoder_res_blocks": 1,
            "upsample_mode": "nearest",
            "use_attention": False,
            "attention_blocks": 0,
            "mse_loss_weight": 1.0,
            "ms_ssim_loss_weight": 0.0,
        },
        "b": {
            "family": "quality_v2_B_backbone",
            "norm_type": "group",
            "activation": "silu",
            "encoder_res_blocks": 2,
            "decoder_res_blocks": 2,
            "upsample_mode": "bilinear",
            "use_attention": False,
            "attention_blocks": 0,
            "mse_loss_weight": 1.0,
            "ms_ssim_loss_weight": 0.0,
        },
        "c": {
            "family": "quality_v2_C_full",
            "norm_type": "group",
            "activation": "silu",
            "encoder_res_blocks": 2,
            "decoder_res_blocks": 2,
            "upsample_mode": "bilinear",
            "use_attention": True,
            "attention_blocks": 1,
            "mse_loss_weight": 0.8,
            "ms_ssim_loss_weight": 0.2,
        },
    }
    if stage not in settings:
        raise ValueError(f"Unknown SIMVQ_EXPERIMENT_STAGE={stage!r}; use full, A, B, or C")
    return settings[stage]


_STAGE = os.environ.get("SIMVQ_EXPERIMENT_STAGE", "full")
_STAGE_SETTINGS = _stage_settings(_STAGE)


class Config:
    EXPERIMENT_STAGE = _STAGE
    EXPERIMENT_FAMILY = _STAGE_SETTINGS["family"]
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    # Change this number to switch the model between 2/3/4-layer U-Net variants.
    UNET_DEPTH = 2
    NUM_DOWNSAMPLE_BLOCKS = UNET_DEPTH
    BASE_CHANNELS = 64
    EMBEDDING_DIM_LIST = _default_embedding_dims(BASE_CHANNELS, UNET_DEPTH)
    # 0.083-BPP target configuration: 4/64 + 5/256 = 0.0820 BPP.
    NUM_EMBEDDINGS_PER_LAYER = None
    NUM_EMBEDDINGS_LIST = [16, 32]
    COMMITMENT_COST = 0.25
    DOWNSAMPLE_STRIDES = [8, 2]
    # The baseline's VQ term dominated its reconstruction loss during early training.
    LAYER_LOSS_WEIGHTS_INIT = _default_loss_weights_init(UNET_DEPTH)
    LAYER_LOSS_WEIGHTS_FINAL = _default_loss_weights_final(UNET_DEPTH)
    # Retain robustness training without discarding half of the high-resolution path.
    SKIP_DROPOUT_P_INIT = _default_skip_dropout_init(UNET_DEPTH)
    SKIP_DROPOUT_P_FINAL = _default_skip_dropout_final(UNET_DEPTH)
    PHASE1_END = 0.4
    PHASE2_END = 0.6
    LEARNING_RATE_G = 5e-5
    CODEBOOK_PROJ_LR = 2e-4
    BETAS = (0.5, 0.999)
    CHANNEL_CODING_RATE_TRAIN = 0.5
    CHANNEL_CODING_RATE_VAL = 0.5
    BLOCK_LENGTH = 256
    SNR_RANGE_DB = [0, 15]
    CHANNEL_TYPE = "AWGN"
    RICIAN_K_FACTOR = 10
    CHANNEL_PROB_START_EPOCH = 80
    CHANNEL_PROB_END_EPOCH = 120
    NORM_TYPE = _STAGE_SETTINGS["norm_type"]
    GROUP_NORM_GROUPS = 32
    ACTIVATION = _STAGE_SETTINGS["activation"]
    ENCODER_RES_BLOCKS = _STAGE_SETTINGS["encoder_res_blocks"]
    DECODER_RES_BLOCKS = _STAGE_SETTINGS["decoder_res_blocks"]
    UPSAMPLE_MODE = _STAGE_SETTINGS["upsample_mode"]
    USE_BOTTLENECK_ATTENTION = _STAGE_SETTINGS["use_attention"]
    BOTTLENECK_ATTENTION_BLOCKS = _STAGE_SETTINGS["attention_blocks"]
    MSE_LOSS_WEIGHT = _STAGE_SETTINGS["mse_loss_weight"]
    MS_SSIM_LOSS_WEIGHT = _STAGE_SETTINGS["ms_ssim_loss_weight"]
    TOTAL_BATCH_SIZE = 24
    MICRO_BATCH_SIZE = 24
    NUM_WORKERS = 8
    PIN_MEMORY = True
    TRAIN_DATASET_PATH = "/workspace/yi/work/Cars196/train_data"
    VAL_DATASET_PATH = "/workspace/yi/work/Cars196/val_data"
    TEST_DATASET_PATH = "/workspace/yi/work/Kodak"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    EXPERIMENT_NAME = _experiment_name(
        EXPERIMENT_FAMILY, UNET_DEPTH, DOWNSAMPLE_STRIDES, NUM_EMBEDDINGS_LIST
    )
    ESTIMATED_SOURCE_BPP = _source_bpp(DOWNSAMPLE_STRIDES, NUM_EMBEDDINGS_LIST)
    CHECKPOINT_DIR = os.path.join("./checkpoints", EXPERIMENT_NAME)
    LOG_DIR = os.path.join("./experiments/tensorboard", EXPERIMENT_NAME)
    METRICS_PATH = os.path.join("./experiments", f"{EXPERIMENT_NAME}_epoch_metrics.csv")
    SCREENING_PATH = os.path.join("./experiments", f"{EXPERIMENT_NAME}_screening.csv")
    SNAPSHOT_DIR = os.path.join("./experiments/snapshots", EXPERIMENT_NAME)
    SAVE_INTERVAL = 10
    NUM_EPOCHS = 200
    RESUME = True
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

    @classmethod
    def validate(cls):
        checks = {
            "DOWNSAMPLE_STRIDES": cls.DOWNSAMPLE_STRIDES,
            "EMBEDDING_DIM_LIST": cls.EMBEDDING_DIM_LIST,
            "NUM_EMBEDDINGS_LIST": cls.NUM_EMBEDDINGS_LIST,
            "LAYER_LOSS_WEIGHTS_INIT": cls.LAYER_LOSS_WEIGHTS_INIT,
            "LAYER_LOSS_WEIGHTS_FINAL": cls.LAYER_LOSS_WEIGHTS_FINAL,
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

    @classmethod
    def architecture_summary(cls):
        return {
            "experiment_name": cls.EXPERIMENT_NAME,
            "experiment_stage": cls.EXPERIMENT_STAGE,
            "unet_depth": cls.UNET_DEPTH,
            "downsample_strides": list(cls.DOWNSAMPLE_STRIDES),
            "total_downsample": math.prod(cls.DOWNSAMPLE_STRIDES),
            "estimated_source_bpp": cls.ESTIMATED_SOURCE_BPP,
            "embedding_dim_list": list(cls.EMBEDDING_DIM_LIST),
            "num_embeddings_list": list(cls.NUM_EMBEDDINGS_LIST),
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
            "use_bottleneck_attention": cls.USE_BOTTLENECK_ATTENTION,
            "bottleneck_attention_blocks": cls.BOTTLENECK_ATTENTION_BLOCKS,
            "mse_loss_weight": cls.MSE_LOSS_WEIGHT,
            "ms_ssim_loss_weight": cls.MS_SSIM_LOSS_WEIGHT,
            "checkpoint_dir": cls.CHECKPOINT_DIR,
        }
