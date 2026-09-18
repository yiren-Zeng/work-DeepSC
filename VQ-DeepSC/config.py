import os

import torch


def _env_int_list(name, default):
    value = os.environ.get(name)
    if not value:
        return list(default)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # Model
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    NUM_DOWNSAMPLE_BLOCKS = 4
    BASE_CHANNELS = 64

    EMBEDDING_DIM_LIST = [128, 256, 512, 1024]
    NUM_EMBEDDINGS_LIST = _env_int_list(
        "VQ_DEEPSC_NUM_EMBEDDINGS_LIST", [64, 64, 64, 64]
    )
    COMMITMENT_COST = 0.25

    # Optimization
    LEARNING_RATE = 1.75e-5
    BETAS = (0.5, 0.999)
    TOTAL_BATCH_SIZE = 24
    MICRO_BATCH_SIZE = 24
    NUM_EPOCHS = int(os.environ.get("VQ_DEEPSC_NUM_EPOCHS", "400"))

    # Data loading
    NUM_WORKERS = 8
    PIN_MEMORY = True
    TRAIN_DATASET_PATH = "/workspace/yi/work/Cars196/train_data"
    VAL_DATASET_PATH = "/workspace/yi/work/Cars196/val_data"
    TEST_DATASET_PATH = "/workspace/yi/work/Kodak-256-transform-resize"
    TEST_BATCH_SIZE = 1

    # Runtime and outputs
    DEVICE = os.environ.get(
        "VQ_DEEPSC_DEVICE", "cuda:2" if torch.cuda.is_available() else "cpu"
    )
    EXPERIMENT_NAME = os.environ.get(
        "VQ_DEEPSC_EXPERIMENT_NAME",
        "k" + "_".join(str(value) for value in NUM_EMBEDDINGS_LIST),
    )
    CHECKPOINT_DIR = os.environ.get(
        "VQ_DEEPSC_CHECKPOINT_DIR",
        os.path.join("./checkpoints", EXPERIMENT_NAME),
    )
    LOG_DIR = os.environ.get(
        "VQ_DEEPSC_LOG_DIR",
        os.path.join("./logs", EXPERIMENT_NAME),
    )
    RESUME = _env_bool("VQ_DEEPSC_RESUME", True)
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")

    @classmethod
    def validate(cls):
        if cls.NUM_EPOCHS < 1:
            raise ValueError("NUM_EPOCHS must be positive")
        if len(cls.NUM_EMBEDDINGS_LIST) != cls.NUM_DOWNSAMPLE_BLOCKS:
            raise ValueError(
                "NUM_EMBEDDINGS_LIST length must equal NUM_DOWNSAMPLE_BLOCKS"
            )
        for value in cls.NUM_EMBEDDINGS_LIST:
            if value < 2 or value & (value - 1):
                raise ValueError("Every codebook size must be a power of two >= 2")
