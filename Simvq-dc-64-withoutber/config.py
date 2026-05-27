import torch
import os


class Config:
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    NUM_DOWNSAMPLE_BLOCKS = 1
    BASE_CHANNELS = 64

    # 1层: 只有1组参数
    EMBEDDING_DIM_LIST = [128]
    NUM_EMBEDDINGS_LIST = [64]
    COMMITMENT_COST = 0.25

    LEARNING_RATE_G = 1.75e-5
    BETAS = (0.5, 0.999)

    # === 梯度累积配置 ===
    TOTAL_BATCH_SIZE = 24
    MICRO_BATCH_SIZE = 24

    NUM_WORKERS = 8
    PIN_MEMORY = True

    TRAIN_DATASET_PATH = "/workspace/yi/work/Cars196/train_data"
    VAL_DATASET_PATH = "/workspace/yi/work/Cars196/val_data"
    TEST_DATASET_PATH = "/workspace/yi/work/Kodak"

    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    SAVE_INTERVAL = 20
    NUM_EPOCHS = 400
    RESUME = True
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
