"""Configuration found on disk before quality-v1-k64 was started."""

import os
import torch


class Config:
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    NUM_DOWNSAMPLE_BLOCKS = 2
    BASE_CHANNELS = 64
    EMBEDDING_DIM_LIST = [128, 256]
    NUM_EMBEDDINGS_LIST = [128, 128]
    COMMITMENT_COST = 0.25
    DOWNSAMPLE_STRIDES = [4, 2]
    LAYER_LOSS_WEIGHTS_INIT = [1, 5]
    LAYER_LOSS_WEIGHTS_FINAL = [1, 1]
    SKIP_DROPOUT_P_INIT = [0.5]
    SKIP_DROPOUT_P_FINAL = [0.0]
    PHASE1_END = 0.6
    PHASE2_END = 0.9
    LEARNING_RATE_G = 1.75e-5
    CODEBOOK_PROJ_LR = 1.75e-4
    BETAS = (0.5, 0.999)
    CHANNEL_CODING_RATE_TRAIN = 0.5
    CHANNEL_CODING_RATE_VAL = 0.5
    BLOCK_LENGTH = 256
    SNR_RANGE_DB = [0, 15]
    CHANNEL_TYPE = "AWGN"
    RICIAN_K_FACTOR = 10
    TOTAL_BATCH_SIZE = 24
    MICRO_BATCH_SIZE = 2
    NUM_WORKERS = 8
    PIN_MEMORY = True
    TRAIN_DATASET_PATH = "/workspace/yi/work/Cars196/train_data"
    VAL_DATASET_PATH = "/workspace/yi/work/Cars196/val_data"
    TEST_DATASET_PATH = "/workspace/yi/work/Kodak"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    SAVE_INTERVAL = 20
    NUM_EPOCHS = 400
    RESUME = True
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
