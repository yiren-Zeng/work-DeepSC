import torch
import os


class Config:
    EXPERIMENT_NAME = "quality_v1_k64"
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    NUM_DOWNSAMPLE_BLOCKS = 2
    BASE_CHANNELS = 64
    EMBEDDING_DIM_LIST = [128, 256]
    # Preserve the baseline source rate (0.4688 BPP) for a meaningful quality comparison.
    NUM_EMBEDDINGS_LIST = [64, 64]
    COMMITMENT_COST = 0.25
    DOWNSAMPLE_STRIDES = [4, 2]
    # The baseline's VQ term dominated its reconstruction loss during early training.
    LAYER_LOSS_WEIGHTS_INIT = [0.25, 0.5]
    LAYER_LOSS_WEIGHTS_FINAL = [0.25, 0.25]
    # Retain robustness training without discarding half of the high-resolution path.
    SKIP_DROPOUT_P_INIT = [0.1]
    SKIP_DROPOUT_P_FINAL = [0.0]
    PHASE1_END = 0.1
    PHASE2_END = 0.4
    LEARNING_RATE_G = 5e-5
    CODEBOOK_PROJ_LR = 2e-4
    BETAS = (0.5, 0.999)
    CHANNEL_CODING_RATE_TRAIN = 0.5
    CHANNEL_CODING_RATE_VAL = 0.5
    BLOCK_LENGTH = 256
    SNR_RANGE_DB = [0, 15]
    CHANNEL_TYPE = "AWGN"
    RICIAN_K_FACTOR = 10
    TOTAL_BATCH_SIZE = 24
    MICRO_BATCH_SIZE = 24
    NUM_WORKERS = 8
    PIN_MEMORY = True
    TRAIN_DATASET_PATH = "/workspace/yi/work/Cars196/train_data"
    VAL_DATASET_PATH = "/workspace/yi/work/Cars196/val_data"
    TEST_DATASET_PATH = "/workspace/yi/work/Kodak"
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./experiments/checkpoints/quality_v1_k64"
    LOG_DIR = "./experiments/tensorboard/quality_v1_k64"
    SAVE_INTERVAL = 10
    NUM_EPOCHS = 200
    RESUME = True
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
