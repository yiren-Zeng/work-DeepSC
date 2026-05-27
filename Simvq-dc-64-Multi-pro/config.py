import torch
import os


class Config:
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    NUM_DOWNSAMPLE_BLOCKS = 4
    BASE_CHANNELS = 64

    # 4层: 4组参数
    EMBEDDING_DIM_LIST = [128, 256, 512, 1024]
    NUM_EMBEDDINGS_LIST = [64, 64, 64, 64]
    COMMITMENT_COST = 0.25

    # 各层VQ损失权重 [Layer0, Layer1, Layer2, Layer3]
    # 初始值（阶段1），会按调度计划退火
    LAYER_LOSS_WEIGHTS_INIT = [1, 1, 5, 10]
    # 最终值（阶段3）
    LAYER_LOSS_WEIGHTS_FINAL = [1, 1, 1, 1]

    # 各层跳跃连接 Dropout 概率 [Layer0, Layer1, Layer2]
    # 初始值（阶段1），会按调度计划衰减
    SKIP_DROPOUT_P_INIT = [0.5, 0.45, 0.05]
    # 最终值（阶段2结束/阶段3）
    SKIP_DROPOUT_P_FINAL = [0.0, 0.0, 0.0]

    # === 训练阶段调度 ===
    # E_total=400: 阶段1 [0, 240), 阶段2 [240, 360), 阶段3 [360, 400]
    PHASE1_END = 0.6   # 0.6 * NUM_EPOCHS
    PHASE2_END = 0.9   # 0.9 * NUM_EPOCHS

    LEARNING_RATE_G = 1.75e-5
    # SimVQ 码本投影层 (ProjectedEmbedding.proj) 单独学习率
    CODEBOOK_PROJ_LR = 1.75e-4
    BETAS = (0.5, 0.999)

    # === 有限码长传输参数 ===
    CHANNEL_CODING_RATE_TRAIN = 0.5
    CHANNEL_CODING_RATE_VAL = 0.5
    BLOCK_LENGTH = 256

    SNR_RANGE_DB = [0, 15]
    CHANNEL_TYPE = "AWGN"
    RICIAN_K_FACTOR = 10

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
