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

    # === 三阶段 Dropout 衰落 & 损失权重退火计划 ===
    # 阶段边界（占总训练比例）
    PHASE1_RATIO = 0.6   # 阶段1结束：深层强制拓荒期
    PHASE2_RATIO = 0.9   # 阶段2结束：全层复苏与退火期

    # 阶段1初始值 [Layer0(浅), Layer1, Layer2(深)]
    SKIP_DROPOUT_P_INIT = [0.5, 0.45, 0.05]
    # 各层VQ损失权重 [Layer0, Layer1, Layer2, Layer3]
    LAYER_LOSS_WEIGHTS_INIT = [1, 1, 5, 10]

    # 阶段3目标值
    SKIP_DROPOUT_P_FINAL = [0.0, 0.0, 0.0]
    # 激进反转：重压高频（浅层），极致压榨量化误差
    LAYER_LOSS_WEIGHTS_FINAL = [1, 1, 1, 1]

    # 兼容旧代码的默认值（训练中会被动态覆盖）
    LAYER_LOSS_WEIGHTS = LAYER_LOSS_WEIGHTS_INIT
    SKIP_DROPOUT_P = SKIP_DROPOUT_P_INIT

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

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    SAVE_INTERVAL = 20
    NUM_EPOCHS = 400
    RESUME = True
    RESUME_PATH = os.path.join(CHECKPOINT_DIR, "last_checkpoint.pth")
