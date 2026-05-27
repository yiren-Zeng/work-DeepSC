import torch
import os


class Config:
    IN_CHANNELS = 3
    OUT_CHANNELS = 3
    NUM_DOWNSAMPLE_BLOCKS = 4
    BASE_CHANNELS = 64

    # 4层特征通道列表（与编码器输出对应）
    EMBEDDING_DIM_LIST = [128, 256, 512, 1024]

    # 各层跳跃连接 Dropout 概率 [Layer0, Layer1, Layer2]
    # Layer0(浅层) 丢弃最多，Layer2(深层) 丢弃最少
    SKIP_DROPOUT_P = [0.5, 0.45, 0.05]

    LEARNING_RATE_G = 1.75e-5
    BETAS = (0.5, 0.999)

    # === 信道参数 ===
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
