import torch
import os


class Config:
    IN_CHANNELS = 3
    OUT_CHANNELS = 3

    # === Swin Transformer 编解码器参数 ===
    PATCH_SIZE = 2  # 总下采样 = stem_stride(2) * 2^0 = 2x，与原项目一致
    WINDOW_SIZE = 4
    NUM_HEADS = 4
    NUM_BLOCKS = 2  # 每个 stage 内的 SwinBlock 数量

    # 1层 Swin 配置（与原项目1层 U-Net 严格对齐：1层 + 2x 总压缩率）
    # 原项目: init_conv(3→64) + DownSampleBlock(64→128, 2x)
    # 新项目: PatchEmbed(3→128, 2x) + 1个SwinStage(128, 2block)
    STAGE_EMBED_DIMS = (128,)
    STAGE_DEPTHS = (2,)
    STAGE_NUM_HEADS = (4,)
    STEM_STRIDE = 2

    EMBED_DIM = 128  # 瓶颈通道维度

    # 1层: 只有1组参数
    EMBEDDING_DIM_LIST = [128]
    NUM_EMBEDDINGS_LIST = [64]
    COMMITMENT_COST = 0.25

    LEARNING_RATE_G = 1.75e-5
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
