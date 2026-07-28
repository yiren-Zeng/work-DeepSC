import torch

class Config:
    IN_CHANNELS = 3  # 输入图像通道
    OUT_CHANNELS = 3 # 输出图像通道
    NUM_DOWNSAMPLE_BLOCKS = 4 # 下采样块数量
    BASE_CHANNELS = 64 # 初始化语义编码器通道数

    EMBEDDING_DIM_LIST = [128, 256, 512, 1024] # 第一个必须是BASE_CHANNELS的两倍，不然就改BASE_CHANNELS
    NUM_EMBEDDINGS_LIST = [64, 64, 64, 64] # N0, N1, N2, N3
    COMMITMENT_COST = 0.25 # 向量量化承诺成本系数
    RAQ_TARGET_LIST= [64, 64, 64, 64] # RAQ 大小
    RAQ_TARGET_LIST_BEST = [32, 128, 64, 32]
    RAQ_MIN_TRG = 2 # 因为Cars196数据集最接近ImageNet，所以这里取32
    RAQ_MAX_TRG = 2048 # ImageNet（256×256）：RAQ 的适配码本大小范围为32~4096
    RAQ_SYNC_EVERY= 100

    LAMBDA_WEIGHT = 0.1
    LEARNING_RATE_G = 1.75e-5 # VQ-DeepSC model的学习率,之前是e-4的，但出现了nan
    LEARNING_RATE_D = 1e-5 # Discriminator 的学习率
    BETAS = (0.5, 0.999) # Adam优化器的beta参数
    BATCH_SIZE = 24
    NUM_EPOCHS = 400

    

    TRAIN_DATASET_PATH = "/home/yi/.conda/envs/worka/xiaoxin/VQ-DeepSC-3/work-vq-deepsc/vq_deepsc/data/Cars196" # Updated path for Cars196 dataset
    VAL_DATASET_PATH = "/home/yi/.conda/envs/worka/xiaoxin/VQ-DeepSC-3/work-vq-deepsc/vq_deepsc/data/Validation_set"
    TEST_DATASET_PATH = "/home/yi/.conda/envs/worka/xiaoxin/VQ-DeepSC-3/work-vq-deepsc/vq_deepsc/data/Kodak" # Updated path for Kodak dataset
    
    # 调制的信道参数
    SNR_RANGE_DB = [0, 20] # 测试需要的SNR
    CHANNEL_TYPE = "AWGN" # "AWGN" or "Rician"
    RICIAN_K_FACTOR = 10
    
    # Device
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    SAVE_INTERVAL = 20






