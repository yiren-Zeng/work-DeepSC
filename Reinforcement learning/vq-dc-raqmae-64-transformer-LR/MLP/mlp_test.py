import random
import torch
import numpy as np
import os
from mlp_train import CodebookPredictorMLP
from models.deepsc import DeepSC
from communications.evaluate import evaluate_metrics_with_channel
from config import Config
from data.datasets import get_dataloader
from communications.ldpc_coding import get_ldpc_code


# 本来的网络权重
CHECKPOINT_PATH = "/home/yi/wk-1/vq-dc-raqmae-64-transformer-LR/checkpoints/best_vq_deepsc.pth"
# 构建MLP的网络权重
WEIGHT_FILE = "/home/yi/wk-1/vq-dc-raqmae-64-transformer-LR/MLP/best_mlp_codebook.pth"

TARGET_SNR = -2
BUDGET_MIN = 700416
BUDGET_MAX = 700417

LDPC_N = 256
LDPC_R = 0.5


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@torch.no_grad()
def predict_best_codebook(snr, budget_min, budget_max, weight_path="best_mlp_codebook.pth"):
    """
    输入单个 SNR 和预算范围，瞬间输出 MLP 预测的最优码本组合。
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 1. 实例化模型并加载权重
    model = CodebookPredictorMLP().to(device)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到权重文件 {weight_path}，请先运行 mlp_train.py")

    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # 2. 严格对齐训练时的归一化逻辑
    # SNR 归一化: (SNR - Min) / (Max - Min)
    snr_min_train, snr_max_train = -10.0, 20.0
    norm_snr = (snr - snr_min_train) / (snr_max_train - snr_min_train)

    # 预算归一化: 除以 1000000.0
    norm_b_min = budget_min / 1000000.0
    norm_b_max = budget_max / 1000000.0

    # 3. 构造输入张量，形状为 [1, 3] (1个样本，3个特征)
    x_input = torch.tensor([[norm_snr, norm_b_min, norm_b_max]], dtype=torch.float32).to(device)

    # 4. 模型前向传播
    outputs = model(x_input)  # 输出形状: [1, 4, 8]

    # 5. 取出概率最大的类别索引
    predicted_indices = torch.argmax(outputs, dim=2).squeeze().cpu().numpy()  # 形状: [4]

    # 6. 将索引 0~7 还原为真实的物理 K 值 (32~4096)
    k_predictions = [int(2 ** (idx + 5)) for idx in predicted_indices]

    return k_predictions


if __name__ == "__main__":
    cfg = Config()
    device = torch.device(cfg.DEVICE)
    setup_seed(42)

    deepsc_model = DeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        raq_min_trg=cfg.RAQ_MIN_TRG,
        raq_max_trg=cfg.RAQ_MAX_TRG,
        device=device
    ).to(device)

    if os.path.exists(CHECKPOINT_PATH):
        deepsc_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    deepsc_model.eval()

    test_dataloader = get_dataloader(  # DataLoader -> 提供 [B, 3, 768, 512] 的数据流
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test',
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )


    LDPC_K = int(LDPC_N * LDPC_R)
    ldpc_code = get_ldpc_code(LDPC_K)


    print(f"📡 当前信道环境 -> SNR: {TARGET_SNR} dB")
    print(f"💰 当前比特预算 -> {BUDGET_MIN} - {BUDGET_MAX} Bits")

    # 调用函数，0.001秒拿到结果
    mlp_k = predict_best_codebook(
        snr=TARGET_SNR,
        budget_min=BUDGET_MIN,
        budget_max=BUDGET_MAX,
        weight_path=WEIGHT_FILE
    )

    print(f"✅ MLP 决策的最优码本组合: {mlp_k}")

    # 将 MLP 的答案送入真实物理信道进行检查
    mlp_ms_ssim, mlp_psnr = evaluate_metrics_with_channel(
        deepsc_model, test_dataloader, mlp_k, TARGET_SNR, ldpc_code, device)

    print(f"   -> 预测完成，正在验证其真实的MS-SSIM分数{mlp_ms_ssim:.4f}，PSNR分数{mlp_psnr:.4f}")
