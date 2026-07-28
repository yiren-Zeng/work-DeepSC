import torch
from torchvision.utils import save_image
import os
import numpy as np
from config import Config
from models.vq_deepsc import VQDeepSC
from data.datasets import get_dataloader
from utils.metrics import calculate_ms_ssim


def test_layer_impact():
    # === 配置加载 ===
    cfg = Config()
    device = torch.device(cfg.DEVICE)

    # 确保输出目录存在
    output_dir = "./layer_impact_analysis"
    os.makedirs(output_dir, exist_ok=True)

    # === 模型初始化 ===
    vq_deepsc_model = VQDeepSC(
        in_channels=cfg.IN_CHANNELS,
        out_channels=cfg.OUT_CHANNELS,
        num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,
        base_channels=cfg.BASE_CHANNELS,
        num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,
        embedding_dim_list=cfg.EMBEDDING_DIM_LIST,
        commitment_cost=cfg.COMMITMENT_COST,
        raq_min_trg=cfg.RAQ_MIN_TRG,
        raq_max_trg=cfg.RAQ_MAX_TRG,
        device=cfg.DEVICE
    ).to(device)

    # 加载权重 (请修改为您实际的权重路径)
    checkpoint_path = "/home/yi/wk-1/vq-dc-raqmae-64/checkpoints/best_vq_deepsc.pth"
    print(f"Loading checkpoint from {checkpoint_path}")
    vq_deepsc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    vq_deepsc_model.eval()

    # === 数据加载 ===
    test_dataloader = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test'
    )

    # === 定义测试方案 ===
    # 假设这是您想使用的 RAQ 配置 (全1024 或 根据您的需求修改)
    # None 代表使用 Source 码本
    fixed_trg_list = cfg.RAQ_TARGET_LIST

    # 定义您想测试的组合
    # key 是保存图片和打印日志用的名字
    # value 是 active_indices 列表，表示哪些层是“开启”的
    layer_scenarios = {
        # --- 单独层测试 (Individual) ---
        "Only_Layer_0": [0],  # 浅层 (高分辨率)
        "Only_Layer_1": [1],
        "Only_Layer_2": [2],
        "Only_Layer_3": [3],  # 深层 (低分辨率/语义层)

        # --- 累积层测试 (Cumulative) ---
        # 注意：顺序取决于您的 encoder 输出。
        # 通常 Index 0 是最浅层，Index 3 是最深层。
        # 如果您是“从深到浅”叠加（通常恢复语义再恢复细节）：
        # "Deepest_Only(L3)": [3],
        # "L3_and_L2": [3, 2],
        # "L3_L2_L1": [3, 2, 1],
        # "All_Layers": [3, 2, 1, 0],

        # 如果您想测试“从浅到深”（虽然较少见，但可以测）：
        "First_1_Layer(L0)": [0],
        "First_2_Layers(L0+1)": [0, 1],
        "First_3_Layers(L0+1+2)": [0, 1, 2],
        "All_Layers(L0+1+2+3)": [0, 1, 2, 3],
        "(L0+2)": [0, 2],
        "(L0+3)": [0, 3],
        "(L1+2)": [1, 2],
        "(L1+3)": [1, 3],
        "(L2+3)": [2, 3],
        "L0+1+3": [0, 1, 3],
        "L0+2+3": [0, 2, 3],
        "L1+2+3": [1, 2, 3]
    }

    print(f"Starting Layer Impact Analysis on {device}...")

    # 存储结果用于计算平均值
    scenario_scores = {k: [] for k in layer_scenarios.keys()}

    with torch.no_grad():

        for i, real_image in enumerate(test_dataloader):

            real_image = real_image.to(device)
            img_name_prefix = f"img_{i + 1}"
            print(f"Processing {img_name_prefix}...")

            # 保存原图一次
            if i == 0:
                save_image((real_image + 1) / 2, os.path.join(output_dir, "original.png"))

            # 遍历所有定义的场景
            for name, active_indices in layer_scenarios.items():

                # 调用我们在 deepsc.py 新加的函数
                recon_img = vq_deepsc_model.forward_test_mask_layers(
                    real_image,
                    trg_list=fixed_trg_list,
                    active_indices=active_indices
                )

                # 计算 MS-SSIM
                score = calculate_ms_ssim(((real_image + 1) / 2), (recon_img + 1) / 2)
                score_db = -10 * np.log10(1 - score)
                scenario_scores[name].append(score_db)

                # 仅对第一张图片保存所有场景的重建结果，方便肉眼观察
                if i == 0:
                    file_name = f"{img_name_prefix}_{name}_{score_db:.2f}dB.png"
                    save_image((recon_img + 1) / 2, os.path.join(output_dir, file_name))

    # === 打印最终统计表格 ===
    print("\n" + "=" * 50)
    print(f"{'Scenario Name':<25} | {'Avg MS-SSIM (dB)':<20}")
    print("-" * 50)

    for name, scores in scenario_scores.items():
        avg_score = np.mean(scores)
        print(f"{name:<25} | {avg_score:.4f}")

    print("=" * 50)
    print(f"Results and images saved to {output_dir}")


if __name__ == "__main__":
    test_layer_impact()