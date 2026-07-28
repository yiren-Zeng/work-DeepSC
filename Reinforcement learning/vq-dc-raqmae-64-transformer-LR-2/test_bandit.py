# import torch
# from torchvision.communications import save_image
# import os
# import numpy as np
# from config import Config
# from models.vq_deepsc import VQDeepSC
# from data.datasets import get_dataloader
# # from communications.channel import awgn_channel, rician_channel
# # from communications.modulation import bpsk_modulate, bpsk_demodulate, qpsk_modulate, qpsk_demodulate, qam16_modulate, qam16_demodulate
# # from communications.ldpc_coding import indices_to_bits, bits_to_indices, ldpc_encode, ldpc_decode, get_ldpc_code
# from communications.metrics import calculate_ms_ssim
# from models.bandit import *
#
#
# def test():
#
#     # 加载配置
#     cfg = Config()
#
#     # 设备设置
#     device = torch.device(cfg.DEVICE)
#
#     # 创建模型实例
#     # Create model instances
#     vq_deepsc_model = VQDeepSC(
#         in_channels=cfg.IN_CHANNELS,  # 输入通道数
#         out_channels=cfg.OUT_CHANNELS,  # 输出通道数
#         num_downsample_blocks=cfg.NUM_DOWNSAMPLE_BLOCKS,  # 下采样块数量
#         base_channels=cfg.BASE_CHANNELS,  # 基础通道数
#         num_embeddings_list=cfg.NUM_EMBEDDINGS_LIST,  # 向量量化字典大小列表
#         embedding_dim_list=cfg.EMBEDDING_DIM_LIST,  # 向量量化维度列表
#         commitment_cost=cfg.COMMITMENT_COST,  # 向量量化承诺成本系数
#         raq_min_trg=cfg.RAQ_MIN_TRG,  # RAQ目标最小值
#         raq_max_trg=cfg.RAQ_MAX_TRG,  # RAQ目标最大值
#         device=cfg.DEVICE  # 设备
#     ).to(device)  # 将模型移动到指定设备
#
#     checkpoint_path = os.path.join("/home/yi/wk-1/vq-dc-raqmae-64/checkpoints/vq_deepsc_epoch_180.pth")
#     print(f"Loading checkpoint from {checkpoint_path}")
#     vq_deepsc_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
#
#
#     # 设置模型为评估模式
#     vq_deepsc_model.eval()
#
#     # 测试数据加载器
#     test_dataloader = get_dataloader(
#         root_dir=cfg.TEST_DATASET_PATH,
#         batch_size=1,  # 测试时批处理大小为1
#         shuffle=False,
#         mode='test'
#     )
#
#
#     # 重建图像输出目录
#     output_dir = "./reconstructed_images"
#     os.makedirs(output_dir, exist_ok=True)
#
#     print(f"Starting testing on {device}...")
#
#     # 信噪比范围（dB）
#     snr_range_db = np.arange(cfg.SNR_RANGE_DB[0], cfg.SNR_RANGE_DB[1] + 1, 5)
#     ms_ssim_scores = []  # 初始化MS-SSIM分数列表
#
#     total_budget = int(sum(cfg.RAQ_TARGET_LIST))
#     b1 = (cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG)
#     b2 = (cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG)
#     b3 = (cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG)
#     b4 = (cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG)
#     layer_bounds = [b1, b2, b3, b4]
#
#     # 5.3 生成候选
#     actions = gen_candidates(total_budget, layer_bounds)
#
#     if len(actions) == 0:
#         raise RuntimeError(
#             f"没有满足 sum(Ki)={total_budget} 的候选。"
#             f"请放宽每层范围（--b1..--b4）或关闭 --only-powers。"
#         )
#     print(f"[info] 候选数: {len(actions)}")
#
#     # 5.4 Bandit 初始化
#     agent = EpsGreedyAgent(actions, eps_start=0.4,
#                            eps_end=0.05, decay=30)
#
#     best_a, best_r = None, -1.0
#
#     episodes=100
#     # 5.5 迭代搜索（每个 ep 评估一个动作）
#     for ep in range(1, episodes + 1):
#         a = agent.select()  # 形如 [K1,K2,K3,K4]
#         r = evaluate_ms_ssim(vq_deepsc_model, test_dataloader, ks=a,
#                              max_batches=10, device=device)
#         agent.update(a, r)
#
#         if r > best_r:
#             best_r, best_a = r, a
#
#         print(f"[ep {ep:02d}] a={a}, MS-SSIM={r:.4f}, "
#               f"Q_est={agent.Q[tuple(a)]:.4f}, N={agent.N[tuple(a)]}, "
#               f"best={best_a} ({best_r:.4f})")
#
#     print("\n==== 搜索完成 ====")
#     print(f"最佳分配 a* = {best_a}，验证 MS-SSIM ≈ {best_r:.4f}")
#     print("推理/部署时可固定：vq_deepsc_model.set_val_targets(best_a) 或 forward_test_raq(..., best_a)")
#
#
# if __name__ == "__main__":
#     test()
#



import torch
from torchvision.utils import save_image
import os
import numpy as np
from config import Config
from models.deepsc import VQDeepSC
from data.datasets import get_dataloader
from utils.metrics import calculate_ms_ssim
from collections import OrderedDict
from models.bandit import *

# 全局缓存避免重复评估
EVALUATION_CACHE = {}


def test():
    # 设置随机种子确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)
    set_seed(42)

    # 加载配置
    cfg = Config()
    device = torch.device(cfg.DEVICE)

    # 创建模型实例
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

    checkpoint_path = os.path.join("/home/yi/wk-1/vq-dc-raqmae-64-transformer-last/checkpoints/best_vq_deepsc.pth")
    print(f"Loading checkpoint from {checkpoint_path}")

    # === 【修改开始】 ===
    # 1. 加载原始的 state_dict
    state_dict = torch.load(checkpoint_path, map_location=device)

    # 2. 创建一个新的字典，去掉 key 中的 'module.' 前缀
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        # 如果 key 是以 'module.' 开头，就去掉前 7 个字符
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    # 3. 使用处理后的字典加载权重
    vq_deepsc_model.load_state_dict(new_state_dict)
    # === 【修改结束】 ===

    vq_deepsc_model.eval()

    # 测试数据加载器
    test_dataloader,test_sampler = get_dataloader(
        root_dir=cfg.TEST_DATASET_PATH,
        batch_size=1,
        shuffle=False,
        mode='test'
    )

    output_dir = "./reconstructed_images"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting testing on {device}...")

    # Bandit参数设置
    total_budget = int(sum(cfg.RAQ_TARGET_LIST))
    layer_bounds = [
        (cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG),
        (cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG),
        (cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG),
        (cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG)
    ]

    # 生成候选
    actions = gen_candidates(total_budget, layer_bounds)
    if len(actions) == 0:
        raise RuntimeError(f"没有满足 sum(Ki)={total_budget} 的候选。")
    print(f"[info] 候选数: {len(actions)}")

    # 验证MS-SSIM一致性
    consistency_ok = verify_ms_ssim_consistency(vq_deepsc_model, test_dataloader,
                                                cfg.RAQ_TARGET_LIST, device)
    if not consistency_ok:
        print("警告: MS-SSIM计算不一致，可能影响结果准确性")

    # Bandit初始化
    agent = EpsGreedyAgent(actions, eps_start=0.4, eps_end=0.05, decay=30)
    best_a, best_r = None, -1.0
    episodes = 100

    print(f"\n开始Bandit搜索 ({episodes}轮)...")

    for ep in range(1, episodes + 1):
        a = agent.select()

        # 使用缓存避免重复评估
        action_key = tuple(a)
        if action_key in EVALUATION_CACHE:
            r = EVALUATION_CACHE[action_key]
            cache_flag = "[缓存]"
        else:
            r = evaluate_ms_ssim(vq_deepsc_model, test_dataloader, ks=a,
                                 max_batches=len(test_dataloader), device=device)
            EVALUATION_CACHE[action_key] = r
            cache_flag = ""

        agent.update(a, r)

        if r > best_r:
            best_r, best_a = r, a.copy()
            print(f"🔥 新最佳: {best_a}, MS-SSIM: {best_r:.4f}")

        # 进度显示
        print(f"[ep {ep:02d}] a={a} {cache_flag}, MS-SSIM={r:.4f}, "
              f"best={best_a} ({best_r:.4f})")

    # 最终结果
    print("\n==== 搜索完成 ====")
    print(f"最佳分配: {best_a}")
    print(f"MS-SSIM: {best_r:.4f}")
    print(f"探索统计: {len(EVALUATION_CACHE)}/{len(actions)} 个配置已评估")


if __name__ == "__main__":
    test()



