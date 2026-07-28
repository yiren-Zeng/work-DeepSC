import os, math, random, itertools, argparse
import torch
from utils.math_utils import powers_of_two
from config import Config



# 动作空间生成器
def gen_candidates(total, layer_bounds, only_powers=True, max_cands=None):
    value_lists = []
    for lo, hi in layer_bounds:
        if only_powers:
            vs = powers_of_two(lo, hi)
        else:
            vs = list(range(lo, hi + 1))
        value_lists.append(vs)

    combos = []
    for k1, k2, k3, k4 in itertools.product(*value_lists):
        if (k1 + k2 + k3 + k4) == total:
            combos.append([k1, k2, k3, k4])

    if max_cands is not None and len(combos) > max_cands:
        random.shuffle(combos)
        combos = combos[:max_cands]
    return combos


def get_feature_map_dims(model, device):
    """动态获取各层特征图尺寸 (H, W)"""
    dummy_input = torch.zeros(1, Config.IN_CHANNELS, 768, 512).to(device)  # torch.Tensor -> [1, 3, 768, 512] 假输入
    model.eval()
    with torch.no_grad():
        features = model.semantic_encoder(
            dummy_input)  # list -> [Tensor(1, 128, 384, 256), Tensor(1, 256, 192, 128), ...] 四层特征图

    # 提取特征图的长宽
    return [(f.shape[2], f.shape[3]) for f in features]  # list -> [(H1, W1), (H2, W2), (H3, W3), (H4, W4)]


def calculate_total_bits(k_list, feature_dims):
    """计算真实消耗的物理比特数"""
    total_bits = 0  # int -> 标量，总比特数
    for i, k in enumerate(k_list):
        h, w = feature_dims[i]  # int, int -> 单层特征图的高和宽
        bits_per_pixel = int(math.log2(k))  # int -> 单个特征像素需要的比特数
        total_bits += h * w * bits_per_pixel
    return total_bits  # int -> 标量，该码本组合下的总比特数

# @torch.no_grad()
# def evaluate_ms_ssim(model, loader, ks, max_batches=24, device="cuda:1"):
#     """
#     修复MS-SSIM计算，确保与test-3.py完全一致
#     """
#     model.eval()
#     total, cnt = 0.0, 0
#     for bi, real_image in enumerate(loader):
#         real_image = real_image.to(device)
#         out = model.forward_test_raq(real_image, ks)
#
#         # 关键修复：确保与test-3.py完全相同的计算
#         real_normalized = (real_image + 1) / 2
#         rec_normalized = (out["reconstructed_images_raq"] + 1) / 2
#
#         # 添加数值稳定性检查
#         real_normalized = torch.clamp(real_normalized, 0.0, 1.0)
#         rec_normalized = torch.clamp(rec_normalized, 0.0, 1.0)
#
#         ms = calculate_ms_ssim(real_normalized, rec_normalized)
#         total += float(ms)
#         cnt += 1
#         if max_batches is not None and cnt >= max_batches:
#             break
#     return total / max(cnt, 1)


# def verify_ms_ssim_consistency(model, test_dataloader, config, device):
#     """
#     快速验证MS-SSIM计算一致性
#     """
#     print(f"验证配置 {config} 的MS-SSIM一致性...")
#
#     # Bandit方式计算
#     bandit_score = evaluate_ms_ssim(model, test_dataloader, config,
#                                     max_batches=len(test_dataloader), device=device) # 这个24是测试数据集的数量
#
#     # test-3.py方式计算
#     model.eval()
#     test_scores = []
#     with torch.no_grad():
#         for i, real_image in enumerate(test_dataloader):
#             real_image = real_image.to(device)
#             out = model.forward_test_raq(real_image, config)
#             ms_ssim = calculate_ms_ssim(
#                 (real_image + 1) / 2,
#                 (out["reconstructed_images_raq"] + 1) / 2
#             )
#             test_scores.append(float(ms_ssim))
#
#     test3_avg = np.mean(test_scores) if test_scores else 0
#
#     print(f"Bandit: {bandit_score:.6f}, test-3.py: {test3_avg:.6f}, 差异: {abs(bandit_score - test3_avg):.6f}")
#
#     if abs(bandit_score - test3_avg) < 0.0001:
#         print("✓ MS-SSIM一致性验证通过")
#         return True
#     else:
#         print("✗ MS-SSIM计算存在不一致！")
#         return False