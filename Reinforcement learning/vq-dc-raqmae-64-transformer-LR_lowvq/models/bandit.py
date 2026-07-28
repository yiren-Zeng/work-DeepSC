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
