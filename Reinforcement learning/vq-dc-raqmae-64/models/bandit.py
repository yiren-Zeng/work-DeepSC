# import os, math, random, itertools, argparse
# import torch
# from communications.metrics import calculate_ms_ssim
#
# # ======================= 辅助：候选生成 & Bandit =======================
#
#
# def set_seed(seed: int):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#
# def powers_of_two(lo: int, hi: int):
#     vals = []
#     v = 1
#     while v < lo:
#         v <<= 1
#     while v <= hi:
#         vals.append(v)
#         v <<= 1
#     return vals
#
# def gen_candidates(total, layer_bounds, only_powers=True, max_cands=None):
#     """
#     生成满足 sum(Ki)=total 的候选组合。
#     layer_bounds: [(lo1,hi1), (lo2,hi2), (lo3,hi3), (lo4,hi4)]
#     only_powers:  是否仅使用 2 的幂
#     max_cands:    候选过多时随机截断
#     """
#     value_lists = []
#     for lo, hi in layer_bounds:
#         if only_powers:
#             vs = powers_of_two(lo, hi)
#         else:
#             vs = list(range(lo, hi + 1))
#         value_lists.append(vs)
#
#     combos = []
#     for k1, k2, k3, k4 in itertools.product(*value_lists):
#         if (k1 + k2 + k3 + k4) == total:
#             combos.append([k1, k2, k3, k4])
#
#     if max_cands is not None and len(combos) > max_cands:
#         random.shuffle(combos)
#         combos = combos[:max_cands]
#     return combos
#
# class EpsGreedyAgent:
#     def __init__(self, actions, eps_start=0.4, eps_end=0.05, decay=30):
#         self.actions = [tuple(a) for a in actions]
#         self.Q = {a: 0.0 for a in self.actions}
#         self.N = {a: 0   for a in self.actions}
#         self.eps_start, self.eps_end, self.decay = eps_start, eps_end, decay
#         self.t = 0
#
#     def epsilon(self):
#         return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.t / self.decay)
#
#     def select(self):
#         self.t += 1
#         if random.random() < self.epsilon():
#             return list(random.choice(self.actions))
#         return list(max(self.actions, key=lambda a: self.Q[a]))
#
#     def update(self, a, r):
#         a = tuple(a)
#         self.N[a] += 1
#         self.Q[a] += (r - self.Q[a]) / self.N[a]
#
# # ======================= 评测：固定 K 计算 MS-SSIM =======================
#
# @torch.no_grad()
# def evaluate_ms_ssim(model, loader, ks, max_batches=10, device="cuda:1"):
#     """
#     固定一组 ks=[K1..K4]，计算若干 batch 的平均 MS-SSIM。
#     默认使用 RAQ 重建图像："reconstructed_images_raq"。
#     如需对比源码本，可改成 "reconstructed_images_src"。
#     """
#     model.eval()
#     # 你的模型 forward_test_raq 接受目标列表（你原 test.py 就是这么用的）
#     total, cnt = 0.0, 0
#     for bi, real_image in enumerate(loader):
#         real_image = real_image.to(device)
#         out = model.forward_test_raq(real_image, ks)
#         # --- 用 RAQ 重建来当评测目标 ---
#         ms = calculate_ms_ssim(((real_image+1)/2), (out["reconstructed_images_raq"]+1)/ 2)  # 你的项目自带的 MS-SSIM
#         total += float(ms)
#         cnt += 1
#         if max_batches is not None and cnt >= max_batches:
#             break
#     return total / max(cnt, 1)

# [file name]: bandit.py
# [file content begin]
import os, math, random, itertools, argparse
import torch
import numpy as np
from utils.metrics import calculate_ms_ssim


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def powers_of_two(lo: int, hi: int):
    vals = []
    v = 1
    while v < lo:
        v <<= 1
    while v <= hi:
        vals.append(v)
        v <<= 1
    return vals


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


class EpsGreedyAgent:
    def __init__(self, actions, eps_start=0.4, eps_end=0.05, decay=30):
        self.actions = [tuple(a) for a in actions]
        self.Q = {a: 0.0 for a in self.actions}
        self.N = {a: 0 for a in self.actions}
        self.eps_start, self.eps_end, self.decay = eps_start, eps_end, decay
        self.t = 0

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.t / self.decay)

    def select(self):
        self.t += 1
        if random.random() < self.epsilon():
            # 优先选择未探索的动作
            unexplored = [a for a in self.actions if self.N[a] == 0]
            if unexplored:
                return list(random.choice(unexplored))
            else:
                return list(random.choice(self.actions))
        return list(max(self.actions, key=lambda a: self.Q[a]))

    def update(self, a, r):
        a = tuple(a)
        self.N[a] += 1
        self.Q[a] += (r - self.Q[a]) / self.N[a]


@torch.no_grad()
def evaluate_ms_ssim(model, loader, ks, max_batches=24, device="cuda:1"):
    """
    修复MS-SSIM计算，确保与test-3.py完全一致
    """
    model.eval()
    total, cnt = 0.0, 0
    for bi, real_image in enumerate(loader):
        real_image = real_image.to(device)
        out = model.forward_test_raq(real_image, ks)

        # 关键修复：确保与test-3.py完全相同的计算
        real_normalized = (real_image + 1) / 2
        rec_normalized = (out["reconstructed_images_raq"] + 1) / 2

        # 添加数值稳定性检查
        real_normalized = torch.clamp(real_normalized, 0.0, 1.0)
        rec_normalized = torch.clamp(rec_normalized, 0.0, 1.0)

        ms = calculate_ms_ssim(real_normalized, rec_normalized)
        total += float(ms)
        cnt += 1
        if max_batches is not None and cnt >= max_batches:
            break
    return total / max(cnt, 1)


def verify_ms_ssim_consistency(model, test_dataloader, config, device):
    """
    快速验证MS-SSIM计算一致性
    """
    print(f"验证配置 {config} 的MS-SSIM一致性...")

    # Bandit方式计算
    bandit_score = evaluate_ms_ssim(model, test_dataloader, config,
                                    max_batches=len(test_dataloader), device=device) # 这个24是测试数据集的数量

    # test-3.py方式计算
    model.eval()
    test_scores = []
    with torch.no_grad():
        for i, real_image in enumerate(test_dataloader):
            real_image = real_image.to(device)
            out = model.forward_test_raq(real_image, config)
            ms_ssim = calculate_ms_ssim(
                (real_image + 1) / 2,
                (out["reconstructed_images_raq"] + 1) / 2
            )
            test_scores.append(float(ms_ssim))

    test3_avg = np.mean(test_scores) if test_scores else 0

    print(f"Bandit: {bandit_score:.6f}, test-3.py: {test3_avg:.6f}, 差异: {abs(bandit_score - test3_avg):.6f}")

    if abs(bandit_score - test3_avg) < 0.0001:
        print("✓ MS-SSIM一致性验证通过")
        return True
    else:
        print("✗ MS-SSIM计算存在不一致！")
        return False