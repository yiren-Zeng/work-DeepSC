import os
import torch
import random
import numpy as np
import itertools
import math
from tqdm import tqdm

from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader
from models.bandit import get_feature_map_dims, calculate_total_bits
from communications.evaluate import evaluate_metrics_with_channel

# 引入物理層仿真和工具模塊
from communications.ldpc_coding import get_ldpc_code
from utils.math_utils import powers_of_two

# 全局緩存避免重複評估
EVALUATION_CACHE = {}

# 权重路径
CHECKPOINT_PATH = "/home/yi/wk-1/vq-dc-raqmae-64-transformer-LR/checkpoints/best_vq_deepsc.pth"

# 基於真實比特消耗生成動作空間 ===
BUDGET_MIN = 700416
BUDGET_MAX = 700417

BANDIT_EPISODES = 100 # int -> 总探索轮数

# 所选SNR
TARGET_SNR = -2

LDPC_N = 256
LDPC_R = 0.5


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


def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def test():
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

    feature_dims = get_feature_map_dims(deepsc_model, device) # [(H1, W1),(H2, W2),...,(HN, WN)]
    valid_ks = powers_of_two(cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG)  # list -> [32, 64, 128, 256, 512, 1024, 2048, 4096]
    all_possible_actions = list(itertools.product(valid_ks, repeat=4)) # list -> [(32,32,32,32), (32,32,32,64), ...] 穷举所有组合
    actions = [] # list -> 筛选后合法的组合集合

    for action in all_possible_actions:
        cost = calculate_total_bits(action, feature_dims)  # int -> 当前组合消耗比特数
        if BUDGET_MIN <= cost <= BUDGET_MAX:
            actions.append(list(action))  # list -> [[K1, K2, K3, K4], ...] 这些是符合预算范围内的K_trg组合

    print(f"[info] 比特預算區間: {BUDGET_MIN} - {BUDGET_MAX}")
    print(f"[info] 合法候選動作數: {len(actions)}")
    print(f"[info] 正在針對惡劣信道 (SNR={TARGET_SNR}dB) 尋找最優魯棒碼本分配...")

    agent = EpsGreedyAgent(actions, eps_start=0.4, eps_end=0.05, decay=30) # EpsGreedyAgent -> 强化学习智能体对象
    best_a, best_r = None, -1.0


    for ep in range(1, BANDIT_EPISODES + 1):
        a = agent.select() # list -> [K1, K2, K3, K4] 当前轮次智能体选出的分配方案
        action_key = tuple(a) # tuple -> (K1, K2, K3, K4) 转换为不可变元组以作为字典键

        if action_key in EVALUATION_CACHE:
            r = EVALUATION_CACHE[action_key] # float -> 标量，从缓存读取的得分
            cache_flag = "[緩存]" # str -> 字符串标识
        else:
            r, _ = evaluate_metrics_with_channel(
                model=deepsc_model, loader=test_dataloader, k_trg=a,
                target_snr=TARGET_SNR, ldpc_code=ldpc_code,
                device=device) # float -> 标量，真实跑模型算出的得分
            EVALUATION_CACHE[action_key] = r
            cache_flag = ""

        agent.update(a, r) # 无返回值 -> 智能体内部更新 Q 账本和 N 账本

        if r > best_r:
            best_r, best_a = r, list(a) # float, list -> 更新全局最佳分数和最佳组合
            print(f"🔥 新最佳: {best_a}, MS-SSIM: {best_r:.4f} (SNR={TARGET_SNR}dB)")

        print(f"[ep {ep:02d}] a={a} {cache_flag}, MS-SSIM={r:.4f}, best={best_a} ({best_r:.4f})")

    # 最终结果
    print("\n==== 搜索完成 ====")
    print(f"最佳分配: {best_a}")
    print(f"MS-SSIM: {best_r:.4f}")
    print(f"探索统计: {len(EVALUATION_CACHE)}/{len(actions)} 个配置已评估")


if __name__ == "__main__":
    test()