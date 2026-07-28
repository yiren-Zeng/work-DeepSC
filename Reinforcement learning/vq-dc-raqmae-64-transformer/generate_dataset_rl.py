import torch
import numpy as np
import pandas as pd
import itertools
import math
from tqdm import tqdm
import random
from config import Config
from models.deepsc import DeepSC
from data.datasets import get_dataloader
from utils.math_utils import powers_of_two
import os
from communications.ldpc_coding import get_ldpc_code, ldpc_encode, ldpc_decode
from models.bandit import get_feature_map_dims, calculate_total_bits
from communications.evaluate import evaluate_metrics_with_channel

OUTPUT_FILE = "optimal_raq_codebook_rl_dataset.csv"
CHECKPOINT_PATH = "/home/yi/wk-1/vq-dc-raqmae-64-transformer/checkpoints/best_vq_deepsc.pth"

# 构造测试 SNR 范围
snr_range_1 = np.arange(-10.0, 0.0, 0.1)  # np.ndarray -> [-10.0, -9.9, ..., -0.1]
snr_range_2 = np.arange(0.0, 10.0, 0.1)  # np.ndarray -> [0.0, 0.1, ..., 9.9]
snr_range_3 = np.arange(10.0, 20.0, 0.1)  # np.ndarray -> [10.0, 10.1, ..., 19.9]
snr_list = np.concatenate([snr_range_1, snr_range_2, snr_range_3])[:300]  # np.ndarray -> [300] 拼接后的一维数组

LDPC_N = 256
LDPC_R = 0.5

# 预算范围 (Bits)
BUDGET_MIN = 700000
BUDGET_MAX = 1800000
BANDIT_EPISODES = 100 # int -> 总探索轮数

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


def main():

    cfg = Config()
    device = torch.device(cfg.DEVICE)
    setup_seed(42)

    # ... (省略模型加载和LDPC初始化的标注，同 test_bandit.py) ...
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
        print("Weights loaded successfully.")
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

    feature_dims = get_feature_map_dims(deepsc_model, device)  # [(H1, W1),(H2, W2),...,(HN, WN)]
    valid_ks = powers_of_two(cfg.RAQ_MIN_TRG, cfg.RAQ_MAX_TRG)  # list -> [32, 64, 128, 256, 512, 1024, 2048, 4096]
    all_possible_actions = list(itertools.product(valid_ks, repeat=4))  # list -> [(32,32,32,32), (32,32,32,64), ...] 穷举所有组合
    actions = []  # list -> 筛选后合法的组合集合

    for action in all_possible_actions:
        cost = calculate_total_bits(action, feature_dims)  # int -> 当前组合消耗比特数
        if BUDGET_MIN <= cost <= BUDGET_MAX:
            actions.append(list(action))  # list -> [[K1, K2, K3, K4], ...] 这些是符合预算范围内的K_trg组合


    data_rows = []  # list -> 二维列表 [[行1数据], [行2数据]...]，用于存储最后写入CSV的数据

    for i, current_snr in enumerate(tqdm(snr_list)):  # current_snr: float -> 当前正在测试的信噪比数值

        SNR_CACHE = {}  # dict -> {(K1, K2, K3, K4): float} 每个SNR独立重置的缓存
        agent = EpsGreedyAgent(actions=actions, eps_start=0.5, eps_end=0.05, decay = 30)  # EpsGreedyAgent -> 针对当前SNR实例化的新智能体
        best_reward = -1.0  # float -> 当前SNR下的最高得分
        best_action = None  # tuple -> (K1, K2, K3, K4) 当前SNR下的最优分配

        for ep in range(1, BANDIT_EPISODES + 1):
            action = agent.select()  # list -> [K1, K2, K3, K4] 当前轮次智能体选出的分配方案
            action_key = tuple(action)  # tuple -> (K1, K2, K3, K4) 转换为不可变元组以作为字典键

            if action_key in SNR_CACHE:
                reward = SNR_CACHE[action_key]  # float -> 标量，直接读缓存
            else:
                reward, _ = evaluate_metrics_with_channel(
                    deepsc_model,
                    test_dataloader,
                    action,
                    current_snr,
                    ldpc_code,
                    device
                )  # float -> 标量，跑模型评测
                SNR_CACHE[action_key] = reward

            agent.update(action, reward)  # 无返回值 -> 智能体学习

            if reward > best_reward:
                best_reward = reward
                best_action = action

        actual_bits = calculate_total_bits(best_action, feature_dims)  # int -> 标量，最优动作对应的真实比特

        # 整理单行数据
        # row 结构 -> [float, str, int, int, int, int, int, float]
        data_rows.append([
            round(current_snr, 4),
            f"{BUDGET_MIN}-{BUDGET_MAX}",
            int(best_action[0]),
            int(best_action[1]),
            int(best_action[2]),
            int(best_action[3]),
            actual_bits,
            round(best_reward, 6)
        ])

    columns = ["SNR", "Bit_Budget_Range", "K1", "K2", "K3", "K4", "Actual_Bits", "Best_Reward"]  # list -> [str] CSV的表头

    # pd.DataFrame 将二维列表转换为 DataFrame 对象，并写入硬盘
    pd.DataFrame(data_rows, columns=columns).to_csv(OUTPUT_FILE, index=False)  # 文件输出 -> 生成 .csv 文件


if __name__ == "__main__":
    main()