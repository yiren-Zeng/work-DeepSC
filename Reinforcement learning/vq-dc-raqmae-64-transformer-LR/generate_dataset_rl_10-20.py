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

OUTPUT_FILE = "optimal_raq_codebook_rl_dataset_new_3.csv"
CHECKPOINT_PATH = "/home/yi/wk-1/vq-dc-raqmae-64-transformer-LR/checkpoints/best_vq_deepsc.pth"

# 构造测试 SNR 范围
snr_list = np.arange(10.0, 20.0, 0.1)  # np.ndarray -> [10.0, 10.1, ..., 19.9]


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

    columns = ["SNR", "Bit_Budget_Range", "K1", "K2", "K3", "K4", "Actual_Bits", "Best_Reward"]

    # ================= 新增逻辑：断点续传检测 =================
    processed_snrs = set()
    if os.path.exists(OUTPUT_FILE):
        try:
            # 读取已存在的文件，获取已经跑完的 SNR 列表
            df_existing = pd.read_csv(OUTPUT_FILE)
            if not df_existing.empty and "SNR" in df_existing.columns:
                # 浮点数可能存在精度问题，保留4位小数存入集合以便比对
                processed_snrs = set(df_existing["SNR"].round(4).tolist())
                print(f"已找到现有文件，跳过已处理的 {len(processed_snrs)} 个 SNR 点。")
        except Exception as e:
            print(f"读取现有文件时出错: {e}，将在原文件后继续追加。")
    else:
        # 如果文件不存在，先创建一个带表头的空文件
        pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False)

    # 过滤掉已经跑过的 SNR
    snr_list_to_run = [snr for snr in snr_list if round(snr, 4) not in processed_snrs]

    if not snr_list_to_run:
        print("所有 SNR 点均已处理完毕！")
        return
    # =======================================================

    # 遍历筛选后尚未处理的 SNR 列表
    for i, current_snr in enumerate(tqdm(snr_list_to_run)):

        SNR_CACHE = {}
        agent = EpsGreedyAgent(actions=actions, eps_start=0.5, eps_end=0.05, decay=30)
        best_reward = -1.0
        best_action = None

        for ep in range(1, BANDIT_EPISODES + 1):
            action = agent.select()
            action_key = tuple(action)

            if action_key in SNR_CACHE:
                reward = SNR_CACHE[action_key]
            else:
                reward, _ = evaluate_metrics_with_channel(
                    deepsc_model,
                    test_dataloader,
                    action,
                    current_snr,
                    ldpc_code,
                    device
                )
                SNR_CACHE[action_key] = reward

            agent.update(action, reward)

            if reward > best_reward:
                best_reward = reward
                best_action = action

        actual_bits = calculate_total_bits(best_action, feature_dims)

        # ================= 新增逻辑：实时追加写入文件 =================
        # 整理当前单个 SNR 的结果数据
        row_data = [[
            round(current_snr, 4),
            f"{BUDGET_MIN}-{BUDGET_MAX}",
            int(best_action[0]),
            int(best_action[1]),
            int(best_action[2]),
            int(best_action[3]),
            actual_bits,
            round(best_reward, 6)
        ]]

        # 以追加模式 (mode='a') 写入文件，不写入表头 (header=False)
        pd.DataFrame(row_data, columns=columns).to_csv(OUTPUT_FILE, mode='a', header=False, index=False)
        # ===========================================================


if __name__ == "__main__":
    main()