import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import OrderedDict
from tqdm import tqdm
import math

# === 【核心修正 1】导入 TensorFlow 并禁用其 GPU ===
# 因为TensorFlow一启动就会默认占满所有 GPU 显存，导致CUDA Out of Memory
import tensorflow as tf
try:
    # 强制让 TensorFlow 只在 CPU 上运行
    tf.config.set_visible_devices([], 'GPU')
    print("[Info] TensorFlow GPU disabled to prevent memory conflict with PyTorch.")
except Exception as e:
    print(f"[Warning] Could not disable TF GPU: {e}")

# === 1. 导入项目模块 ===
from config import Config
from models.deepsc import VQDeepSC
from data.datasets import get_dataloader
from utils.ldpc_coding import indices_to_bits
from utils.modulation import bpsk_modulate, qpsk_modulate
from utils.channel import awgn_channel

# 兼容不同路径的导入
from models.suit_ber import FiniteBlocklengthChannel
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder


# === 2. 辅助工具函数扩展 ===

def get_flexible_ldpc_code(n, k):
    """扩展 LDPC 功能，支持自定义 n 和 k"""
    # 强制将 n 设为偶数以适配 QPSK/Sionna 的某些约束
    if n % 2 != 0:
        n = n - 1

    # 初始化 Sionna 编码器 (运行在 CPU)
    encoder = LDPC5GEncoder(k=k, n=n)
    decoder = LDPC5GDecoder(encoder, num_iter=20)
    return {"encoder": encoder, "decoder": decoder, "k": k, "n": n}


def compute_llr(rx_symbols, snr_db, modulation=str):
    """计算 LLR (返回 PyTorch Tensor)"""
    snr_linear = 10 ** (snr_db / 10.0)
    noise_var = 1.0 / snr_linear
    noise_var = max(noise_var, 1e-10)

    # 确保输入在 CPU 上，以便与 Numpy 兼容或进行数学运算
    if rx_symbols.is_cuda:
        rx_symbols = rx_symbols.cpu()

    if modulation == 'bpsk':
        return 2 * rx_symbols.real / noise_var
    elif modulation == 'qpsk':
        scale = math.sqrt(2)
        llr_real = 2 * rx_symbols.real / noise_var * scale
        llr_imag = 2 * rx_symbols.imag / noise_var * scale
        return torch.stack([llr_real, llr_imag], dim=1).flatten()
    else:
        raise ValueError("仅支持 BPSK 和 QPSK")


def theoretical_modulation_ber(snr_db, modulation='bpsk'):
    """理论无编码误码率"""
    snr_linear = 10 ** (snr_db / 10.0)
    if modulation == 'bpsk':
        x = np.sqrt(2 * snr_linear)
    else:
        x = np.sqrt(snr_linear)
    return 0.5 * math.erfc(x / math.sqrt(2))


# === 3. 核心分析类 ===

class ModelBERAnalyzer:
    def __init__(self):
        self.cfg = Config()
        self.device = torch.device(self.cfg.DEVICE)

        # # 初始化 FBL 计算器 (用于任务 2)
        self.fbl_calculator = FiniteBlocklengthChannel(channel_coding_rate=Config.CHANNEL_CODING_RATE_TRAIN,
                                                       modulation_bits=Config.MODULATION_BITS,
                                                       coded_block_length_bits=Config.BLOCK_LENGTH,
                                                       device="cpu")
        print(f"Running PyTorch Model on device: {self.device}")

    def load_model(self):
        """加载模型权重"""
        print("\n=== Loading Model Weights ===")
        model = VQDeepSC(
            in_channels=self.cfg.IN_CHANNELS,
            out_channels=self.cfg.OUT_CHANNELS,
            num_downsample_blocks=self.cfg.NUM_DOWNSAMPLE_BLOCKS,
            base_channels=self.cfg.BASE_CHANNELS,
            num_embeddings_list=self.cfg.NUM_EMBEDDINGS_LIST,
            embedding_dim_list=self.cfg.EMBEDDING_DIM_LIST,
            commitment_cost=self.cfg.COMMITMENT_COST,
            raq_min_trg=self.cfg.RAQ_MIN_TRG,
            raq_max_trg=self.cfg.RAQ_MAX_TRG,
            device=self.cfg.DEVICE
        ).to(self.device)

        checkpoint_path = self.cfg.RESUME_PATH
        if not os.path.exists(checkpoint_path):
            backup_path = "/home/yi/wk-1/vq-dc-raqmae-64-transformer-last/checkpoints/vq_deepsc_epoch_60.pth"
            if os.path.exists(backup_path):
                checkpoint_path = backup_path

        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights!")
        else:
            print(f"Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)

            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict, strict=False)

        model.eval()
        return model

    def extract_source_bits(self, model):
        """提取特征比特流"""
        print("\n=== Extracting Source Bits from Model ===")
        try:
            test_loader = get_dataloader(
                root_dir=self.cfg.TEST_DATASET_PATH,
                batch_size=1,
                shuffle=False,
                mode='test',
                num_workers=self.cfg.NUM_WORKERS,
                pin_memory=self.cfg.PIN_MEMORY
            )
        except Exception as e:
            print(f"Error loading dataloader: {e}")
            return np.random.randint(0, 2, size=100000)

        all_bits = []
        with torch.no_grad():
            for i, img in enumerate(test_loader):
                if i >= 1: break
                img = img.to(self.device)

                out = model.forward_test_raq(img, self.cfg.RAQ_TARGET_LIST)
                indices = out['indices_src']

                flat_bits, _, _ = indices_to_bits(indices, self.cfg.NUM_EMBEDDINGS_LIST)
                all_bits.append(flat_bits)

        if len(all_bits) > 0:
            source_bits = np.concatenate(all_bits)
            print(f"Extracted {len(source_bits)} bits from VQ-DeepSC model features.")
            return source_bits
        else:
            return np.random.randint(0, 2, size=100000)

    def run_ber_analysis(self):
        # 1. 准备数据
        model = self.load_model()
        source_bits_pool = self.extract_source_bits(model)

        # # 释放显存
        # del model
        # torch.cuda.empty_cache()

        # 2. 实验配置
        snr_range = np.arange(-2, 11, 1)  # SNR dB
        ldpc_configs = [
            (256, 1 / 2),
            (512, 1 / 2),
            (256, 2 / 3),
            (512, 2 / 3)
        ]

        plt.figure(figsize=(12, 10))
        colors = ['r', 'g', 'b', 'm']
        markers = ['o', 's', '^', 'D']

        # 3. 绘制理论基准线(任务3）
        smooth_snr = np.linspace(-2, 10, 100)
        plt.semilogy(smooth_snr, [theoretical_modulation_ber(s, 'bpsk') for s in smooth_snr],
                     'k--', alpha=0.5, label='Theory BPSK (Uncoded)')
        plt.semilogy(smooth_snr, [theoretical_modulation_ber(s, 'qpsk') for s in smooth_snr],
                     'k-.', alpha=0.5, label='Theory QPSK (Uncoded)')

        # 4. 遍历配置进行仿真
        MODULATION = 'qpsk'

        for idx, (L, R) in enumerate(ldpc_configs):
            print(f"\nProcessing Config: L={L}, R={R:.2f}, Mod={MODULATION}")

            # --- Task 2: 有限码长理论曲线 ---
            fbl_ber_curve = []
            for s in smooth_snr:
                s_tensor = torch.tensor(s, dtype=torch.float32)
                L_tensor = torch.tensor(L, dtype=torch.float32)
                ber_val = self.fbl_calculator.compute_ber(s_tensor, L_tensor,rc=0.5)
                fbl_ber_curve.append(ber_val.item())

            color = colors[idx]
            plt.semilogy(smooth_snr, fbl_ber_curve, color=color, linestyle='-', alpha=0.6, linewidth=1.5,
                         label=f'Theory FBL (L={L}, R={R:.2f})')

            # --- Task 1: LDPC 仿真 ---
            k = int(L * R)
            code_dict = get_flexible_ldpc_code(n=L, k=k)
            L_actual = code_dict['n']

            ber_results = []

            for snr in tqdm(snr_range, desc=f"Simulating SNR"):
                max_bits_to_test = 50000
                num_blocks = min(len(source_bits_pool) // k, max_bits_to_test // k)
                if num_blocks == 0:
                    print("Warning: Not enough bits, skipping this config")
                    break

                bits_to_send = source_bits_pool[:num_blocks * k]

                # A. 准备数据: Numpy -> Torch (CUDA)
                bits_reshaped = bits_to_send.reshape(num_blocks, k).astype(np.float32)

                # B. Sionna Encoding (Runs on CPU because TF GPU is disabled)
                codewords_tf = code_dict['encoder'](bits_reshaped)
                # Convert to Torch Tensor on GPU for Modulation/Channel
                codewords = torch.from_numpy(codewords_tf.numpy()).float().to(self.device)

                # C. Modulate
                if MODULATION == 'bpsk':
                    tx_symbols = bpsk_modulate(codewords)
                else:
                    tx_symbols = qpsk_modulate(codewords)

                # D. Channel
                rx_symbols = awgn_channel(tx_symbols, snr)

                # E. Demodulate (LLR)
                # compute_llr returns a PyTorch Tensor
                rx_llr = compute_llr(rx_symbols, snr, modulation=MODULATION)

                # F. Decoding
                # 【核心修正】：Sionna Decoder 必须接收 TF Tensor 或 Numpy 数组
                # 之前直接传了 PyTorch Tensor 导致报错 'get_shape'
                # 修正：.cpu().numpy()
                rx_llr_reshaped = rx_llr.reshape(num_blocks, L_actual).cpu().numpy()

                decoded_bits_tf = code_dict['decoder'](rx_llr_reshaped)
                decoded_bits = decoded_bits_tf.numpy().flatten()

                # G. BER
                error_count = np.sum(bits_to_send != decoded_bits)
                ber = error_count / len(bits_to_send)
                ber_results.append(ber)

            if len(ber_results) == len(snr_range):
                plt.semilogy(snr_range, ber_results, color=color, marker=markers[idx], linestyle='--',
                             label=f'Model+LDPC Sim (L={L}, R={R:.2f})')

        plt.title(f"BER vs SNR Analysis\n(VQ-DeepSC Model Bits + {MODULATION.upper()} + LDPC)", fontsize=14)
        plt.xlabel("SNR (dB)", fontsize=12)
        plt.ylabel("Bit Error Rate (BER)", fontsize=12)
        plt.grid(True, which="both", linestyle='--', alpha=0.4)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.ylim(1e-5, 1)
        plt.tight_layout()

        save_path = "model_ber_analysis.png"
        plt.savefig(save_path, dpi=300)
        print(f"\nAnalysis Complete. Result saved to {save_path}")
        plt.show()


if __name__ == "__main__":
    analyzer = ModelBERAnalyzer()
    analyzer.run_ber_analysis()

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from collections import OrderedDict
# from tqdm import tqdm
# import math
#
# # === 【核心修正 1】导入 TensorFlow 并禁用其 GPU ===
# import tensorflow as tf
#
# try:
#     tf.config.set_visible_devices([], 'GPU')
#     print("[Info] TensorFlow GPU disabled.")
# except Exception as e:
#     print(f"[Warning] Could not disable TF GPU: {e}")
#
# # 强制 PyTorch 使用 CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
#
# # === 1. 导入项目模块 ===
# from config import Config
# from models.vq_deepsc import VQDeepSC
# from data.datasets import get_dataloader
# from communications.ldpc_coding import indices_to_bits
# from communications.modulation import bpsk_modulate, qpsk_modulate
# from communications.channel import awgn_channel
# from models.suit_ber import FiniteBlocklengthChannel
# from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
#
#
# # === 2. 辅助工具函数扩展 ===
# def get_flexible_ldpc_code(n, k):
#     if n % 2 != 0:
#         n = n - 1
#     encoder = LDPC5GEncoder(k=k, n=n)
#     decoder = LDPC5GDecoder(encoder, num_iter=20)
#     return {"encoder": encoder, "decoder": decoder, "k": k, "n": n}
#
#
# def compute_llr(rx_symbols, snr_db, modulation=str):
#     snr_linear = 10 ** (snr_db / 10.0)
#     noise_var = 1.0 / snr_linear
#     noise_var = max(noise_var, 1e-10)
#
#     if modulation == 'bpsk':
#         return 2 * rx_symbols.real / noise_var
#     elif modulation == 'qpsk':
#         scale = math.sqrt(2)
#         llr_real = 2 * rx_symbols.real / noise_var * scale
#         llr_imag = 2 * rx_symbols.imag / noise_var * scale
#         return torch.stack([llr_real, llr_imag], dim=1).flatten()
#     else:
#         raise ValueError("仅支持 BPSK 和 QPSK")
#
#
# def theoretical_modulation_ber(snr_db, modulation='bpsk'):
#     snr_linear = 10 ** (snr_db / 10.0)
#     if modulation == 'bpsk':
#         x = np.sqrt(2 * snr_linear)
#     else:
#         x = np.sqrt(snr_linear)
#     return 0.5 * math.erfc(x / math.sqrt(2))
#
#
# # === 3. 核心分析类 ===
# class ModelBERAnalyzer:
#     def __init__(self):
#         self.cfg = Config()
#         self.device = torch.device("cpu")  # 强制 CPU
#
#         # 【修正】处理 Config 可能缺失的属性
#         # 默认 QPSK (2 bits) 和 块长 256
#         mod_bits = getattr(Config, 'MODULATION_BITS', 2)
#         block_len = getattr(Config, 'BLOCK_LENGTH', 256)
#
#         self.fbl_calculator = FiniteBlocklengthChannel(
#             channel_coding_rate=0.5,  # 默认值，会被 loop 中的 rc=R 覆盖
#             modulation_bits=mod_bits,
#             coded_block_length_bits=block_len,
#             device="cpu"
#         )
#         print(f"Running Analysis on device: {self.device} (CPU)")
#
#     def load_model(self):
#         print("\n=== Loading Model Weights ===")
#         model = VQDeepSC(
#             in_channels=self.cfg.IN_CHANNELS,
#             out_channels=self.cfg.OUT_CHANNELS,
#             num_downsample_blocks=self.cfg.NUM_DOWNSAMPLE_BLOCKS,
#             base_channels=self.cfg.BASE_CHANNELS,
#             num_embeddings_list=self.cfg.NUM_EMBEDDINGS_LIST,
#             embedding_dim_list=self.cfg.EMBEDDING_DIM_LIST,
#             commitment_cost=self.cfg.COMMITMENT_COST,
#             raq_min_trg=self.cfg.RAQ_MIN_TRG,
#             raq_max_trg=self.cfg.RAQ_MAX_TRG,
#             device="cpu"
#         ).to(self.device)
#
#         checkpoint_path = self.cfg.RESUME_PATH
#         if not os.path.exists(checkpoint_path):
#             backup_path = "/home/yi/wk-1/vq-dc-raqmae-64-transformer-last/checkpoints/vq_deepsc_epoch_60.pth"
#             if os.path.exists(backup_path):
#                 checkpoint_path = backup_path
#
#         if not os.path.exists(checkpoint_path):
#             print(f"Warning: Checkpoint not found at {checkpoint_path}, using random weights!")
#         else:
#             print(f"Loading checkpoint from: {checkpoint_path}")
#             state_dict = torch.load(checkpoint_path, map_location=self.device)
#             if 'model_state_dict' in state_dict:
#                 state_dict = state_dict['model_state_dict']
#             new_state_dict = OrderedDict()
#             for k, v in state_dict.items():
#                 name = k[7:] if k.startswith('module.') else k
#                 new_state_dict[name] = v
#             model.load_state_dict(new_state_dict, strict=False)
#         model.eval()
#         return model
#
#     def extract_source_bits(self, model):
#         print("\n=== Extracting Source Bits from Model ===")
#         try:
#             val_loader = get_dataloader(
#                 root_dir=self.cfg.VAL_DATASET_PATH,
#                 batch_size=4,
#                 shuffle=False,
#                 mode='val',
#                 num_workers=0  # CPU 模式建议 0
#             )
#         except Exception as e:
#             print(f"Error loading dataloader: {e}")
#             return np.random.randint(0, 2, size=100000)
#
#         all_bits = []
#         with torch.no_grad():
#             for i, img in enumerate(val_loader):
#                 if i >= 1: break
#                 img = img.to(self.device)
#                 out = model.forward_test_raq(img, self.cfg.RAQ_TARGET_LIST)
#                 indices = out['indices_src']
#                 flat_bits, _, _ = indices_to_bits(indices, self.cfg.NUM_EMBEDDINGS_LIST)
#                 all_bits.append(flat_bits)
#
#         if len(all_bits) > 0:
#             source_bits = np.concatenate(all_bits)
#             print(f"Extracted {len(source_bits)} bits from VQ-DeepSC model features.")
#             return source_bits
#         else:
#             return np.random.randint(0, 2, size=100000)
#
#     def run_ber_analysis(self):
#         model = self.load_model()
#         source_bits_pool = self.extract_source_bits(model)
#
#
#         snr_range = np.arange(-2, 11, 1)
#         ldpc_configs = [
#             (256, 1 / 2),
#             (512, 1 / 2),
#             (256, 2 / 3),
#             (512, 2 / 3)
#         ]
#
#         plt.figure(figsize=(12, 10))
#         colors = ['r', 'g', 'b', 'm']
#         markers = ['o', 's', '^', 'D']
#
#         # Task 3: 理论基准线
#         smooth_snr = np.linspace(-2, 10, 100)
#         plt.semilogy(smooth_snr, [theoretical_modulation_ber(s, 'bpsk') for s in smooth_snr],
#                      'k--', alpha=0.5, label='Theory BPSK (Uncoded)')
#         plt.semilogy(smooth_snr, [theoretical_modulation_ber(s, 'qpsk') for s in smooth_snr],
#                      'k-.', alpha=0.5, label='Theory QPSK (Uncoded)')
#
#         MODULATION = 'qpsk'
#
#         for idx, (L, R) in enumerate(ldpc_configs):
#             print(f"\nProcessing Config: L={L}, R={R:.2f}, Mod={MODULATION}")
#
#             # --- Task 2: 有限码长理论曲线 ---
#             fbl_ber_curve = []
#             for s in smooth_snr:
#                 s_tensor = torch.tensor(s, dtype=torch.float32)
#                 L_tensor = torch.tensor(L, dtype=torch.float32)
#
#                 # 【修正】：使用关键字参数 n_block 传递 L，并且 rc=R (动态)
#                 ber_val = self.fbl_calculator.compute_ber(s_tensor, n_block=L_tensor, rc=R)
#                 fbl_ber_curve.append(ber_val.item())
#
#             color = colors[idx]
#             plt.semilogy(smooth_snr, fbl_ber_curve, color=color, linestyle='-', alpha=0.6, linewidth=1.5,
#                          label=f'Theory FBL (L={L}, R={R:.2f})')
#
#             # --- Task 1: LDPC 仿真 ---
#             k = int(L * R)
#             code_dict = get_flexible_ldpc_code(n=L, k=k)
#             L_actual = code_dict['n']
#
#             ber_results = []
#             for snr in tqdm(snr_range, desc=f"Simulating SNR"):
#                 max_bits_to_test = 50000
#                 num_blocks = min(len(source_bits_pool) // k, max_bits_to_test // k)
#                 if num_blocks == 0: break
#
#                 bits_to_send = source_bits_pool[:num_blocks * k]
#                 bits_reshaped = bits_to_send.reshape(num_blocks, k).astype(np.float32)
#
#                 codewords_tf = code_dict['encoder'](bits_reshaped)
#                 codewords = torch.from_numpy(codewords_tf.numpy()).float().to(self.device)
#
#                 if MODULATION == 'bpsk':
#                     tx_symbols = bpsk_modulate(codewords)
#                 else:
#                     tx_symbols = qpsk_modulate(codewords)
#
#                 rx_symbols = awgn_channel(tx_symbols, snr)
#                 rx_llr = compute_llr(rx_symbols, snr, modulation=MODULATION)
#
#                 rx_llr_reshaped = rx_llr.reshape(num_blocks, L_actual).numpy()
#                 decoded_bits_tf = code_dict['decoder'](rx_llr_reshaped)
#                 decoded_bits = decoded_bits_tf.numpy().flatten()
#
#                 error_count = np.sum(bits_to_send != decoded_bits)
#                 ber = error_count / len(bits_to_send)
#                 ber_results.append(ber)
#
#             if len(ber_results) == len(snr_range):
#                 plt.semilogy(snr_range, ber_results, color=color, marker=markers[idx], linestyle='--',
#                              label=f'Model+LDPC Sim (L={L}, R={R:.2f})')
#
#         plt.title(f"BER vs SNR Analysis\n(VQ-DeepSC Model Bits + {MODULATION.upper()} + LDPC)", fontsize=14)
#         plt.xlabel("SNR (dB)", fontsize=12)
#         plt.ylabel("Bit Error Rate (BER)", fontsize=12)
#         plt.grid(True, which="both", linestyle='--', alpha=0.4)
#         plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
#         plt.ylim(1e-5, 1)
#         plt.tight_layout()
#
#         save_path = "model_ber_analysis.png"
#         plt.savefig(save_path, dpi=300)
#         print(f"\nAnalysis Complete. Result saved to {save_path}")
#         plt.show()
#
#
# if __name__ == "__main__":
#     analyzer = ModelBERAnalyzer()
#     analyzer.run_ber_analysis()