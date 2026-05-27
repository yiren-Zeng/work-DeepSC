import tensorflow as tf
import numpy as np
import torch

# =========================================================================
# 【修复方案】强制 TensorFlow 仅使用 CPU，避免与 PyTorch 抢占 GPU 触发 ECC 错误
# =========================================================================
try:
    # 隐藏所有物理 GPU，强制 TF 在 CPU 上初始化上下文
    tf.config.set_visible_devices([], 'GPU')
    print("[LDPC Info] TensorFlow GPU 已经被成功屏蔽，LDPC 模块将运行在 CPU 上。")
except RuntimeError as e:
    # 必须在 TF 初始化其他内容前调用
    print(f"[LDPC Warning] TensorFlow 可见设备配置失败: {e}")

# =========================================================================
# 在配置好 TF 后，再导入 Sionna
# =========================================================================
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder


def get_ldpc_code(block_length, rate=0.5):
    """
    生成LDPC编码器和解码器实例，支持动态修改码率。
    """
    # 5G NR LDPC 码长设置
    k_sionna = block_length
    n_sionna = int(k_sionna / rate)  # 根据输入的码率，动态计算总码长 N

    # Sionna 运行在 CPU 上 (因为上面已经禁用了 TF 的 GPU)
    encoder = LDPC5GEncoder(k=k_sionna, n=n_sionna)
    decoder = LDPC5GDecoder(encoder)

    # 返回时顺便把 n 和 rate 也带上，方便调试
    return {"encoder": encoder, "decoder": decoder, "k": k_sionna, "n": n_sionna, "rate": rate}


def ldpc_encode(bits, code=None):
    """
    使用Sionna LDPC编码器对输入比特流进行编码。
    """
    if code is None:
        return bits

    encoder = code["encoder"]
    k = code["k"]

    # 1. 填充 (Padding)
    num_blocks = (len(bits) + k - 1) // k
    padded_len = num_blocks * k
    padded_bits = np.pad(bits, (0, padded_len - len(bits)), 'constant', constant_values=0)

    # 2. 转为 TF Tensor (自动在 CPU)
    # 即使传入 tensor，TF 也会因为 visible_devices=[] 而将其放在 CPU
    bits_tf = tf.constant(padded_bits.reshape(num_blocks, k), dtype=tf.float32)

    # 3. 编码
    encoded_bits_tf = encoder(bits_tf)

    # 4. 转回 Numpy
    encoded_bits = encoded_bits_tf.numpy().flatten()

    return encoded_bits


def ldpc_decode(received_llr, code=None):
    """
    使用Sionna LDPC解码器对接收到的LLR进行解码。
    """
    if code is None:
        return (received_llr < 0).astype(int)

    decoder = code["decoder"]
    encoder_for_n = decoder.encoder
    n_sionna = encoder_for_n.n

    # 1. 填充 (Padding)
    num_blocks = (len(received_llr) + n_sionna - 1) // n_sionna
    padded_len = num_blocks * n_sionna
    padded_llr = np.pad(received_llr, (0, padded_len - len(received_llr)), 'constant', constant_values=0.0)

    # 2. 转为 TF Tensor (CPU)
    llr_tf = tf.constant(padded_llr.reshape(num_blocks, n_sionna), dtype=tf.float32)

    # 3. 解码
    decoded_bits_tf = decoder(llr_tf)

    # 4. 转回 Numpy 并强制转换为整数 【本次关键修改】
    # Sionna 默认返回 float32 (0.0, 1.0)，必须转为 int (0, 1) 才能被后续处理
    decoded_bits = decoded_bits_tf.numpy().flatten().astype(int)

    return decoded_bits

