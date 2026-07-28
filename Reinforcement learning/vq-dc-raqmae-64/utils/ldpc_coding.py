import numpy as np
import torch
import tensorflow as tf
from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder

# 设置设备，优先使用GPU
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# # 配置TensorFlow使用CPU，因为Sionna可能默认尝试使用GPU，而沙盒环境的GPU可能不可用或配置复杂
# # 确保TensorFlow在CPU上运行，避免与PyTorch的GPU使用冲突
# tf.config.set_visible_devices([], 'GPU')  # 禁用所有GPU


def get_ldpc_code(block_length):
    """
    生成LDPC编码器和解码器实例。
    Sionna的LDPC5G编码器和解码器基于5G NR标准，支持特定的码长和码率。
    这里需要根据实际需求选择合适的参数。

    参数:
        block_length (int): 码字长度 n。Sionna LDPC5G支持的码长是有限的，例如648, 1024, 1296等。
    返回:
        dict: 包含LDPC编码器和解码器实例的字典。
    """
    # Sionna的LDPC5GEncoder/Decoder的k和n是内部根据5G标准确定的，不能直接设置。
    # 它们只接受一个`k`参数，然后根据这个`k`和选择的基图来确定`n`。
    # 为了演示Sionna的集成，我们选择一个固定的5G NR LDPC码配置。
    # 考虑到用户提到比特数巨大，我们选择一个较大的k，例如 k=6480。
    # Sionna的LDPC5GEncoder会根据k=6480自动选择n=6480/0.5=12960 (如果码率为0.5)。
    # 实际的n会由Sionna内部根据5G NR标准确定。

    # 注意：Sionna的LDPC5GEncoder/Decoder是TensorFlow层，它们需要TensorFlow张量作为输入。

    # 返回Sionna的编码器和解码器实例
    k_sionna = block_length  # 5G NR LDPC码的信息比特长度，选择一个较大的值以适应大量比特
    n_sionna = k_sionna*2 # 5G NR LDPC码的码字长度
    encoder = LDPC5GEncoder(k=k_sionna,n=n_sionna)  # 创建LDPC5G编码器
    decoder = LDPC5GDecoder(encoder)  # 创建LDPC5G解码器，需要传入编码器实例

    return {"encoder": encoder, "decoder": decoder, "k": k_sionna}


def ldpc_encode(bits, code=None):
    """
    使用Sionna LDPC编码器对输入比特流进行编码。
    参数:
        bits (numpy.ndarray): 待编码的原始比特流，一维数组。
        code (dict): 包含Sionna LDPC编码器实例的字典。如果为None，则直接返回原始比特流。
    返回:
        numpy.ndarray: 编码后的比特流。
    """
    if code is None:
        return bits

    encoder = code["encoder"]
    k = code["k"]

    # 将numpy数组转换为TensorFlow张量，并确保数据类型为tf.float32
    # Sionna的LDPC5GEncoder期望输入形状为 (batch_size, k)
    # 因此，我们需要将一维比特流分块，并转换为tf.float32

    # 计算需要的编码块数
    num_blocks = (len(bits) + k - 1) // k
    padded_len = num_blocks * k
    padded_bits = np.pad(bits, (0, padded_len - len(bits)), 'constant', constant_values=0)

    # 将填充后的比特流重塑为 (num_blocks, k)，类比与[batch_size,k]
    bits_tf = tf.constant(padded_bits.reshape(num_blocks, k), dtype=tf.float32)

    # 执行编码
    encoded_bits_tf = encoder(bits_tf)

    # 将TensorFlow张量转换回numpy数组
    encoded_bits = encoded_bits_tf.numpy().flatten()

    return encoded_bits


def ldpc_decode(received_llr, code=None):
    """
    使用Sionna LDPC解码器对接收到的LLR进行解码。
    参数:
        received_llr (numpy.ndarray): 接收到的对数似然比(LLR)数组，一维数组。
        code (dict): 包含Sionna LDPC解码器实例的字典。如果为None，则直接进行硬判决。
        max_iter (int): 解码算法的最大迭代次数（Sionna解码器内部可能使用此参数）。
    返回:
        numpy.ndarray: 解码后的比特流。
    """
    if code is None:
        return (received_llr < 0).astype(int)

    decoder = code["decoder"]

    # Sionna的LDPC5GDecoder期望输入形状为 (batch_size, n)，数据类型为tf.float32
    # n是编码后的码字长度，可以通过encoder获取
    # 注意：Sionna的LDPC5GDecoder的输入LLR长度是固定的，由编码器决定。
    # 我们需要获取编码器对应的n。

    # 获取编码器实例，从解码器中获取其编码器属性
    encoder_for_n = decoder.encoder  # LDPC5GDecoder内部存储了其对应的LDPC5GEncoder
    n_sionna = encoder_for_n.n  # 获取编码器确定的码字长度n

    # 计算需要的解码块数
    num_blocks = (len(received_llr) + n_sionna - 1) // n_sionna
    padded_len = num_blocks * n_sionna
    padded_llr = np.pad(received_llr, (0, padded_len - len(received_llr)), 'constant', constant_values=0.0)

    # 将填充后的LLR重塑为 (num_blocks, n_sionna)
    llr_tf = tf.constant(padded_llr.reshape(num_blocks, n_sionna), dtype=tf.float32)

    # 执行解码
    # Sionna的LDPC5GDecoder返回的是解码后的信息比特 (batch_size, k)
    decoded_bits_tf = decoder(llr_tf)

    # 将TensorFlow张量转换回numpy数组
    decoded_bits = decoded_bits_tf.numpy().flatten()

    return decoded_bits


# --- 以下函数用于VQ-DeepSC模型中的比特流与索引转换，无需修改 --- #
def indices_to_bits(indices_list, num_embeddings_list):
    """
    将多尺度的编码索引转换成扁平化的比特流。
    参数:
        indices_list (list): 包含每个尺度编码索引的PyTorch张量列表。
        num_embeddings_list (list): 每个量化层中嵌入向量的数量列表。
    返回:
        tuple: (numpy.ndarray: 扁平化的比特流，一维数组, list: 原始空间维度列表, list: 原始嵌入数量列表)。
    """
    bit_stream = []
    original_spatial_dims = []

    # 计算每个索引所需的比特数
    bits_per_index_list = [int(np.log2(n)) for n in num_embeddings_list]

    for i, indices in enumerate(indices_list):
        original_spatial_dims.append(indices.shape[1:]) # 存储原始空间尺寸 (H, W)
        bits_per_index = bits_per_index_list[i]  # 获取当前层的索引所需的比特数
        # 遍历当前尺度下的所有索引
        for index in indices.flatten().cpu().numpy():  # 将PyTorch张量展平并转换为numpy数组
            # 将索引转换为二进制字符串，并用零填充到固定长度
            # 例如，如果索引是5，bits_per_index是4，则format(5, "04b")会得到"0101"
            binary_str = format(int(index), f"0{bits_per_index}b")
            # 将二进制字符串中的每个字符转换为整数（0或1），并添加到比特流列表中
            bit_stream.extend([int(b) for b in binary_str])

    return np.array(bit_stream, dtype=np.uint8), original_spatial_dims, num_embeddings_list

def bits_to_indices(bit_stream, original_spatial_dims, original_num_embeddings_list):
    """
    将扁平化的比特流转换回多尺度的编码索引。
    参数:
        bit_stream (numpy.ndarray): 扁平化的比特流，一维数组。
        original_spatial_dims (list): 原始空间维度列表，每个元素为(H, W)元组，对应每个尺度的特征图。
        original_num_embeddings_list (list): 原始的每个量化层中嵌入向量的数量列表。
    返回:
        list: 包含每个尺度编码索引的PyTorch张量列表。
    """
    indices_list = []
    # 计算每个索引所需的比特数，例如，如果嵌入数量为8，则需要log2(8)=3比特来表示索引
    bits_per_index_list = [int(np.log2(n)) for n in original_num_embeddings_list]
    current_pos = 0  # 当前比特流的读取位置

    for i, bits_per_index in enumerate(bits_per_index_list):
        h, w = original_spatial_dims[i][0], original_spatial_dims[i][1]  # 获取当前尺度的特征图高和宽
        num_indices_in_scale = h * w  # 当前尺度包含的索引总数
        num_bits_for_scale = num_indices_in_scale * bits_per_index  # 当前尺度所需的总比特数

        # 检查比特流长度是否足够，如果不足则进行填充并发出警告
        if current_pos + num_bits_for_scale > len(bit_stream):
            padding_needed = current_pos + num_bits_for_scale - len(bit_stream)
            scale_bits = np.pad(bit_stream[current_pos:], (0, padding_needed), 'constant', constant_values=0)
            print(
                f"Warning: Bit stream padded for scale {i}. Expected {num_bits_for_scale} bits, got {len(bit_stream) - current_pos}.")
        else:
            scale_bits = bit_stream[current_pos: current_pos + num_bits_for_scale]
        current_pos += num_bits_for_scale  # 更新读取位置

        # 将当前尺度的比特流重塑为 (索引数量, 每个索引的比特数)
        scale_bits_reshaped = scale_bits.reshape(num_indices_in_scale, bits_per_index)

        indices = np.zeros(num_indices_in_scale, dtype=int)  # 初始化索引数组
        for j in range(num_indices_in_scale):
            # 将二进制数组转换为整数索引
            indices[j] = int("".join(str(x) for x in scale_bits_reshaped[j]), 2)

        # 将numpy数组转换为PyTorch张量，并重塑回原始的(H, W)形状
        indices_list.append(torch.from_numpy(indices.reshape(h, w)).long())

    return indices_list
