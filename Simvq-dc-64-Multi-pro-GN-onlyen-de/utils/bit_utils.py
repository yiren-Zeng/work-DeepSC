import numpy as np
import torch

BITS_PER_VALUE = 16  # 使用 float16 (IEEE 754 半精度)，每个值 16 比特


def features_to_bits(features_list):
    """
    将连续特征直接转成比特流（利用 float16 的 IEEE 754 二进制表示）
    无任何量化操作，直接把每个 float16 值的 16 位内存布局展开为比特
    Args:
        features_list: 各层特征张量列表 [Tensor(B,C,H,W), ...]
    Returns:
        bit_stream: 拼接的比特流 (numpy uint8)
        metadata: 各层的形状信息 [(B,C,H,W), ...]，用于还原
    """
    bit_stream_parts = []
    metadata = []

    for feat in features_list:
        B, C, H, W = feat.shape
        metadata.append((B, C, H, W))

        # float32 → float16，再转 uint16 视图（保持内存布局不变）
        feat_np = feat.detach().cpu().to(torch.float16).numpy().flatten()
        uint16_view = feat_np.view(np.uint16)

        # 每个 uint16 值展开为 16 个比特
        shifts = np.arange(15, -1, -1, dtype=np.uint16)
        bits = ((uint16_view[:, None] >> shifts) & 1).flatten().astype(np.uint8)
        bit_stream_parts.append(bits)

    return np.concatenate(bit_stream_parts), metadata


def bits_to_features(bit_stream, metadata):
    """
    将比特流还原为连续特征（float16 IEEE 754 二进制表示还原）
    信道误码可能导致 NaN/Inf，此处做安全清理
    Args:
        bit_stream: 比特流 (numpy uint8)
        metadata: 各层形状信息 [(B,C,H,W), ...]
    Returns:
        features_list: 还原的特征张量列表
    """
    features_list = []
    offset = 0

    for (B, C, H, W) in metadata:
        num_values = B * C * H * W
        num_bits = num_values * BITS_PER_VALUE

        # 提取该层的比特
        layer_bits = bit_stream[offset:offset + num_bits]
        if len(layer_bits) < num_bits:
            layer_bits = np.pad(layer_bits, (0, num_bits - len(layer_bits)), 'constant')
        offset += num_bits

        # 比特 → uint16
        bits_reshaped = layer_bits.reshape(num_values, BITS_PER_VALUE)
        powers = 1 << np.arange(15, -1, -1, dtype=np.uint16)
        uint16_values = np.sum(bits_reshaped * powers, axis=1).astype(np.uint16)

        # uint16 → float16 视图 → float32
        float16_values = uint16_values.view(np.float16).astype(np.float32)

        # 【关键】清理信道误码导致的 NaN / Inf，替换为 0.0
        nan_mask = np.isnan(float16_values)
        inf_mask = np.isinf(float16_values)
        bad_mask = nan_mask | inf_mask
        if np.any(bad_mask):
            float16_values[bad_mask] = 0.0

        feat = torch.from_numpy(float16_values.reshape(B, C, H, W)).float()
        features_list.append(feat)

    return features_list
