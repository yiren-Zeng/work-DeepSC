import torch
import torch.nn as nn
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from config import Config


class AWGNChannel(nn.Module):
    """连续特征 AWGN 信道：直接对特征张量加高斯白噪声"""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def apply_noise(self, features, snr_db):
        """
        对连续特征施加 AWGN 噪声
        Args:
            features: 编码器输出特征张量
            snr_db: 信噪比 (dB)
        Returns:
            加噪后的特征张量
        """
        snr_linear = 10 ** (snr_db / 10.0)
        power_signal = torch.mean(features ** 2)
        noise_power = power_signal / snr_linear
        noise = torch.sqrt(noise_power) * torch.randn_like(features)
        return features + noise


class DeepSC(nn.Module):
    """
    四层 U-Net（无量化模块、无噪声训练）
    消融实验版本：去除所有量化操作，训练时不引入信道噪声
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_downsample_blocks,
                 base_channels,
                 embedding_dim_list,
                 device
                 ):
        super(DeepSC, self).__init__()
        self.semantic_encoder = SemanticEncoder(in_channels, num_downsample_blocks, base_channels)
        self.semantic_decoder = SemanticDecoder(embedding_dim_list, out_channels, skip_dropout_p=Config.SKIP_DROPOUT_P)
        self.channel = AWGNChannel(device=device)
        self.device = device

    def forward_train(self, x):
        encoder_features = self.semantic_encoder(x)
        # 训练时无噪声，编码器特征直接传给解码器
        reconstructed_images = self.semantic_decoder(encoder_features)
        return {
            "reconstructed_images": reconstructed_images,
        }

    def forward_val(self, x):
        encoder_features = self.semantic_encoder(x)
        # 验证时也无噪声
        reconstructed_images = self.semantic_decoder(encoder_features)
        return {
            "reconstructed_images": reconstructed_images,
        }

    def forward_test(self, x):
        """测试时直接返回编码器特征（无量化索引）"""
        encoder_features = self.semantic_encoder(x)
        return {"features": encoder_features}

    def reconstruct_from_features(self, features):
        """从特征列表重建图像"""
        return self.semantic_decoder(features)
