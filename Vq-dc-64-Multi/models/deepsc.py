import torch
import torch.nn as nn
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer
from .channel import FiniteBlocklengthChannel
from config import Config


class DeepSC(nn.Module):
    """
    四层 U-Net + 纯 SRC 支路 + 传统 VQ
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_downsample_blocks,
                 base_channels,
                 num_embeddings_list,
                 embedding_dim_list,
                 commitment_cost,
                 device
                 ):
        super(DeepSC, self).__init__()
        self.semantic_encoder = SemanticEncoder(in_channels, num_downsample_blocks, base_channels)
        self.semantic_decoder = SemanticDecoder(embedding_dim_list, out_channels)
        self.vector_quantizers = nn.ModuleList()
        for i in range(num_downsample_blocks):
            self.vector_quantizers.append(VectorQuantizer(
                num_embeddings_list[i],
                embedding_dim_list[i],
                commitment_cost
            ))

        self.device = device
        self.num_embeddings_list = num_embeddings_list
        self.embedding_dim_list = embedding_dim_list

        self.channel = FiniteBlocklengthChannel(
            channel_coding_rate=Config.CHANNEL_CODING_RATE_TRAIN,
            coded_block_length_bits=Config.BLOCK_LENGTH,
            device=self.device
        )

    def _sample_mod_bits(self, snr_db):
        if snr_db < 4.0:
            return random.choice([1, 2])
        elif snr_db < 8.0:
            return random.choice([1, 2, 4])
        else:
            return random.choice([2, 4])

    def forward_train(self, x):
        """训练前向：SRC 单支路 + 信道噪声 + STE"""
        snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = Config.CHANNEL_CODING_RATE_TRAIN

        encoder_features = self.semantic_encoder(x)

        quantized_corrupted = []
        vq_losses = []

        for i, feat in enumerate(encoder_features):
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)

            corrupted_idx, _ = self.channel.apply_channel_noise(
                encoding_idx,
                self.num_embeddings_list[i],
                snr_tensor,
                current_rc,
                mod_bits=current_mod_bits
            )

            quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)

            # STE
            quantized_final = quantized_clean + (quantized_noisy - quantized_clean).detach()
            quantized_corrupted.append(quantized_final)

        reconstructed_images = self.semantic_decoder(quantized_corrupted)

        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "current_snr": snr_db,
        }

    def forward_val(self, x):
        """验证前向"""
        snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)
        current_mod_bits = self._sample_mod_bits(snr_db)
        current_rc = Config.CHANNEL_CODING_RATE_VAL

        encoder_features = self.semantic_encoder(x)

        quantized_corrupted = []
        vq_losses = []

        for i, feat in enumerate(encoder_features):
            vq_loss, _, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses.append(vq_loss)

            corrupted_idx, _ = self.channel.apply_channel_noise(
                encoding_idx,
                self.num_embeddings_list[i],
                snr_tensor,
                current_rc,
                mod_bits=current_mod_bits
            )

            quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)
            quantized_corrupted.append(quantized_noisy)

        reconstructed_images = self.semantic_decoder(quantized_corrupted)

        return {
            "reconstructed_images": reconstructed_images,
            "vq_losses": vq_losses,
            "current_snr": snr_db,
        }

    def forward_test(self, x):
        """测试前向：只编码输出索引"""
        encoder_features = self.semantic_encoder(x)

        indices_list = []
        for i, feat in enumerate(encoder_features):
            _, _, encoding_idx = self.vector_quantizers[i](feat)
            indices_list.append(encoding_idx)

        return {
            "indices": indices_list,
        }

    def reconstruct_from_indices(self, all_encoding_indices):
        """从索引重建图像"""
        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices):
            quantized = self.vector_quantizers[i].get_quantized_features(encoding_indices)
            quantized_features.append(quantized)

        reconstructed_image = self.semantic_decoder(quantized_features)
        return reconstructed_image

    @torch.no_grad()
    def compute_codebook_utilization(self, dataloader, max_batches=None, device=None):
        """
        遍历数据集，统计 Source 码本的利用率

        Args:
            dataloader: 数据加载器
            max_batches: 最多统计多少个 batch (None=全部)
            device: 计算设备

        Returns:
            dict: {
                'src': [Layer0_stats, Layer1_stats, ...]
            }
        """
        if device is None:
            device = self.device

        self.eval()

        num_layers = len(self.vector_quantizers)

        # 累积所有 batch 的索引
        all_indices_src = [[] for _ in range(num_layers)]

        batch_count = 0
        for images in dataloader:
            if max_batches is not None and batch_count >= max_batches:
                break

            images = images.to(device)
            encoder_features = self.semantic_encoder(images)

            for i, feat in enumerate(encoder_features):
                # === Source 支路 ===
                _, _, encoding_idx_src = self.vector_quantizers[i](feat)
                all_indices_src[i].append(encoding_idx_src.cpu())

            batch_count += 1

        # 统计各层码本利用率
        results = {'src': []}

        for i in range(num_layers):
            # --- Source 支路 ---
            src_all = torch.cat(all_indices_src[i], dim=0)
            src_stats = VectorQuantizer.compute_codebook_stats(src_all, self.num_embeddings_list[i])
            # 计算源码本码字间最小L2距离与坍缩统计
            src_codebook = self.vector_quantizers[i].codebook.weight if hasattr(self.vector_quantizers[i], 'codebook') else self.vector_quantizers[i].embedding.weight
            src_l2_stats = VectorQuantizer.compute_min_l2_distance(src_codebook)
            src_stats['min_l2_dist'] = src_l2_stats['min_l2_dist']
            src_stats['collapse_count'] = src_l2_stats['collapse_count']
            src_stats['collapse_ratio'] = src_l2_stats['collapse_ratio']
            results['src'].append(src_stats)

        return results

    @staticmethod
    def print_codebook_utilization(results, num_embeddings_list=None):
        """
        格式化打印码本利用率统计结果

        Args:
            results: compute_codebook_utilization 返回的字典
            num_embeddings_list: 各层码本大小列表 (可选，用于显示)
        """
        num_layers = len(results['src'])

        print("\n" + "=" * 80)
        print("  码本利用率统计报告 (Codebook Utilization Report)")
        print("=" * 80)

        for i in range(num_layers):
            k_src = num_embeddings_list[i] if num_embeddings_list else "?"

            print(f"\n  Layer {i} (K={k_src})")
            print("  " + "-" * 60)

            # Source 支路
            s = results['src'][i]
            print(f"  [VQ] 活跃率: {s['active_ratio']:.2%}  |  "
                  f"活跃码字: {s['active_count']}/{k_src}  |  "
                  f"死码字: {s['dead_count']}  |  "
                  f"困惑度: {s['perplexity']:.1f}/{k_src}  |  "
                  f"最小L2距离: {s['min_l2_dist']:.4f}  |  "
                  f"坍缩码字: {s['collapse_count']}/{k_src} ({s['collapse_ratio']:.2%})")

        # 汇总
        print("\n" + "-" * 80)
        src_avg = sum(s['active_ratio'] for s in results['src']) / num_layers
        print(f"  [VQ] 平均活跃率: {src_avg:.2%}")
        print("=" * 80 + "\n")
