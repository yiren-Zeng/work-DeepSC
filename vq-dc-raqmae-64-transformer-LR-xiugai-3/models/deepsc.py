import torch
import torch.nn as nn
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer
from .raq import RAQ
from .suit_ber import FiniteBlocklengthChannel
from config import Config
from utils.math_utils import sample_trg


class DeepSC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_downsample_blocks,
                 base_channels,
                 num_embeddings_list,
                 embedding_dim_list,
                 commitment_cost,
                 raq_min_trg,
                 raq_max_trg,
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
        self.raq_min_trg = raq_min_trg
        self.raq_max_trg = raq_max_trg

        if RAQ is None:
            raise ImportError(
                "未找到 raq_module.api.RAQ。请将提供的 raq_module/ 文件夹放到工程根目录，"
                "或检查 Python 导入路径。"
            )
        self.raqs = nn.ModuleList()

        for Ki, Di in zip(self.num_embeddings_list, self.embedding_dim_list):
            raq = RAQ(embedding_dim=Di, n_embed_src=Ki, n_embed_min_trg=self.raq_min_trg, n_embed_max_trg=self.raq_max_trg, device=self.device)
            self.raqs.append(raq)

        # === 【修改】初始化信道模拟器 ===
        self.channel = FiniteBlocklengthChannel(
            channel_coding_rate=Config.CHANNEL_CODING_RATE_TRAIN,  # 默认 0.5
            coded_block_length_bits=Config.BLOCK_LENGTH,  # 默认 256
            device=self.device
        )


    def forward_train_raq(self, x, trg_list=None):
        """
        启用 RAQ 的双支路前向，并在中间引入基于有限码长理论的信道噪声模拟。
        """
        # 1. 随机均匀采样 SNR (0 ~ 15 dB)
        snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)

        # ==========================================
        # 【核心修改】基于物理常识的动态阶数采样
        # ==========================================
        if snr_db < 4.0:
            # 极差信道: 强制只用 BPSK 或 QPSK，绝对禁止 16-QAM
            current_mod_bits = random.choice([1, 2])
        elif snr_db < 8.0:
            # 中等信道: BPSK, QPSK 都可以，偶尔让 16-QAM 承受一点中度噪声
            current_mod_bits = random.choice([1, 2, 4])
        else:
            # 优良信道: 偏向于高阶调制，榨干网络的高清重建能力
            current_mod_bits = random.choice([2, 4])

        # 2. 编码率 (固定为 0.5)
        current_rc = Config.CHANNEL_CODING_RATE_TRAIN

        encoder_features = self.semantic_encoder(x)

        # === 支路 1: 源码本  ===
        quantized_src_corrupted = []  # 用于存放经过信道干扰后的特征
        vq_losses_src = []
        indices_src = []

        for i, feat in enumerate(encoder_features):
            # 获取无噪声的量化特征和索引
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)

            vq_losses_src.append(vq_loss)
            indices_src.append(encoding_idx)

            # --- 信道模拟 ---
            # 直接传入 current_rc，compute_ber 内部会乘以 modulation_bits
            corrupted_idx, _ = self.channel.apply_channel_noise(
                encoding_idx,
                self.num_embeddings_list[i],
                snr_tensor,
                current_rc,
                mod_bits=current_mod_bits
            )

            # 使用损坏的索引查表，得到带噪声的量化向量
            quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)

            # 梯度直通 (Straight-Through Estimator for Channel):
            # 前向传播使用 noisy 特征 (模拟接收端看到的数据)，
            # 反向传播梯度传给 clean 特征 (让编码器学习对抗噪声)
            quantized_final = quantized_clean + (quantized_noisy - quantized_clean).detach()

            quantized_src_corrupted.append(quantized_final)

        # 解码器使用受损特征重建
        reconstructed_images_src = self.semantic_decoder(quantized_src_corrupted)
        # 修改后：确保传给 Loss 函数的是带有梯度且真实生效的投影码本
        codebooks_src_list = [vq.codebook.projected_weight() for vq in self.vector_quantizers]

        # === 支路 2: 目标码本 (RAQ ) ===
        quantized_raq_corrupted = []
        vq_losses_raq = []
        indices_raq = []
        codebooks_trg_list = []  # 创建一个空列表用来装 W_trg

        for i, feat in enumerate(encoder_features):

            # === 如果外部传入了 trg_list，就用外部的；否则自己随机 ===
            if trg_list is not None:
                K_tilde = trg_list[i]
            else:
                K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)

            # 获取当前层投影后的 VQ 权重
            current_vq_weight = self.vector_quantizers[i].codebook.projected_weight()
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde, current_vq_weight)
            codebooks_trg_list.append(W_trg)

            # 获取无噪声的量化特征和索引
            vq_loss_t, quantized_t_clean, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)

            vq_losses_raq.append(vq_loss_t)
            indices_raq.append(encoding_idx_t)

            # --- 信道模拟 ---
            # 注意：这里的码本大小是动态的 K_tilde
            corrupted_idx_t, _ = self.channel.apply_channel_noise(
                encoding_idx_t,
                K_tilde,
                snr_tensor,
                current_rc,
                mod_bits=current_mod_bits
            )

            quantized_noisy_t = self.vector_quantizers[i].get_quantized_features(corrupted_idx_t, W_trg)

            # 梯度直通
            quantized_final_t = quantized_t_clean + (quantized_noisy_t - quantized_t_clean).detach()
            quantized_raq_corrupted.append(quantized_final_t)

        reconstructed_images_raq = self.semantic_decoder(quantized_raq_corrupted)

        return {
            "reconstructed_images_src": reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "indices_src": indices_src,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "indices_raq": indices_raq,
            "current_snr": snr_db,  # 返回 SNR 供日志记录
            "W_src_list": codebooks_src_list,  # 把源码本列表传出去
            "W_trg_list": codebooks_trg_list  # 把目标码本列表传出去
        }

    def forward_val_raq(self, x):
        """
        启用 RAQ 的双支路前向，并在中间引入基于有限码长理论的信道噪声模拟。
        """
        # 1. 随机采样当前 Batch 的 SNR
        snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)

        # ==========================================
        # 【核心修改】基于物理常识的动态阶数采样
        # ==========================================
        if snr_db < 4.0:
            # 极差信道: 强制只用 BPSK 或 QPSK，绝对禁止 16-QAM
            current_mod_bits = random.choice([1, 2])
        elif snr_db < 8.0:
            # 中等信道: BPSK, QPSK 都可以，偶尔让 16-QAM 承受一点中度噪声
            current_mod_bits = random.choice([1, 2, 4])
        else:
            # 优良信道: 偏向于高阶调制，榨干网络的高清重建能力
            current_mod_bits = random.choice([2, 4])

        current_rc = Config.CHANNEL_CODING_RATE_VAL # 使用验证集固定的码率

        # 编码
        encoder_features = self.semantic_encoder(x)

        # === 支路 1: 源码本  ===
        quantized_src_corrupted = []  # 用于存放经过信道干扰后的特征
        vq_losses_src = []
        indices_src = []

        for i, feat in enumerate(encoder_features):
            # 获取无噪声的量化特征和索引
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses_src.append(vq_loss)
            indices_src.append(encoding_idx)

            corrupted_idx, _ = self.channel.apply_channel_noise(
                encoding_idx,
                self.num_embeddings_list[i],
                snr_tensor,
                current_rc,
                mod_bits=current_mod_bits  # 使用传入的调制阶数
            )

            # 使用损坏的索引查表，得到带噪声的量化向量
            quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)

            quantized_src_corrupted.append(quantized_noisy)

        # 解码器使用受损特征重建
        reconstructed_images_src = self.semantic_decoder(quantized_src_corrupted)
        # 修改后：确保传给 Loss 函数的是带有梯度且真实生效的投影码本
        codebooks_src_list = [vq.codebook.projected_weight() for vq in self.vector_quantizers]

        # === 支路 2: 目标码本 (RAQ) ===
        quantized_raq_corrupted = []
        vq_losses_raq = []
        indices_raq = []
        codebooks_trg_list = []

        for i, feat in enumerate(encoder_features):
            K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)

            # 获取当前层投影后的 VQ 权重
            current_vq_weight = self.vector_quantizers[i].codebook.projected_weight()

            # 把 current_vq_weight 传进去！
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde, current_vq_weight)
            codebooks_trg_list.append(W_trg)

            # 获取无噪声的量化特征和索引
            vq_loss_t, quantized_t_clean, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)
            vq_losses_raq.append(vq_loss_t)
            indices_raq.append(encoding_idx_t)

            corrupted_idx_t, _ = self.channel.apply_channel_noise(
                encoding_idx_t,
                K_tilde,
                snr_tensor,
                current_rc,
                mod_bits=current_mod_bits  # 使用传入的调制阶数
            )

            quantized_noisy_t = self.vector_quantizers[i].get_quantized_features(corrupted_idx_t, W_trg)
            quantized_raq_corrupted.append(quantized_noisy_t)

        reconstructed_images_raq = self.semantic_decoder(quantized_raq_corrupted)

        return {
            "reconstructed_images_src": reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "indices_src": indices_src,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "indices_raq": indices_raq,
            "current_snr": snr_db,  # 返回 SNR 供日志记录
            "W_src_list": codebooks_src_list,
            "W_trg_list": codebooks_trg_list
        }

    def forward_test_raq(self, x, trg : list ):
        """
        测试阶段引入信道噪声。
        """
        # 编码
        encoder_features = self.semantic_encoder(x)

        # 源码本支路
        indices_src = []

        for i, feat in enumerate(encoder_features):
            _, _, encoding_idx = self.vector_quantizers[i](feat)
            indices_src.append(encoding_idx)

        # 修改后：确保传给 Loss 函数的是带有梯度且真实生效的投影码本
        codebooks_src_list = [vq.codebook.projected_weight() for vq in self.vector_quantizers]

        # 目标码本支路
        indices_raq = []
        codebooks_trg_list = []

        for i, feat in enumerate(encoder_features):
            K_tilde = trg[i]
            # 获取当前层投影后的 VQ 权重
            current_vq_weight = self.vector_quantizers[i].codebook.projected_weight()
            # 把 current_vq_weight 传进去！
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde, current_vq_weight)
            codebooks_trg_list.append(W_trg)

            _, _, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)
            indices_raq.append(encoding_idx_t)

        return {
            "indices_src": indices_src,
            "indices_raq": indices_raq,
            "W_src_list": codebooks_src_list,
            "W_trg_list": codebooks_trg_list
        }


    def reconstruct_from_indices(self, all_encoding_indices, codebooks=None):
        """
        Args:
            all_encoding_indices: 各层的编码索引列表
            codebooks: 各层对应的码本权重列表
        Returns:
            重建的图像
        """
        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices): # all_encoding_indices是一个列表，每个元素都是一个张量，每一个张量表示该层的的编码索引
            # 取出对应的外部码本（如果有的话）
            raq_weight = codebooks[i] if codebooks is not None else None

            # 调用我们刚刚在 VectorQuantizer 里写好的统一接口
            quantized = self.vector_quantizers[i].get_quantized_features(encoding_indices, raq_weight=raq_weight)
            quantized_features.append(quantized)

        reconstructed_image = self.semantic_decoder(quantized_features)
        return reconstructed_image

    @torch.no_grad()
    def compute_codebook_utilization(self, dataloader, max_batches=None, device=None):
        """
        遍历数据集，统计 Source 和 RAQ 码本的利用率

        Args:
            dataloader: 数据加载器
            max_batches: 最多统计多少个 batch (None=全部)
            device: 计算设备

        Returns:
            dict: {
                'src': [Layer0_stats, Layer1_stats, ...],
                'raq': [Layer0_stats, Layer1_stats, ...]
            }
        """
        if device is None:
            device = self.device

        self.eval()

        num_layers = len(self.vector_quantizers)

        # 累积所有 batch 的索引
        all_indices_src = [[] for _ in range(num_layers)]
        all_indices_raq = [[] for _ in range(num_layers)]
        all_k_trg = [[] for _ in range(num_layers)]
        last_w_trg = [None] * num_layers  # 保存最后一次的RAQ码本权重，用于计算相似度

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

                # === RAQ 支路 ===
                K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)

                current_vq_weight = self.vector_quantizers[i].codebook.projected_weight()
                W_trg = self.raqs[i].generate_codebook_transformer(K_tilde, current_vq_weight)
                _, _, encoding_idx_raq = self.vector_quantizers[i].forward_raq(feat, W_trg)

                all_indices_raq[i].append(encoding_idx_raq.cpu())
                all_k_trg[i].append(K_tilde)
                last_w_trg[i] = W_trg.detach()

            batch_count += 1

        # 统计各层码本利用率
        results = {'src': [], 'raq': []}

        for i in range(num_layers):
            # --- Source 支路 ---
            src_all = torch.cat(all_indices_src[i], dim=0)
            src_stats = VectorQuantizer.compute_codebook_stats(src_all, self.num_embeddings_list[i])
            # 计算源码本码字间最小L2距离与坍缩统计
            src_codebook = self.vector_quantizers[i].codebook.projected_weight()
            src_l2_stats = VectorQuantizer.compute_min_l2_distance(src_codebook)
            src_stats['min_l2_dist'] = src_l2_stats['min_l2_dist']
            src_stats['collapse_count'] = src_l2_stats['collapse_count']
            src_stats['collapse_ratio'] = src_l2_stats['collapse_ratio']
            results['src'].append(src_stats)

            # --- RAQ 支路 ---
            raq_all = torch.cat(all_indices_raq[i], dim=0)
            max_k_trg = max(all_k_trg[i])
            raq_stats = VectorQuantizer.compute_codebook_stats(raq_all, max_k_trg)
            raq_stats['max_k_trg'] = max_k_trg
            # 计算最后一次生成的 RAQ 码本码字间最小L2距离与坍缩统计
            if last_w_trg[i] is not None:
                raq_l2_stats = VectorQuantizer.compute_min_l2_distance(last_w_trg[i])
                raq_stats['min_l2_dist'] = raq_l2_stats['min_l2_dist']
                raq_stats['collapse_count'] = raq_l2_stats['collapse_count']
                raq_stats['collapse_ratio'] = raq_l2_stats['collapse_ratio']
            else:
                raq_stats['min_l2_dist'] = 0.0
                raq_stats['collapse_count'] = 0
                raq_stats['collapse_ratio'] = 0.0
            results['raq'].append(raq_stats)

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

            print(f"\n  Layer {i} (Source K={k_src})")
            print("  " + "-" * 60)

            # Source 支路
            s = results['src'][i]
            print(f"  [SRC] 活跃率: {s['active_ratio']:.2%}  |  "
                  f"活跃码字: {s['active_count']}/{k_src}  |  "
                  f"死码字: {s['dead_count']}  |  "
                  f"困惑度: {s['perplexity']:.1f}/{k_src}  |  "
                  f"最小L2距离: {s['min_l2_dist']:.4f}  |  "
                  f"坍缩码字: {s['collapse_count']}/{k_src} ({s['collapse_ratio']:.2%})")

            # RAQ 支路
            r = results['raq'][i]
            max_k = r.get('max_k_trg', '?')
            print(f"  [RAQ] 活跃率: {r['active_ratio']:.2%}  |  "
                  f"活跃码字: {r['active_count']}/{max_k}  |  "
                  f"死码字: {r['dead_count']}  |  "
                  f"困惑度: {r['perplexity']:.1f}/{max_k}  |  "
                  f"最小L2距离: {r['min_l2_dist']:.4f}  |  "
                  f"坍缩码字: {r['collapse_count']}/{max_k} ({r['collapse_ratio']:.2%})")

        # 汇总
        print("\n" + "-" * 80)
        src_avg = sum(s['active_ratio'] for s in results['src']) / num_layers
        raq_avg = sum(r['active_ratio'] for r in results['raq']) / num_layers
        print(f"  [SRC] 平均活跃率: {src_avg:.2%}")
        print(f"  [RAQ] 平均活跃率: {raq_avg:.2%}")
        print("=" * 80 + "\n")
