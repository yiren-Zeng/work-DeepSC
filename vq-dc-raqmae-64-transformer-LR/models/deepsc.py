import torch
import torch.nn as nn
import math, random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer
from .raq import RAQ
# === 【新增】导入信道模块和配置 ===
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
        self.sync_raq_from_vq()

        # === 【修改】初始化信道模拟器 ===
        # 传入 Config 中定义的新参数
        self.channel = FiniteBlocklengthChannel(
            channel_coding_rate=Config.CHANNEL_CODING_RATE_TRAIN,  # 默认 0.5
            modulation_bits=Config.MODULATION_BITS,  # 默认 2 (QPSK)
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

        # 2. 编码率 (固定为 0.5)
        current_rc = Config.CHANNEL_CODING_RATE_TRAIN

        encoder_features = self.semantic_encoder(x)

        # === 支路 1: 源码本 (Fixed Codebook) ===
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
                current_rc
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

        # === 支路 2: 目标码本 (RAQ / Adaptive Codebook) ===
        quantized_raq_corrupted = []
        vq_losses_raq = []
        indices_raq = []

        for i, feat in enumerate(encoder_features):

            # === 如果外部传入了 trg_list，就用外部的；否则自己随机 ===
            if trg_list is not None:
                K_tilde = trg_list[i]

            else:
                K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)

            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde)  # [K̃, D_i]

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
                current_rc
            )

            # 查 RAQ 生成的动态码本
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
            "current_snr": snr_db  # 返回 SNR 供日志记录
        }

    def forward_val_raq(self, x):
        """
        启用 RAQ 的双支路前向，并在中间引入基于有限码长理论的信道噪声模拟。
        """
        # 1. 随机采样当前 Batch 的 SNR
        snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)


        current_rc = Config.CHANNEL_CODING_RATE_VAL # 使用验证集固定的码率

        # 编码
        encoder_features = self.semantic_encoder(x)

        # === 支路 1: 源码本 (Fixed Codebook) ===
        quantized_src_corrupted = []  # 用于存放经过信道干扰后的特征
        vq_losses_src = []
        indices_src = []

        for i, feat in enumerate(encoder_features):
            # 获取无噪声的量化特征和索引
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)

            vq_losses_src.append(vq_loss)
            indices_src.append(encoding_idx)

            # --- 信道模拟 (Bit Flip) ---
            # [cite_start]基于公式 (14) 计算 BER 并翻转索引 [cite: 721]
            corrupted_idx, _ = self.channel.apply_channel_noise(
                encoding_idx,
                self.num_embeddings_list[i],
                snr_tensor,
                current_rc
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

        # === 支路 2: 目标码本 (RAQ / Adaptive Codebook) ===
        quantized_raq_corrupted = []
        vq_losses_raq = []
        indices_raq = []

        for i, feat in enumerate(encoder_features):
            K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde)  # [K̃, D_i]

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
                current_rc
            )

            # 查 RAQ 生成的动态码本
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
            "current_snr": snr_db  # 返回 SNR 供日志记录
        }

    def forward_test_raq(self, x, trg : list ):
        """
        测试阶段引入信道噪声。
        """
        # 测试时一般会指定 SNR 循环测试，这里先随机采样一个，或者你可以改为由外部传入
        snr_db = random.uniform(Config.SNR_RANGE_DB[0], Config.SNR_RANGE_DB[1])
        snr_tensor = torch.tensor(snr_db, device=self.device)

        # 编码
        encoder_features = self.semantic_encoder(x)

        # 源码本支路
        quantized_src_corrupted = []
        vq_losses_src = []
        indices_src = []
        for i, feat in enumerate(encoder_features):
            vq_loss, quantized, encoding_idx = self.vector_quantizers[i](feat)

            vq_losses_src.append(vq_loss)
            indices_src.append(encoding_idx)

            # 信道噪声
            corrupted_idx, _ = self.channel.apply_channel_noise(encoding_idx, self.num_embeddings_list[i], snr_tensor)
            quantized_noisy = self.vector_quantizers[i].get_quantized_features(corrupted_idx)
            # 测试时不需要梯度回传，直接用 noisy 特征即可，但保持结构一致也无妨
            quantized_src_corrupted.append(quantized_noisy)

        reconstructed_images_src = self.semantic_decoder(quantized_src_corrupted)

        # 目标码本支路
        quantized_raq_corrupted = []
        vq_losses_raq = []
        indices_raq = []
        codebooks = []

        for i, feat in enumerate(encoder_features):
            K_tilde = trg[i]
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde)
            codebooks.append(W_trg)

            vq_loss_t, quantized_t, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)

            vq_losses_raq.append(vq_loss_t)
            indices_raq.append(encoding_idx_t)

            # 信道噪声
            corrupted_idx_t, _ = self.channel.apply_channel_noise(encoding_idx_t, K_tilde, snr_tensor)
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
            "codebooks": codebooks,
            "current_snr": snr_db
        }

    @torch.no_grad()
    def forward_test_mask_layers(self, x, trg_list, active_indices=None):
        """
        测试特定层组合的效果。
        active_indices: 一个列表，包含需要“激活”的层索引（0到3）。
                        不在列表中的层将被置为全0（Mask掉）。
                        例如：[0] 表示只用第0层；[0, 1] 表示用第0和第1层。
                        注意：列表索引对应 embedding_dim_list 的顺序。
                              通常 0 是浅层(高分辨率), 3 是深层(低分辨率)。
                              具体取决于您 SemanticEncoder 的输出顺序。
        """
        # 1. 编码
        encoder_features = self.semantic_encoder(x)

        # 2. 量化 (获取所有层的量化特征，暂时不管 mask)
        quantized_raq = []

        # 先全部计算出来
        for i, feat in enumerate(encoder_features):
            target_k = trg_list[i]
            if target_k is None:
                # 使用 Source 码本
                W_current = self.vector_quantizers[i].embeddings.weight
            else:
                # 使用 RAQ 码本
                W_current = self.raqs[i].generate_codebook_transformer(target_k)

            # 量化
            _, quantized_t, _ = self.vector_quantizers[i].forward_raq(feat, W_current)
            quantized_raq.append(quantized_t)

        # 3. 应用 Mask (关键步骤)
        # 如果 active_indices 为 None，则默认全开
        if active_indices is not None:
            masked_quantized = []
            for i, q_feat in enumerate(quantized_raq):
                if i in active_indices:
                    # 激活层：保留原特征
                    masked_quantized.append(q_feat)
                else:
                    # 非激活层：置为全 0
                    masked_quantized.append(torch.zeros_like(q_feat))
            quantized_raq = masked_quantized

        # 4. 解码
        reconstructed_images_raq = self.semantic_decoder(quantized_raq)

        return reconstructed_images_raq


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
    def sync_raq_from_vq(self):
        for rq, vq in zip(self.raqs, self.vector_quantizers):
            rq.set_src_weight(vq.embeddings.weight.data)