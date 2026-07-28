import torch
import torch.nn as nn
import math, random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer
from .raq import RAQ # 是不是有毒？加个.就可以导入get_quantized_features这个新加的方法了
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

        # ========== RAQ:每层一个（维度各自独立） ==========
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

        # 先用 CPU/默认设备初始化；首次真正使用时再迁移到输入张量同设备（_ensure_raq_device）
        for Ki, Di in zip(self.num_embeddings_list, self.embedding_dim_list):
            raq = RAQ(embedding_dim=Di, n_embed_src=Ki,n_embed_min_trg=self.raq_min_trg, n_embed_max_trg=self.raq_max_trg,
                      device=self.device)
            self.raqs.append(raq)
        self.sync_raq_from_vq()  # 首次同步源码本


    def forward_train_raq(self, x):
        """
        启用 RAQ 的双支路前向：
          - 源码本支路（与你当前训练一致）
          - 目标码本支路（由 RAQ 生成的 K̃ 码本）
        返回一个 dict，包含两路重建与损失项材料。
        """
        # 编码
        encoder_features = self.semantic_encoder(x)

        # 源码本支路
        quantized_src = []
        vq_losses_src = []
        indices_src = []
        for i, feat in enumerate(encoder_features):
            vq_loss, quantized, encoding_idx = self.vector_quantizers[i](feat)

            vq_losses_src.append(vq_loss)
            quantized_src.append(quantized) # 这里的向量量化后的特征
            indices_src.append(encoding_idx)

        reconstructed_images_src = self.semantic_decoder(quantized_src)


        # 目标码本支路（Seq2Seq 生成 K̃_i × D_i）
        quantized_raq = []
        vq_losses_raq = []
        indices_raq = []
        for i, feat in enumerate(encoder_features):
            # 假设 self.raq 暴露了下/上界
            K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde) # [K̃, D_i]
            # 用你自带的量化器计算 VQ 损失，与主干口径一致
            vq_loss_t, quantized_t, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)

            vq_losses_raq.append(vq_loss_t)
            quantized_raq.append(quantized_t)
            indices_raq.append(encoding_idx_t)

        reconstructed_images_raq = self.semantic_decoder(quantized_raq)

        return {
            "reconstructed_images_src":reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "indices_src": indices_src,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "indices_raq": indices_raq
        }

    def forward_val_raq(self, x):
        """
        启用 RAQ 的双支路前向：
          - 源码本支路（与你当前训练一致）
          - 目标码本支路（由 RAQ 生成的 K̃ 码本）
        返回一个 dict，包含两路重建与损失项材料。
        """
        # 编码
        encoder_features = self.semantic_encoder(x)

        # 源码本支路
        quantized_src = []
        vq_losses_src = []
        indices_src = []
        for i, feat in enumerate(encoder_features):
            vq_loss, quantized, encoding_idx = self.vector_quantizers[i](feat)

            vq_losses_src.append(vq_loss)
            quantized_src.append(quantized) # 这里的向量量化后的特征
            indices_src.append(encoding_idx)

        reconstructed_images_src = self.semantic_decoder(quantized_src)


        # 目标码本支路（Seq2Seq 生成 K̃_i × D_i）
        quantized_raq = []
        vq_losses_raq = []
        indices_raq = []
        for i, feat in enumerate(encoder_features):
            # 假设 self.raq 暴露了下/上界
            K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde) # [K̃, D_i]
            # 用你自带的量化器计算 VQ 损失，与主干口径一致
            vq_loss_t, quantized_t, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)

            vq_losses_raq.append(vq_loss_t)
            quantized_raq.append(quantized_t)
            indices_raq.append(encoding_idx_t)

        reconstructed_images_raq = self.semantic_decoder(quantized_raq)

        return {
            "reconstructed_images_src":reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "indices_src": indices_src,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "indices_raq": indices_raq
        }


    def forward_test_raq(self, x, trg:list):
        """
            启用 RAQ 的双支路前向：
                - 源码本支路（与你当前训练一致）
                - 目标码本支路（由 RAQ 生成的 K̃ 码本）
            返回一个 dict，包含两路重建与损失项材料。
        """
        # 编码
        encoder_features = self.semantic_encoder(x)

        # 源码本支路
        quantized_src = []
        vq_losses_src = []
        indices_src = []

        for i, feat in enumerate(encoder_features):
            vq_loss, quantized, encoding_idx = self.vector_quantizers[i](feat)

            vq_losses_src.append(vq_loss)
            quantized_src.append(quantized)  # 这里的向量量化后的特征
            indices_src.append(encoding_idx)

        reconstructed_images_src = self.semantic_decoder(quantized_src)

        # 目标码本支路（Seq2Seq 生成 K̃_i × D_i）
        quantized_raq = []
        vq_losses_raq = []
        indices_raq = []
        codebooks = [] # 保存生成的码本

        for i, feat in enumerate(encoder_features):

            K_tilde = trg[i]
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde)  # [K̃, D_i]
            codebooks.append(W_trg) # 保存生成的码本

            # 用你自带的量化器计算 VQ 损失，与主干口径一致
            vq_loss_t, quantized_t, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)

            vq_losses_raq.append(vq_loss_t)
            quantized_raq.append(quantized_t)
            indices_raq.append(encoding_idx_t)

        reconstructed_images_raq = self.semantic_decoder(quantized_raq)

        return {
            "reconstructed_images_src": reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "indices_src": indices_src,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "indices_raq": indices_raq,
            "codebooks": codebooks
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
        """
        把当前各层源码本复制到 RAQ；建议训练中每 N 步调用一次（例如每 100 步）。
        """
        # if not getattr(self, "raq_enable", False):
        #     return
        for rq, vq in zip(self.raqs, self.vector_quantizers):
            rq.set_src_weight(vq.embeddings.weight.data)


