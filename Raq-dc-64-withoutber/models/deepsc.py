import torch
import torch.nn as nn
import random
from .semantic_encoder import SemanticEncoder
from .semantic_decoder import SemanticDecoder
from .vector_quantizer import VectorQuantizer
from .raq import RAQ
from config import Config
from utils.math_utils import sample_trg


class DeepSC(nn.Module):
    """
    单层 U-Net + 双支路(SRC+RAQ) + SimVQ，loss 无 repulsion_loss
    (Without BER: 无信道噪声训练版本)
    """
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

        self.raqs = nn.ModuleList()
        for Ki, Di in zip(self.num_embeddings_list, self.embedding_dim_list):
            raq = RAQ(embedding_dim=Di, n_embed_src=Ki,
                      n_embed_min_trg=self.raq_min_trg, n_embed_max_trg=self.raq_max_trg,
                      device=self.device)
            self.raqs.append(raq)

    def forward_train_raq(self, x, trg_list=None):
        """训练前向：无信道噪声，SRC 和 RAQ 支路均直接量化重建"""
        encoder_features = self.semantic_encoder(x)

        # === 支路 1: SRC ===
        quantized_src_list = []
        vq_losses_src = []
        indices_src = []

        for i, feat in enumerate(encoder_features):
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses_src.append(vq_loss)
            indices_src.append(encoding_idx)
            quantized_src_list.append(quantized_clean)

        reconstructed_images_src = self.semantic_decoder(quantized_src_list)

        # === 支路 2: RAQ ===
        quantized_raq_list = []
        vq_losses_raq = []
        indices_raq = []
        codebooks_trg_list = []

        for i, feat in enumerate(encoder_features):
            if trg_list is not None:
                K_tilde = trg_list[i]
            else:
                K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)

            current_vq_weight = self.vector_quantizers[i].codebook.projected_weight()
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde, current_vq_weight)
            codebooks_trg_list.append(W_trg)

            vq_loss_t, quantized_t_clean, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)
            vq_losses_raq.append(vq_loss_t)
            indices_raq.append(encoding_idx_t)
            quantized_raq_list.append(quantized_t_clean)

        reconstructed_images_raq = self.semantic_decoder(quantized_raq_list)

        return {
            "reconstructed_images_src": reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "indices_src": indices_src,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "indices_raq": indices_raq,
            "W_trg_list": codebooks_trg_list
        }

    def forward_val_raq(self, x):
        """验证前向：无信道噪声"""
        encoder_features = self.semantic_encoder(x)

        # SRC
        quantized_src_list = []
        vq_losses_src = []
        indices_src = []
        for i, feat in enumerate(encoder_features):
            vq_loss, quantized_clean, encoding_idx = self.vector_quantizers[i](feat)
            vq_losses_src.append(vq_loss)
            indices_src.append(encoding_idx)
            quantized_src_list.append(quantized_clean)
        reconstructed_images_src = self.semantic_decoder(quantized_src_list)

        # RAQ
        quantized_raq_list = []
        vq_losses_raq = []
        indices_raq = []
        codebooks_trg_list = []
        for i, feat in enumerate(encoder_features):
            K_tilde = sample_trg(self.raq_min_trg, self.raq_max_trg)
            current_vq_weight = self.vector_quantizers[i].codebook.projected_weight()
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde, current_vq_weight)
            codebooks_trg_list.append(W_trg)

            vq_loss_t, quantized_t_clean, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)
            vq_losses_raq.append(vq_loss_t)
            indices_raq.append(encoding_idx_t)
            quantized_raq_list.append(quantized_t_clean)
        reconstructed_images_raq = self.semantic_decoder(quantized_raq_list)

        return {
            "reconstructed_images_src": reconstructed_images_src,
            "vq_losses_src": vq_losses_src,
            "indices_src": indices_src,
            "reconstructed_images_raq": reconstructed_images_raq,
            "vq_losses_raq": vq_losses_raq,
            "indices_raq": indices_raq,
            "W_trg_list": codebooks_trg_list
        }

    def forward_test_raq(self, x, trg: list):
        encoder_features = self.semantic_encoder(x)
        indices_src = []
        for i, feat in enumerate(encoder_features):
            _, _, encoding_idx = self.vector_quantizers[i](feat)
            indices_src.append(encoding_idx)

        indices_raq = []
        codebooks_trg_list = []
        for i, feat in enumerate(encoder_features):
            K_tilde = trg[i]
            current_vq_weight = self.vector_quantizers[i].codebook.projected_weight()
            W_trg = self.raqs[i].generate_codebook_transformer(K_tilde, current_vq_weight)
            codebooks_trg_list.append(W_trg)
            _, _, encoding_idx_t = self.vector_quantizers[i].forward_raq(feat, W_trg)
            indices_raq.append(encoding_idx_t)

        return {
            "indices_src": indices_src,
            "indices_raq": indices_raq,
            "W_trg_list": codebooks_trg_list
        }

    def reconstruct_from_indices(self, all_encoding_indices, codebooks=None):
        quantized_features = []
        for i, encoding_indices in enumerate(all_encoding_indices):
            raq_weight = codebooks[i] if codebooks is not None else None
            quantized = self.vector_quantizers[i].get_quantized_features(encoding_indices, raq_weight=raq_weight)
            quantized_features.append(quantized)
        reconstructed_image = self.semantic_decoder(quantized_features)
        return reconstructed_image

    @torch.no_grad()
    def compute_codebook_utilization(self, dataloader, max_batches=None, device=None):
        """
        遍历数据集，统计 Source 和 RAQ 码本的利用率
        """
        if device is None:
            device = self.device

        self.eval()

        num_layers = len(self.vector_quantizers)

        all_indices_src = [[] for _ in range(num_layers)]
        all_indices_raq = [[] for _ in range(num_layers)]
        all_k_trg = [[] for _ in range(num_layers)]
        last_w_trg = [None] * num_layers

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

        results = {'src': [], 'raq': []}

        for i in range(num_layers):
            # --- Source 支路 ---
            src_all = torch.cat(all_indices_src[i], dim=0)
            src_stats = VectorQuantizer.compute_codebook_stats(src_all, self.num_embeddings_list[i])
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