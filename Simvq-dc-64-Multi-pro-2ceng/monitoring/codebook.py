import torch

from models.vector_quantizer import VectorQuantizer


@torch.no_grad()
def compute_codebook_utilization(model, dataloader, max_batches=None, device=None):
    if device is None:
        device = model.device

    model.eval()
    num_layers = len(model.vector_quantizers)
    all_indices_src = [[] for _ in range(num_layers)]

    for batch_count, images in enumerate(dataloader):
        if max_batches is not None and batch_count >= max_batches:
            break

        images = images.to(device)
        encoder_features = model.semantic_encoder(images)
        for i, feat in enumerate(encoder_features):
            _, _, encoding_idx_src = model.vector_quantizers[i](feat)
            all_indices_src[i].append(encoding_idx_src.cpu())

    results = {"src": []}
    for i in range(num_layers):
        src_all = torch.cat(all_indices_src[i], dim=0)
        src_stats = VectorQuantizer.compute_codebook_stats(src_all, model.num_embeddings_list[i])
        src_codebook = model.vector_quantizers[i].codebook.projected_weight()
        src_l2_stats = VectorQuantizer.compute_min_l2_distance(src_codebook)
        src_stats["min_l2_dist"] = src_l2_stats["min_l2_dist"]
        src_stats["collapse_count"] = src_l2_stats["collapse_count"]
        src_stats["collapse_ratio"] = src_l2_stats["collapse_ratio"]
        results["src"].append(src_stats)

    return results


def print_codebook_utilization(results, num_embeddings_list=None):
    num_layers = len(results["src"])

    print("\n" + "=" * 80)
    print("  码本利用率统计报告 (Codebook Utilization Report)")
    print("=" * 80)

    for i in range(num_layers):
        k_src = num_embeddings_list[i] if num_embeddings_list else "?"
        s = results["src"][i]

        print(f"\n  Layer {i} (K={k_src})")
        print("  " + "-" * 60)
        print(f"  [SimVQ] 活跃率: {s['active_ratio']:.2%}  |  "
              f"活跃码字: {s['active_count']}/{k_src}  |  "
              f"死码字: {s['dead_count']}  |  "
              f"困惑度: {s['perplexity']:.1f}/{k_src}  |  "
              f"最小L2距离: {s['min_l2_dist']:.4f}  |  "
              f"坍缩码字: {s['collapse_count']}/{k_src} ({s['collapse_ratio']:.2%})")

    src_avg = sum(s["active_ratio"] for s in results["src"]) / num_layers
    print("\n" + "-" * 80)
    print(f"  [SimVQ] 平均活跃率: {src_avg:.2%}")
    print("=" * 80 + "\n")


def write_codebook_tensorboard(writer, results, epoch):
    for i, stats in enumerate(results["src"]):
        writer.add_scalar(f"Codebook/L{i}/ActiveRatio", stats["active_ratio"], epoch)
        writer.add_scalar(f"Codebook/L{i}/Perplexity", stats["perplexity"], epoch)
        writer.add_scalar(f"Codebook/L{i}/MinL2Dist", stats["min_l2_dist"], epoch)
        writer.add_scalar(f"Codebook/L{i}/CollapseRatio", stats["collapse_ratio"], epoch)
