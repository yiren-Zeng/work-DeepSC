import torch


@torch.no_grad()
def compute_codebook_utilization(model, dataloader, max_batches=None, device=None):
    if device is None:
        device = model.device

    model.eval()
    num_layers = len(model.vector_quantizers)
    all_indices_src = [[] for _ in range(num_layers)]
    all_indices_raq = [[] for _ in range(num_layers)]
    last_w_trg = [None for _ in range(num_layers)]
    use_raq = getattr(model, "use_raq", False)

    for batch_count, images in enumerate(dataloader):
        if max_batches is not None and batch_count >= max_batches:
            break

        images = images.to(device)
        encoder_features = model.semantic_encoder(images)
        encoder_features[-1] = model.bottleneck_attention(encoder_features[-1])
        for i, feat in enumerate(encoder_features):
            _, _, encoding_idx_src = model.vector_quantizers[i](feat)
            all_indices_src[i].append(encoding_idx_src.cpu())
            if use_raq:
                k_trg = int(model.raq_target_list[i])
                w_trg = model._generate_raq_codebook(i, k_trg)
                _, _, encoding_idx_raq = model.vector_quantizers[i].forward_raq(feat, w_trg)
                all_indices_raq[i].append(encoding_idx_raq.cpu())
                last_w_trg[i] = w_trg.detach()

    results = {"src": [], "quantizer_type": getattr(model, "quantizer_type", "simvq")}
    if use_raq:
        results["raq"] = []
    for i in range(num_layers):
        src_all = torch.cat(all_indices_src[i], dim=0)
        quantizer = model.vector_quantizers[i]
        src_stats = quantizer.compute_codebook_stats(src_all, model.num_embeddings_list[i])
        src_codebook = quantizer.transformed_weight()
        src_l2_stats = quantizer.compute_min_l2_distance(src_codebook)
        src_stats["min_l2_dist"] = src_l2_stats["min_l2_dist"]
        src_stats["collapse_count"] = src_l2_stats["collapse_count"]
        src_stats["collapse_ratio"] = src_l2_stats["collapse_ratio"]
        src_stats["distance_reference_count"] = src_l2_stats["distance_reference_count"]
        src_stats["distance_stats_exact"] = src_l2_stats["distance_stats_exact"]
        results["src"].append(src_stats)

        if use_raq:
            raq_all = torch.cat(all_indices_raq[i], dim=0)
            k_trg = int(model.raq_target_list[i])
            raq_stats = quantizer.compute_codebook_stats(raq_all, k_trg)
            raq_l2_stats = quantizer.compute_min_l2_distance(last_w_trg[i])
            raq_stats["min_l2_dist"] = raq_l2_stats["min_l2_dist"]
            raq_stats["collapse_count"] = raq_l2_stats["collapse_count"]
            raq_stats["collapse_ratio"] = raq_l2_stats["collapse_ratio"]
            raq_stats["distance_reference_count"] = raq_l2_stats["distance_reference_count"]
            raq_stats["distance_stats_exact"] = raq_l2_stats["distance_stats_exact"]
            results["raq"].append(raq_stats)

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
        distance_mode = "精确" if s["distance_stats_exact"] else f"采样{s['distance_reference_count']}"
        quantizer_label = results.get("quantizer_type", "simvq")
        print(f"  [{quantizer_label}] 活跃率: {s['active_ratio']:.2%}  |  "
              f"活跃码字: {s['active_count']}/{k_src}  |  "
              f"死码字: {s['dead_count']}  |  "
              f"困惑度: {s['perplexity']:.1f}/{k_src}  |  "
              f"最小L2距离({distance_mode}): {s['min_l2_dist']:.4f}  |  "
              f"坍缩码字: {s['collapse_count']}/{k_src} ({s['collapse_ratio']:.2%})")

        if "raq" in results:
            r = results["raq"][i]
            k_raq = r["usage_counts"].numel()
            distance_mode = "精确" if r["distance_stats_exact"] else f"采样{r['distance_reference_count']}"
            print(f"  [RAQ] 活跃率: {r['active_ratio']:.2%}  |  "
                  f"活跃码字: {r['active_count']}/{k_raq}  |  "
                  f"死码字: {r['dead_count']}  |  "
                  f"困惑度: {r['perplexity']:.1f}/{k_raq}  |  "
                  f"最小L2距离({distance_mode}): {r['min_l2_dist']:.4f}  |  "
                  f"坍缩码字: {r['collapse_count']}/{k_raq} ({r['collapse_ratio']:.2%})")

    src_avg = sum(s["active_ratio"] for s in results["src"]) / num_layers
    print("\n" + "-" * 80)
    print(f"  [{results.get('quantizer_type', 'simvq')}] 平均活跃率: {src_avg:.2%}")
    if "raq" in results:
        raq_avg = sum(s["active_ratio"] for s in results["raq"]) / num_layers
        print(f"  [RAQ] 平均活跃率: {raq_avg:.2%}")
    print("=" * 80 + "\n")


def write_codebook_tensorboard(writer, results, epoch):
    for i, stats in enumerate(results["src"]):
        writer.add_scalar(f"Codebook/L{i}/ActiveRatio", stats["active_ratio"], epoch)
        writer.add_scalar(f"Codebook/L{i}/Perplexity", stats["perplexity"], epoch)
        writer.add_scalar(f"Codebook/L{i}/MinL2Dist", stats["min_l2_dist"], epoch)
        writer.add_scalar(f"Codebook/L{i}/CollapseRatio", stats["collapse_ratio"], epoch)
    for i, stats in enumerate(results.get("raq", [])):
        writer.add_scalar(f"CodebookRAQ/L{i}/ActiveRatio", stats["active_ratio"], epoch)
        writer.add_scalar(f"CodebookRAQ/L{i}/Perplexity", stats["perplexity"], epoch)
        writer.add_scalar(f"CodebookRAQ/L{i}/MinL2Dist", stats["min_l2_dist"], epoch)
        writer.add_scalar(f"CodebookRAQ/L{i}/CollapseRatio", stats["collapse_ratio"], epoch)
