import torch

from models.independent_raq_rvq import quantize_independent_raq_rvq
from models.shared_raq_rvq import quantize_shared_raq_rvq


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
    independent_rvq = getattr(
        model, "use_independent_raq_rvq", False
    )
    independent_depth = int(
        getattr(model, "independent_raq_rvq_depth", 2)
    )
    all_indices_raq_stages = [
        [[] for _ in range(independent_depth)]
        for _ in range(num_layers)
    ]
    last_w_trg_stages = [
        [None for _ in range(independent_depth)]
        for _ in range(num_layers)
    ]

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
                if independent_rvq:
                    stage_k_list = (
                        model.independent_raq_rvq_k_lists[i]
                    )
                    source_codebook = (
                        model.vector_quantizers[i].transformed_weight()
                    )
                    generators = (
                        model.raqs[i],
                        model.raqs_rvq_stage2[i],
                    )
                    stage_codebooks = [
                        generator.generate_codebook_transformer(
                            int(stage_k), source_codebook
                        )
                        for generator, stage_k in zip(
                            generators, stage_k_list
                        )
                    ]
                    rvq_result = quantize_independent_raq_rvq(
                        model.vector_quantizers[i],
                        feat,
                        stage_codebooks,
                    )
                    for stage_index, (
                        stage_indices,
                        stage_codebook,
                    ) in enumerate(zip(
                        rvq_result["indices"], stage_codebooks
                    )):
                        all_indices_raq_stages[i][stage_index].append(
                            stage_indices.cpu()
                        )
                        last_w_trg_stages[i][stage_index] = (
                            stage_codebook.detach()
                        )
                    continue

                k_trg = int(model.raq_target_list[i])
                w_trg = model._generate_raq_codebook(i, k_trg)
                if getattr(model, "use_shared_raq_rvq", False):
                    rvq_result = quantize_shared_raq_rvq(
                        model.vector_quantizers[i],
                        feat,
                        w_trg,
                        depth=model.shared_raq_rvq_depth,
                    )
                    encoding_idx_raq = rvq_result["indices"]
                else:
                    _, _, encoding_idx_raq = model.vector_quantizers[i].forward_raq(
                        feat, w_trg
                    )
                all_indices_raq[i].append(encoding_idx_raq.cpu())
                last_w_trg[i] = w_trg.detach()

    results = {"src": [], "quantizer_type": getattr(model, "quantizer_type", "simvq")}
    if use_raq:
        if independent_rvq:
            results["raq_stages"] = [
                [] for _ in range(num_layers)
            ]
        else:
            results["raq"] = []
        results["raq_rvq_depth"] = (
            independent_depth
            if independent_rvq
            else (
                int(model.shared_raq_rvq_depth)
                if getattr(model, "use_shared_raq_rvq", False)
                else 1
            )
        )
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

        if use_raq and not independent_rvq:
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
        elif use_raq:
            quantizer = model.vector_quantizers[i]
            for stage_index, stage_k in enumerate(
                model.independent_raq_rvq_k_lists[i]
            ):
                stage_all = torch.cat(
                    all_indices_raq_stages[i][stage_index], dim=0
                )
                stage_stats = quantizer.compute_codebook_stats(
                    stage_all, int(stage_k)
                )
                stage_l2_stats = quantizer.compute_min_l2_distance(
                    last_w_trg_stages[i][stage_index]
                )
                stage_stats.update({
                    "min_l2_dist": stage_l2_stats["min_l2_dist"],
                    "collapse_count": stage_l2_stats["collapse_count"],
                    "collapse_ratio": stage_l2_stats["collapse_ratio"],
                    "distance_reference_count": stage_l2_stats[
                        "distance_reference_count"
                    ],
                    "distance_stats_exact": stage_l2_stats[
                        "distance_stats_exact"
                    ],
                    "stage": stage_index,
                })
                results["raq_stages"][i].append(stage_stats)

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
            raq_label = (
                f"RAQ-RVQ d={results['raq_rvq_depth']}"
                if results.get("raq_rvq_depth", 1) > 1
                else "RAQ"
            )
            print(f"  [{raq_label}] 活跃率: {r['active_ratio']:.2%}  |  "
                  f"活跃码字: {r['active_count']}/{k_raq}  |  "
                  f"死码字: {r['dead_count']}  |  "
                  f"困惑度: {r['perplexity']:.1f}/{k_raq}  |  "
                  f"最小L2距离({distance_mode}): {r['min_l2_dist']:.4f}  |  "
                  f"坍缩码字: {r['collapse_count']}/{k_raq} ({r['collapse_ratio']:.2%})")
        for stage_index, r in enumerate(
            results.get("raq_stages", [[] for _ in range(num_layers)])[i]
        ):
            k_raq = r["usage_counts"].numel()
            distance_mode = (
                "精确"
                if r["distance_stats_exact"]
                else f"采样{r['distance_reference_count']}"
            )
            print(
                f"  [Independent RAQ-RVQ stage={stage_index}] "
                f"活跃率: {r['active_ratio']:.2%}  |  "
                f"活跃码字: {r['active_count']}/{k_raq}  |  "
                f"死码字: {r['dead_count']}  |  "
                f"困惑度: {r['perplexity']:.1f}/{k_raq}  |  "
                f"最小L2距离({distance_mode}): "
                f"{r['min_l2_dist']:.4f}  |  "
                f"坍缩码字: {r['collapse_count']}/{k_raq} "
                f"({r['collapse_ratio']:.2%})"
            )

    src_avg = sum(s["active_ratio"] for s in results["src"]) / num_layers
    print("\n" + "-" * 80)
    print(f"  [{results.get('quantizer_type', 'simvq')}] 平均活跃率: {src_avg:.2%}")
    if "raq" in results:
        raq_avg = sum(s["active_ratio"] for s in results["raq"]) / num_layers
        raq_label = (
            f"RAQ-RVQ d={results['raq_rvq_depth']}"
            if results.get("raq_rvq_depth", 1) > 1
            else "RAQ"
        )
        print(f"  [{raq_label}] 平均活跃率: {raq_avg:.2%}")
    if "raq_stages" in results:
        stage_values = [
            stats["active_ratio"]
            for scale_stats in results["raq_stages"]
            for stats in scale_stats
        ]
        print(
            "  [Independent RAQ-RVQ] 平均活跃率: "
            f"{sum(stage_values) / len(stage_values):.2%}"
        )
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
    for i, scale_stats in enumerate(results.get("raq_stages", [])):
        for stage_index, stats in enumerate(scale_stats):
            prefix = f"CodebookRAQ/L{i}/Stage{stage_index}"
            writer.add_scalar(
                f"{prefix}/ActiveRatio", stats["active_ratio"], epoch
            )
            writer.add_scalar(
                f"{prefix}/Perplexity", stats["perplexity"], epoch
            )
            writer.add_scalar(
                f"{prefix}/MinL2Dist", stats["min_l2_dist"], epoch
            )
            writer.add_scalar(
                f"{prefix}/CollapseRatio",
                stats["collapse_ratio"],
                epoch,
            )
