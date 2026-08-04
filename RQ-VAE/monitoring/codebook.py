import math

import torch


def _usage_stats(indices, num_embeddings):
    flat = indices.detach().reshape(-1).long().cpu()
    if flat.numel() and (flat.min().item() < 0 or flat.max().item() >= num_embeddings):
        raise ValueError(f"Code indices must be in [0, {num_embeddings - 1}].")
    usage_counts = torch.bincount(flat, minlength=num_embeddings).float()
    total = int(flat.numel())
    active_count = int((usage_counts > 0).sum().item())
    if total:
        probabilities = usage_counts / total
        probabilities = probabilities[probabilities > 0]
        perplexity = float(torch.exp(-(probabilities * probabilities.log()).sum()).item())
    else:
        perplexity = 0.0
    return {
        "usage_counts": usage_counts,
        "active_count": active_count,
        "active_ratio": active_count / num_embeddings,
        "dead_count": int(num_embeddings - active_count),
        "perplexity": perplexity,
    }


def _diagnostics_dict(quantizer):
    getter = getattr(quantizer, "get_last_diagnostics", None)
    if callable(getter):
        diagnostics = getter()
        if isinstance(diagnostics, dict):
            return diagnostics
    for name in ("last_diagnostics", "last_stats"):
        diagnostics = getattr(quantizer, name, None)
        if isinstance(diagnostics, dict):
            return diagnostics
    return {}


def _as_float_list(value, depth):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        values = value.detach().reshape(-1).cpu().tolist()
    elif isinstance(value, (list, tuple)):
        values = []
        for item in value:
            if isinstance(item, torch.Tensor):
                values.extend(item.detach().reshape(-1).cpu().tolist())
            else:
                values.append(float(item))
    else:
        values = [float(value)]
    if len(values) == 1 and depth > 1:
        values = values * depth
    if len(values) < depth:
        values.extend([float("nan")] * (depth - len(values)))
    return [float(item) for item in values[:depth]]


def _diagnostic_values(diagnostics, names, depth):
    for name in names:
        if name in diagnostics:
            return _as_float_list(diagnostics[name], depth)
    return None


def _accumulate_diagnostics(accumulator, diagnostics, depth, batch_size):
    candidates = {
        "codebook_per_depth": (
            "codebook_per_depth",
            "codebook_loss_per_depth",
            "codebook_losses",
        ),
        "commitment_per_depth": (
            "commitment_per_depth",
            "commitment_loss_per_depth",
            "commitment_losses",
        ),
        "residual_norm_per_depth": (
            "residual_norm_per_depth",
            "residual_norms",
            "residual_per_depth",
        ),
    }
    for output_name, names in candidates.items():
        values = _diagnostic_values(diagnostics, names, depth)
        if values is None:
            continue
        sums = accumulator.setdefault(output_name, [0.0] * depth)
        weights = accumulator.setdefault(output_name + "_weights", [0.0] * depth)
        for index, value in enumerate(values):
            if math.isfinite(value):
                sums[index] += value * batch_size
                weights[index] += batch_size


def _finalize_diagnostics(accumulator, name, depth):
    sums = accumulator.get(name, [0.0] * depth)
    weights = accumulator.get(name + "_weights", [0.0] * depth)
    return [
        sums[index] / weights[index] if weights[index] else float("nan")
        for index in range(depth)
    ]


def _restart_counts(quantizer, diagnostics, depth):
    per_depth_value = None
    for name in (
        "restarted_codes_per_depth",
        "restart_counts",
    ):
        if name in diagnostics:
            per_depth_value = diagnostics[name]
            break

    if per_depth_value is None and hasattr(quantizer, "codebooks"):
        counts = []
        for codebook in quantizer.codebooks:
            for name in ("restarted_codes", "restart_count", "num_restarted_codes"):
                if hasattr(codebook, name):
                    counts.append(getattr(codebook, name))
                    break
        if counts:
            per_depth_value = counts

    aggregate_value = diagnostics.get("restarted_codes")
    if aggregate_value is None:
        for name in ("restarted_codes", "restart_count", "num_restarted_codes"):
            if hasattr(quantizer, name):
                candidate = getattr(quantizer, name)
                if not callable(candidate):
                    aggregate_value = candidate
                    break

    if per_depth_value is None and aggregate_value is not None:
        per_depth_value = aggregate_value
    values = (
        _as_float_list(per_depth_value, depth)
        if per_depth_value is not None
        else [0.0] * depth
    )
    values = [0 if not math.isfinite(item) else int(item) for item in values]
    if aggregate_value is not None:
        aggregate_values = _as_float_list(aggregate_value, 1)
        aggregate = int(aggregate_values[0]) if aggregate_values else 0
    elif per_depth_value is None:
        aggregate = 0
    else:
        aggregate = sum(values)
    return values, aggregate


def _projected_codebook_weight(codebook):
    projected_weight = getattr(codebook, "projected_weight", None)
    if callable(projected_weight):
        weight = projected_weight()
        if isinstance(weight, torch.Tensor):
            return weight

    transformed_weight = getattr(codebook, "transformed_weight", None)
    if callable(transformed_weight):
        weight = transformed_weight()
        if isinstance(weight, torch.Tensor):
            return weight

    embed = getattr(codebook, "embed", None)
    weight = getattr(embed, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    weight = getattr(codebook, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight
    return None


def _codebook_weight(quantizer, num_embeddings, depth_index=None):
    codebooks = getattr(quantizer, "codebooks", None)
    if depth_index is not None and codebooks is not None:
        if 0 <= int(depth_index) < len(codebooks):
            weight = _projected_codebook_weight(codebooks[int(depth_index)])
            if isinstance(weight, torch.Tensor):
                return weight[:num_embeddings]

    transformed = getattr(quantizer, "transformed_weight", None)
    if callable(transformed):
        try:
            weight = transformed()
        except TypeError:
            weight = None
        if isinstance(weight, torch.Tensor):
            if weight.ndim == 3:
                selected_depth = 0 if depth_index is None else int(depth_index)
                weight = weight[selected_depth]
            return weight[:num_embeddings]

    if codebooks is not None and len(codebooks):
        selected_depth = 0 if depth_index is None else int(depth_index)
        weight = _projected_codebook_weight(codebooks[selected_depth])
        if isinstance(weight, torch.Tensor):
            return weight[:num_embeddings]
    codebook = getattr(quantizer, "codebook", None)
    weight = _projected_codebook_weight(codebook)
    if isinstance(weight, torch.Tensor):
        return weight[:num_embeddings]
    weight = getattr(quantizer, "weight", None)
    if isinstance(weight, torch.Tensor):
        return weight[:num_embeddings]
    return None


def _distance_stats_from_weight(quantizer, weight):
    defaults = {
        "min_l2_dist": float("nan"),
        "collapse_count": 0,
        "collapse_ratio": 0.0,
        "distance_reference_count": 0,
        "distance_stats_exact": False,
    }
    if weight is None or weight.shape[0] < 2:
        return defaults

    calculator = getattr(quantizer, "compute_min_l2_distance", None)
    if callable(calculator):
        try:
            return {**defaults, **calculator(weight)}
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass

    weight = weight.detach().float()
    distances = torch.cdist(weight, weight)
    distances.fill_diagonal_(float("inf"))
    nearest = distances.min(dim=1).values
    collapse = nearest < 0.1
    return {
        "min_l2_dist": float(nearest.min().item()),
        "collapse_count": int(collapse.sum().item()),
        "collapse_ratio": float(collapse.float().mean().item()),
        "distance_reference_count": int(weight.shape[0]),
        "distance_stats_exact": True,
    }


def _distance_stats(quantizer, num_embeddings, depth_index=None):
    weight = _codebook_weight(
        quantizer, num_embeddings, depth_index=depth_index
    )
    return _distance_stats_from_weight(quantizer, weight)


def _projection_grad_norm(quantizer):
    codebooks = list(getattr(quantizer, "codebooks", ()) or ())
    codebook = getattr(quantizer, "codebook", None)
    if codebook is not None:
        codebooks.append(codebook)

    # A shared residual quantizer exposes the same object once per depth.
    # Deduplicate it so its projection gradient is not counted repeatedly.
    projections = []
    seen = set()
    for item in codebooks:
        projection = getattr(item, "proj", None)
        if projection is None:
            nested = getattr(item, "codebook", None)
            projection = getattr(nested, "proj", None)
        if projection is not None and id(projection) not in seen:
            seen.add(id(projection))
            projections.append(projection)

    squared_norm = None
    if not projections:
        return float("nan")
    for projection in projections:
        for parameter in projection.parameters():
            if parameter.grad is None:
                continue
            value = parameter.grad.detach().float().pow(2).sum()
            squared_norm = value if squared_norm is None else squared_norm + value
    if squared_norm is None:
        return float("nan")
    return float(torch.sqrt(squared_norm).item())


def _is_rq_quantizer(model, quantizer):
    return (
        str(getattr(model, "quantizer_type", "")).lower()
        in {"rq_ema", "residual_simvq", "stagewise_residual_simvq"}
        or quantizer.__class__.__name__
        in {
            "RQEMAQuantizer",
            "ResidualSimVQQuantizer",
            "StagewiseResidualSimVQQuantizer",
        }
    )


def _is_stagewise_quantizer(model, quantizer):
    return (
        str(getattr(model, "quantizer_type", "")).lower()
        == "stagewise_residual_simvq"
        or quantizer.__class__.__name__ == "StagewiseResidualSimVQQuantizer"
    )


def _codebook_size_from_module(codebook):
    for name in ("num_embeddings", "codebook_size"):
        value = getattr(codebook, name, None)
        if value is not None:
            return int(value)
    embed = getattr(codebook, "embed", None)
    value = getattr(embed, "num_embeddings", None)
    if value is not None:
        return int(value)
    weight = _projected_codebook_weight(codebook)
    return int(weight.shape[0]) if isinstance(weight, torch.Tensor) else None


def _depth_codebook_sizes(model, quantizer, scale, depth):
    value = None
    for name in (
        "num_embeddings_per_depth",
        "rq_codebook_sizes",
        "rq_codebook_size_list",
        "codebook_size_list",
    ):
        candidate = getattr(quantizer, name, None)
        if candidate is not None:
            value = candidate
            break

    if value is None:
        nested = getattr(model, "rq_codebook_size_lists", None)
        if nested is not None:
            value = nested[scale]

    if value is None and _is_stagewise_quantizer(model, quantizer):
        codebooks = getattr(quantizer, "codebooks", None)
        if codebooks is not None:
            inferred = [_codebook_size_from_module(item) for item in codebooks]
            if all(size is not None for size in inferred):
                value = inferred

    if value is None:
        flat_sizes = getattr(model, "num_embeddings_list", None)
        if flat_sizes is None:
            value = [int(getattr(quantizer, "num_embeddings"))] * depth
        else:
            value = [int(flat_sizes[scale])] * depth

    sizes = [int(item) for item in value]
    if len(sizes) != depth:
        raise ValueError(
            f"Residual quantizer scale {scale} has {len(sizes)} codebook "
            f"sizes, expected {depth}."
        )
    if any(size <= 0 for size in sizes):
        raise ValueError("Residual quantizer codebook sizes must be positive.")
    return sizes


def _aggregate_disjoint_usage_stats(per_depth):
    usage_counts = torch.cat(
        [stats["usage_counts"].detach().reshape(-1).cpu() for stats in per_depth]
    ).float()
    codebook_size = int(usage_counts.numel())
    total = int(usage_counts.sum().item())
    active_count = int((usage_counts > 0).sum().item())
    if total:
        probabilities = usage_counts / total
        probabilities = probabilities[probabilities > 0]
        perplexity = float(
            torch.exp(-(probabilities * probabilities.log()).sum()).item()
        )
    else:
        perplexity = 0.0
    return {
        "usage_counts": usage_counts,
        "active_count": active_count,
        "active_ratio": active_count / codebook_size if codebook_size else 0.0,
        "dead_count": codebook_size - active_count,
        "perplexity": perplexity,
        "codebook_size": codebook_size,
    }


@torch.no_grad()
def compute_codebook_utilization(model, dataloader, max_batches=None, device=None):
    if device is None:
        device = getattr(model, "encoder_device", getattr(model, "device", "cpu"))
    device = torch.device(device)

    was_training = model.training
    model.eval()
    num_layers = len(model.vector_quantizers)
    all_indices_src = [[] for _ in range(num_layers)]
    diagnostic_accumulators = [{} for _ in range(num_layers)]
    last_diagnostics = [{} for _ in range(num_layers)]

    try:
        for batch_count, images in enumerate(dataloader):
            if max_batches is not None and batch_count >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            to_encoder = getattr(model, "_to_encoder_device", None)
            if callable(to_encoder):
                images = to_encoder(images)
            encoder_features = model.semantic_encoder(images)
            if isinstance(encoder_features, torch.Tensor):
                encoder_features = [encoder_features]
            else:
                encoder_features = list(encoder_features)
            if len(encoder_features) > 0 and hasattr(model, "bottleneck_attention"):
                encoder_features[-1] = model.bottleneck_attention(encoder_features[-1])

            for scale, feat in enumerate(encoder_features):
                _, _, encoding_indices = model.vector_quantizers[scale](feat)
                all_indices_src[scale].append(encoding_indices.detach().cpu())
                quantizer = model.vector_quantizers[scale]
                if _is_rq_quantizer(model, quantizer):
                    depth = int(
                        getattr(
                            quantizer,
                            "rq_depth",
                            encoding_indices.shape[-1] if encoding_indices.ndim >= 4 else 1,
                        )
                    )
                    diagnostics = _diagnostics_dict(quantizer)
                    last_diagnostics[scale] = diagnostics
                    _accumulate_diagnostics(
                        diagnostic_accumulators[scale], diagnostics, depth, images.shape[0]
                    )
    finally:
        model.train(was_training)

    if num_layers and any(not scale_indices for scale_indices in all_indices_src):
        raise ValueError("Codebook monitoring received no usable batches from the dataloader.")

    results = {
        "src": [],
        "rq_scales": [],
        "quantizer_type": getattr(model, "quantizer_type", "simvq"),
    }
    if str(results["quantizer_type"]).lower() == "stagewise_residual_simvq":
        results["rq_codebook_size_lists"] = []
    for scale in range(num_layers):
        src_all = torch.cat(all_indices_src[scale], dim=0)
        quantizer = model.vector_quantizers[scale]

        if not _is_rq_quantizer(model, quantizer):
            num_embeddings = int(model.num_embeddings_list[scale])
            src_stats = quantizer.compute_codebook_stats(src_all, num_embeddings)
            src_stats.update(_distance_stats(quantizer, num_embeddings))
            src_stats["codebook_size"] = num_embeddings
            results["src"].append(src_stats)
            continue

        if src_all.ndim < 4:
            raise ValueError(
                f"Residual quantizer scale {scale} indices must have shape "
                f"[B,H,W,D], got {tuple(src_all.shape)}"
            )
        depth = int(src_all.shape[-1])
        configured_depth = int(getattr(quantizer, "rq_depth", depth))
        if depth != configured_depth:
            raise ValueError(
                f"Residual quantizer scale {scale} returned depth {depth}, "
                f"expected {configured_depth}."
            )
        depth_codebook_sizes = _depth_codebook_sizes(
            model, quantizer, scale, depth
        )
        is_stagewise = _is_stagewise_quantizer(model, quantizer)
        if is_stagewise:
            results.setdefault("rq_codebook_size_lists", []).append(
                list(depth_codebook_sizes)
            )

        codebook = _finalize_diagnostics(
            diagnostic_accumulators[scale], "codebook_per_depth", depth
        )
        commitment = _finalize_diagnostics(
            diagnostic_accumulators[scale], "commitment_per_depth", depth
        )
        residual_norm = _finalize_diagnostics(
            diagnostic_accumulators[scale], "residual_norm_per_depth", depth
        )
        restarted_per_depth, restarted_aggregate = _restart_counts(
            quantizer, last_diagnostics[scale], depth
        )
        per_depth = []
        usage_per_depth = []
        usage_counts_per_depth = []
        perplexity_per_depth = []
        for depth_index in range(depth):
            depth_codebook_size = depth_codebook_sizes[depth_index]
            stats = _usage_stats(
                src_all[..., depth_index], depth_codebook_size
            )
            stats.update(
                _distance_stats(
                    quantizer,
                    depth_codebook_size,
                    depth_index=depth_index,
                )
            )
            stats.update(
                {
                    "depth": depth_index,
                    "codebook_size": depth_codebook_size,
                    "codebook_loss": codebook[depth_index],
                    "commitment": commitment[depth_index],
                    "commitment_loss": commitment[depth_index],
                    "residual_norm": residual_norm[depth_index],
                    "restarted_codes": restarted_per_depth[depth_index],
                }
            )
            per_depth.append(stats)
            usage_per_depth.append(stats["active_ratio"])
            usage_counts_per_depth.append(stats["usage_counts"])
            perplexity_per_depth.append(stats["perplexity"])

        if is_stagewise:
            aggregate = _aggregate_disjoint_usage_stats(per_depth)
            weights = [
                _codebook_weight(
                    quantizer,
                    codebook_size,
                    depth_index=depth_index,
                )
                for depth_index, codebook_size in enumerate(
                    depth_codebook_sizes
                )
            ]
            aggregate_weight = (
                torch.cat(weights, dim=0)
                if weights and all(
                    isinstance(weight, torch.Tensor) for weight in weights
                )
                else None
            )
            aggregate.update(
                _distance_stats_from_weight(quantizer, aggregate_weight)
            )
        else:
            num_embeddings = depth_codebook_sizes[0]
            aggregate = _usage_stats(src_all, num_embeddings)
            aggregate["codebook_size"] = num_embeddings
            aggregate.update(_distance_stats(quantizer, num_embeddings))
        finite_codebook = [value for value in codebook if math.isfinite(value)]
        codebook_loss = (
            sum(finite_codebook) / len(finite_codebook)
            if finite_codebook
            else float("nan")
        )
        finite_commitments = [value for value in commitment if math.isfinite(value)]
        commitment_loss = (
            sum(finite_commitments) / len(finite_commitments)
            if finite_commitments
            else float("nan")
        )
        final_residual_norm = residual_norm[-1] if residual_norm else float("nan")
        aggregate.update(
            {
                "rq_depth": depth,
                "per_depth": per_depth,
                "codebook_loss": codebook_loss,
                "codebook_per_depth": codebook,
                "commitment": commitment_loss,
                "commitment_loss": commitment_loss,
                "residual_norm": final_residual_norm,
                "commitment_per_depth": commitment,
                "residual_norm_per_depth": residual_norm,
                "usage_per_depth": usage_per_depth,
                "usage_counts_per_depth": usage_counts_per_depth,
                "perplexity_per_depth": perplexity_per_depth,
                "aggregate_usage": aggregate["active_ratio"],
                "aggregate_usage_counts": aggregate["usage_counts"],
                "aggregate_perplexity": aggregate["perplexity"],
                "dead_codes": aggregate["dead_count"],
                "restarted_codes": restarted_aggregate,
                "projection_grad_norm": _projection_grad_norm(quantizer),
            }
        )
        # `src` retains the old per-scale aggregate schema.  RQ-only consumers
        # can use rq_scales/per_depth without breaking legacy CSV/report tools.
        results["src"].append(aggregate)
        results["rq_scales"].append(aggregate)

    return results


def print_codebook_utilization(results, num_embeddings_list=None):
    num_layers = len(results["src"])
    print("\n" + "=" * 80)
    print("  码本利用率统计报告 (Codebook Utilization Report)")
    print("=" * 80)

    for scale, stats in enumerate(results["src"]):
        configured_size = num_embeddings_list[scale] if num_embeddings_list else None
        if isinstance(configured_size, (list, tuple)):
            configured_depth_sizes = [int(value) for value in configured_size]
        else:
            configured_depth_sizes = None
        depth_sizes = [
            int(
                depth_stats.get(
                    "codebook_size",
                    configured_depth_sizes[depth_index]
                    if configured_depth_sizes is not None
                    and depth_index < len(configured_depth_sizes)
                    else configured_size or stats.get("codebook_size", 0),
                )
            )
            for depth_index, depth_stats in enumerate(
                stats.get("per_depth", [])
            )
        ]
        aggregate_size = int(
            stats.get(
                "codebook_size",
                sum(configured_depth_sizes)
                if configured_depth_sizes is not None
                else configured_size or 0,
            )
        )
        k_label = depth_sizes if depth_sizes else (configured_size or "?")
        print(f"\n  Layer {scale} (K={k_label})")
        print("  " + "-" * 60)
        for depth_index, depth_stats in enumerate(stats.get("per_depth", [])):
            depth_size = (
                depth_sizes[depth_index]
                if depth_index < len(depth_sizes)
                else aggregate_size
            )
            codebook_loss = float(
                depth_stats.get("codebook_loss", float("nan"))
            )
            commitment = depth_stats.get("commitment", float("nan"))
            residual = depth_stats.get("residual_norm", float("nan"))
            loss_desc = (
                f"Q {codebook_loss:.6f} | C {commitment:.6f}"
                if math.isfinite(codebook_loss)
                else f"Commitment {commitment:.6f}"
            )
            print(
                f"  Depth {depth_stats['depth']}: 活跃率 "
                f"{depth_stats['active_ratio']:.2%} | "
                f"活跃 {depth_stats['active_count']}/{depth_size} | "
                f"困惑度 {depth_stats['perplexity']:.2f}/{depth_size} | "
                f"{loss_desc} | ResidualNorm {residual:.6f} | "
                f"重启 {depth_stats.get('restarted_codes', 0)}"
            )

        distance_exact = stats.get("distance_stats_exact", False)
        distance_count = stats.get("distance_reference_count", 0)
        distance_mode = "精确" if distance_exact else f"采样{distance_count}"
        quantizer_label = results.get("quantizer_type", "simvq")
        print(
            f"  [{quantizer_label}] 聚合活跃率: {stats['active_ratio']:.2%}  |  "
            f"活跃码字: {stats['active_count']}/{aggregate_size or '?'}  |  "
            f"死码字: {stats['dead_count']}  |  "
            f"困惑度: {stats['perplexity']:.1f}/{aggregate_size or '?'}  |  "
            f"最小L2距离({distance_mode}): {stats.get('min_l2_dist', float('nan')):.4f}  |  "
            f"坍缩码字: {stats.get('collapse_count', 0)}/{aggregate_size or '?'} "
            f"({stats.get('collapse_ratio', 0.0):.2%})  |  "
            f"重启: {stats.get('restarted_codes', 0)}"
        )
        codebook_loss = float(stats.get("codebook_loss", float("nan")))
        if math.isfinite(codebook_loss):
            print(
                f"  Q/C聚合: Q={codebook_loss:.6f} | "
                f"C={float(stats.get('commitment_loss', float('nan'))):.6f}"
            )
        projection_grad_norm = float(
            stats.get("projection_grad_norm", float("nan"))
        )
        if math.isfinite(projection_grad_norm):
            print(f"  投影梯度范数: {projection_grad_norm:.6f}")

    src_avg = (
        sum(stats["active_ratio"] for stats in results["src"]) / num_layers
        if num_layers
        else 0.0
    )
    print("\n" + "-" * 80)
    print(f"  [{results.get('quantizer_type', 'simvq')}] 平均活跃率: {src_avg:.2%}")
    print("=" * 80 + "\n")


def write_codebook_tensorboard(writer, results, epoch):
    for scale, stats in enumerate(results["src"]):
        prefix = f"Codebook/L{scale}"
        writer.add_scalar(f"{prefix}/ActiveRatio", stats["active_ratio"], epoch)
        writer.add_scalar(f"{prefix}/Perplexity", stats["perplexity"], epoch)
        writer.add_scalar(f"{prefix}/DeadCodes", stats["dead_count"], epoch)
        writer.add_scalar(f"{prefix}/RestartedCodes", stats.get("restarted_codes", 0), epoch)
        min_l2 = stats.get("min_l2_dist", float("nan"))
        if math.isfinite(min_l2):
            writer.add_scalar(f"{prefix}/MinL2Dist", min_l2, epoch)
        writer.add_scalar(f"{prefix}/CollapseRatio", stats.get("collapse_ratio", 0.0), epoch)
        codebook_loss = float(stats.get("codebook_loss", float("nan")))
        if math.isfinite(codebook_loss):
            writer.add_scalar(f"{prefix}/CodebookLoss", codebook_loss, epoch)
            writer.add_scalar(
                f"{prefix}/CommitmentLoss",
                stats.get("commitment_loss", float("nan")),
                epoch,
            )
        projection_grad_norm = float(
            stats.get("projection_grad_norm", float("nan"))
        )
        if math.isfinite(projection_grad_norm):
            writer.add_scalar(
                f"{prefix}/ProjectionGradNorm", projection_grad_norm, epoch
            )

        for depth_stats in stats.get("per_depth", []):
            depth = depth_stats["depth"]
            depth_prefix = f"{prefix}/Depth{depth}"
            writer.add_scalar(
                f"{depth_prefix}/ActiveRatio", depth_stats["active_ratio"], epoch
            )
            writer.add_scalar(
                f"{depth_prefix}/Perplexity", depth_stats["perplexity"], epoch
            )
            writer.add_scalar(
                f"{depth_prefix}/DeadCodes", depth_stats["dead_count"], epoch
            )
            writer.add_scalar(
                f"{depth_prefix}/RestartedCodes",
                depth_stats.get("restarted_codes", 0),
                epoch,
            )
            codebook_loss = float(
                depth_stats.get("codebook_loss", float("nan"))
            )
            if math.isfinite(codebook_loss):
                writer.add_scalar(
                    f"{depth_prefix}/CodebookLoss", codebook_loss, epoch
                )
            for name, tag in (
                ("commitment", "Commitment"),
                ("residual_norm", "ResidualNorm"),
            ):
                value = float(depth_stats.get(name, float("nan")))
                if math.isfinite(value):
                    writer.add_scalar(f"{depth_prefix}/{tag}", value, epoch)
