import csv
import json
import math
import os

import torch


EPOCH_METRIC_FIELDS = [
    "run_id",
    "epoch",
    "train_recon",
    "train_vq",
    "train_total",
    "val_recon",
    "val_vq",
    "val_psnr",
    "val_ms_ssim",
    "best_val_recon",
    "is_best",
    "phase",
    "channel_prob",
    "channel_usage_ratio",
    "mean_channel_snr",
    "source_bits_per_image",
    "source_bpp",
    "transmission_ratio",
    "learning_rate",
    "train_rq_codebook_loss_per_scale",
    "train_rq_commitment_loss_per_scale",
    "train_rq_codebook_per_depth",
    "train_rq_commitment_per_depth",
    "train_rq_residual_norm_per_depth",
    "train_rq_usage_per_depth",
    "train_rq_perplexity_per_depth",
    "train_rq_aggregate_usage_per_scale",
    "train_rq_aggregate_perplexity_per_scale",
    "train_rq_projection_grad_norm_per_scale",
    "val_rq_codebook_loss_per_scale",
    "val_rq_commitment_loss_per_scale",
    "val_rq_codebook_per_depth",
    "val_rq_commitment_per_depth",
    "val_rq_residual_norm_per_depth",
    "val_rq_usage_per_depth",
    "val_rq_perplexity_per_depth",
    "val_rq_aggregate_usage_per_scale",
    "val_rq_aggregate_perplexity_per_scale",
    "val_rq_projection_grad_norm_per_scale",
]

CODEBOOK_METRIC_FIELDS = [
    "run_id",
    "epoch",
    "layer",
    "depth",
    "scope",
    "codebook_size",
    "active_ratio",
    "active_count",
    "dead_count",
    "perplexity",
    "codebook_loss",
    "commitment",
    "residual_norm",
    "projection_grad_norm",
    "restarted_codes",
    "usage_counts",
    "min_l2_dist",
    "collapse_count",
    "collapse_ratio",
    "distance_reference_count",
    "distance_stats_exact",
]


def _ensure_parent(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _existing_fieldnames(path, defaults):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return list(defaults), True
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    return (header or list(defaults)), False


def _float_field(value, digits=10):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return ""
        value = value.detach().cpu().item()
    value = float(value)
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def _int_field(value):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return ""
        value = value.detach().cpu().item()
    return int(value)


def _usage_field(value):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    return json.dumps([int(item) for item in value], separators=(",", ":"))


def append_epoch_record(path, row):
    _ensure_parent(path)
    fieldnames, write_header = _existing_fieldnames(path, EPOCH_METRIC_FIELDS)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        # Existing experiment CSVs keep their historical header; new optional
        # metrics are ignored rather than making resumed runs fail.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def rq_epoch_metric_fields(prefix, diagnostics):
    """Flatten per-scale RQ diagnostics into explicit JSON-valued CSV fields."""
    if prefix not in {"train", "val"}:
        raise ValueError("prefix must be 'train' or 'val'")
    diagnostics = list(diagnostics or [])

    def encoded(values):
        return json.dumps(values, separators=(",", ":"), allow_nan=False)

    def finite_or_none(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return [finite_or_none(item) for item in value.reshape(-1)]
            value = value.detach().cpu().item()
        value = float(value)
        return value if math.isfinite(value) else None

    def scalar_list(key):
        return encoded([finite_or_none(scale.get(key)) for scale in diagnostics])

    def nested_list(key):
        values = []
        for scale in diagnostics:
            values.append(
                [finite_or_none(item) for item in scale.get(key, [])]
            )
        return encoded(values)

    return {
        f"{prefix}_rq_codebook_loss_per_scale": scalar_list("codebook_loss"),
        f"{prefix}_rq_commitment_loss_per_scale": scalar_list(
            "commitment_loss"
        ),
        f"{prefix}_rq_codebook_per_depth": nested_list(
            "codebook_per_depth"
        ),
        f"{prefix}_rq_commitment_per_depth": nested_list(
            "commitment_per_depth"
        ),
        f"{prefix}_rq_residual_norm_per_depth": nested_list(
            "residual_norm_per_depth"
        ),
        f"{prefix}_rq_usage_per_depth": nested_list("usage_per_depth"),
        f"{prefix}_rq_perplexity_per_depth": nested_list(
            "perplexity_per_depth"
        ),
        f"{prefix}_rq_aggregate_usage_per_scale": scalar_list(
            "aggregate_usage"
        ),
        f"{prefix}_rq_aggregate_perplexity_per_scale": scalar_list(
            "aggregate_perplexity"
        ),
        f"{prefix}_rq_projection_grad_norm_per_scale": scalar_list(
            "projection_grad_norm"
        ),
    }


def _codebook_row(run_id, epoch, layer, depth, scope, codebook_size, stats):
    return {
        "run_id": run_id,
        "epoch": epoch,
        "layer": layer,
        "depth": depth,
        "scope": scope,
        "codebook_size": codebook_size,
        "active_ratio": _float_field(stats.get("active_ratio")),
        "active_count": _int_field(stats.get("active_count")),
        "dead_count": _int_field(stats.get("dead_count", stats.get("dead_codes"))),
        "perplexity": _float_field(stats.get("perplexity")),
        "codebook_loss": _float_field(
            stats.get("codebook_loss", stats.get("codebook"))
        ),
        "commitment": _float_field(
            stats.get("commitment", stats.get("commitment_loss"))
        ),
        "residual_norm": _float_field(stats.get("residual_norm")),
        "projection_grad_norm": _float_field(
            stats.get("projection_grad_norm")
        ),
        "restarted_codes": _int_field(stats.get("restarted_codes", 0)),
        "usage_counts": _usage_field(stats.get("usage_counts")),
        "min_l2_dist": _float_field(stats.get("min_l2_dist")),
        "collapse_count": _int_field(stats.get("collapse_count")),
        "collapse_ratio": _float_field(stats.get("collapse_ratio")),
        "distance_reference_count": _int_field(stats.get("distance_reference_count")),
        "distance_stats_exact": (
            int(bool(stats["distance_stats_exact"]))
            if stats.get("distance_stats_exact") is not None
            else ""
        ),
    }


def append_codebook_records(path, run_id, epoch, results, num_embeddings_list):
    _ensure_parent(path)
    fieldnames, write_header = _existing_fieldnames(path, CODEBOOK_METRIC_FIELDS)
    supports_depth_rows = "depth" in fieldnames and "scope" in fieldnames

    rows = []
    for layer, stats in enumerate(results["src"]):
        configured_sizes = num_embeddings_list[layer]
        if isinstance(configured_sizes, torch.Tensor):
            configured_sizes = configured_sizes.detach().cpu().reshape(-1).tolist()
        if isinstance(configured_sizes, (list, tuple)):
            depth_codebook_sizes = [int(value) for value in configured_sizes]
        else:
            result_size_lists = results.get("rq_codebook_size_lists")
            if result_size_lists is not None and layer < len(result_size_lists):
                depth_codebook_sizes = [
                    int(value) for value in result_size_lists[layer]
                ]
            else:
                depth_codebook_sizes = []

        flat_codebook_size = (
            int(configured_sizes)
            if not isinstance(configured_sizes, (list, tuple))
            else None
        )
        aggregate_codebook_size = int(
            stats.get(
                "codebook_size",
                sum(depth_codebook_sizes)
                if depth_codebook_sizes
                else flat_codebook_size,
            )
        )
        if supports_depth_rows:
            for depth_index, depth_stats in enumerate(
                stats.get("per_depth", [])
            ):
                depth_codebook_size = int(
                    depth_stats.get(
                        "codebook_size",
                        depth_codebook_sizes[depth_index]
                        if depth_index < len(depth_codebook_sizes)
                        else flat_codebook_size,
                    )
                )
                rows.append(
                    _codebook_row(
                        run_id,
                        epoch,
                        layer,
                        int(depth_stats.get("depth", len(rows))),
                        "depth",
                        depth_codebook_size,
                        depth_stats,
                    )
                )
        rows.append(
            _codebook_row(
                run_id,
                epoch,
                layer,
                "",
                "aggregate",
                aggregate_codebook_size,
                stats,
            )
        )

    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
