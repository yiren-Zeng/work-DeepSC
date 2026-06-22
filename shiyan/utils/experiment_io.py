import csv
import os


EPOCH_METRIC_FIELDS = [
    "run_id", "epoch", "train_recon", "train_vq", "val_recon",
    "best_val_recon", "is_best", "phase", "channel_prob", "learning_rate",
]

CODEBOOK_METRIC_FIELDS = [
    "run_id", "epoch", "branch", "layer", "codebook_size", "active_ratio",
    "active_count", "dead_count", "perplexity", "min_l2_dist",
    "collapse_count", "collapse_ratio", "distance_reference_count",
    "distance_stats_exact",
]


def append_epoch_record(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_codebook_records(path, run_id, epoch, results, num_embeddings_list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CODEBOOK_METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        branches = [("src", results["src"], num_embeddings_list)]
        if "raq" in results:
            raq_sizes = [stats["usage_counts"].numel() for stats in results["raq"]]
            branches.append(("raq", results["raq"], raq_sizes))
        for branch, branch_stats, codebook_sizes in branches:
            for layer, stats in enumerate(branch_stats):
                writer.writerow({
                    "run_id": run_id,
                    "epoch": epoch,
                    "branch": branch,
                    "layer": layer,
                    "codebook_size": codebook_sizes[layer],
                    "active_ratio": f"{stats['active_ratio']:.10f}",
                    "active_count": stats["active_count"],
                    "dead_count": stats["dead_count"],
                    "perplexity": f"{stats['perplexity']:.10f}",
                    "min_l2_dist": f"{stats['min_l2_dist']:.10f}",
                    "collapse_count": stats["collapse_count"],
                    "collapse_ratio": f"{stats['collapse_ratio']:.10f}",
                    "distance_reference_count": stats["distance_reference_count"],
                    "distance_stats_exact": int(stats["distance_stats_exact"]),
                })
