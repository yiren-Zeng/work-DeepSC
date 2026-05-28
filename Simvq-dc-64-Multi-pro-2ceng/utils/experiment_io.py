import csv
import os


EPOCH_METRIC_FIELDS = [
    "run_id", "epoch", "train_recon", "train_vq", "val_recon",
    "best_val_recon", "is_best", "phase", "channel_prob", "learning_rate",
]


def append_epoch_record(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EPOCH_METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
