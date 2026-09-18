"""Export an unsmoothed epoch-loss snapshot without touching training logs."""
import argparse
import csv
import datetime
import io
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.runtime/cache/matplotlib')
os.environ['TMPDIR'] = str(ROOT / '.runtime/tmp')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run = Path(args.run_dir).resolve()
    output = Path(args.output).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('Output must be inside RAQ-VAE-main')
    # Read once; ignore a potentially incomplete final row during live writes.
    snapshot = (run / 'csv/metrics.csv').read_text()
    snapshot = snapshot[:snapshot.rfind('\n') + 1]
    metrics = ['train_total_loss_epoch', 'train_recon_loss_epoch',
               'train_latent_loss_epoch', 'val_recon_loss',
               'val_recon_loss(src)', 'val_recon_loss(trg)']
    epochs = {}
    for row in csv.DictReader(io.StringIO(snapshot)):
        if not row.get('epoch'):
            continue
        values = {key: float(row[key]) for key in metrics if row.get(key)}
        if values:
            epochs.setdefault(int(float(row['epoch'])) + 1, {}).update(values)
    if not epochs:
        raise ValueError('No epoch metrics found')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'epoch_losses.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch'] + metrics)
        writer.writeheader()
        writer.writerows({'epoch': epoch, **values} for epoch, values in sorted(epochs.items()))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    groups = [
        [('train_total_loss_epoch', 'Total'), ('train_recon_loss_epoch', 'Reconstruction'),
         ('train_latent_loss_epoch', 'Quantization')],
        [('train_recon_loss_epoch', 'Train: source + target'),
         ('val_recon_loss', 'Validation: source + target')],
        [('val_recon_loss(src)', 'Source branch'), ('val_recon_loss(trg)', 'Target branch')],
    ]
    titles = ['Training loss (log scale)', 'Reconstruction MSE', 'Validation reconstruction MSE']
    for ax, group, title in zip(axes, groups, titles):
        for key, label in group:
            points = [(epoch, values[key]) for epoch, values in sorted(epochs.items()) if key in values]
            ax.plot([p[0] for p in points], [p[1] for p in points], label=label, linewidth=1.4)
        ax.set(title=title, xlabel='Epoch (1-based)', ylabel='Loss')
        ax.grid(True, alpha=.25)
        ax.legend(fontsize=8)
    axes[0].set_yscale('log')
    val = [(values['val_recon_loss'], epoch) for epoch, values in epochs.items() if 'val_recon_loss' in values]
    if val:
        best, best_epoch = min(val)
        axes[1].scatter([best_epoch], [best], marker='*', s=100, c='black', zorder=5)
        axes[1].annotate(f'Best: epoch {best_epoch}\nMSE={best:.6f}', (best_epoch, best),
                         xytext=(-100, 35), textcoords='offset points', fontsize=8,
                         arrowprops={'arrowstyle': '->'})
    stamp = datetime.datetime.now().astimezone().isoformat(timespec='seconds')
    fig.suptitle(f'RAQVAE_TWO / Cars196 | completed epoch metrics through {max(epochs)} | {stamp}\n'
                 'Unsmoothed epoch means; validation every 2 epochs; target K sampled randomly', fontsize=11)
    fig.savefig(output / 'loss_curves.png', dpi=170)
    fig.savefig(output / 'loss_curves.svg')
    plt.close(fig)
    print(f'CURVE {output / "loss_curves.png"}')
    print(f'LAST_EPOCH {max(epochs)}')
    if val:
        print(f'BEST_VALIDATION epoch={best_epoch} mse={best}')


if __name__ == '__main__':
    main()
