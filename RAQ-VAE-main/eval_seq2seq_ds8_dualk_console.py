"""Print dual-K channel trials and their mean without saving test artifacts."""

import argparse
import contextlib
from fractions import Fraction
import io
import math
import os
import sys

sys.dont_write_bytecode = True


@contextlib.contextmanager
def quiet_evaluation():
    """Keep dependency chatter in memory; replay it if evaluation fails.

    TensorFlow/absl writes directly to fd 2, bypassing Python's redirect_stderr.
    Linux memfd captures those messages in RAM without creating a log file.
    Restore stderr before exceptions propagate so tracebacks remain visible.
    """
    stdout = io.StringIO()
    with os.fdopen(os.memfd_create("raq-console-stderr", os.MFD_CLOEXEC), "w+b") as stderr:
        sys.stderr.flush()
        original_stderr = os.dup(2)
        failed = False
        try:
            os.dup2(stderr.fileno(), 2)
            with contextlib.redirect_stdout(stdout):
                yield
        except BaseException:
            failed = True
            raise
        finally:
            sys.stderr.flush()
            os.dup2(original_stderr, 2)
            os.close(original_stderr)
            if failed:
                stderr.seek(0)
                sys.stderr.write(stdout.getvalue())
                sys.stderr.write(stderr.read().decode("utf-8", errors="replace"))
                sys.stderr.flush()


def run_trials(evaluator, evaluator_args, seeds, compatibility_signature):
    """Run each seed with the original evaluator and aggregate full-precision metrics."""
    if any(flag in evaluator_args for flag in ("--help", "-h", "--dry-run")):
        evaluator.main([*evaluator_args, "--console-only", "--seed", str(seeds[0])])
        return
    trials = []
    signature = None
    for seed in seeds:
        with quiet_evaluation():
            report = evaluator.main([
                *evaluator_args, "--console-only", "--seed", str(seed),
            ])
        current_signature = compatibility_signature(report)
        if signature is None:
            signature = current_signature
            meta = report["metadata"]
            ratio = report["transmission_ratio"]
            rate = Fraction(meta["ldpc"]["rate"]).limit_denominator()
            cscr = "on" if meta["cscr"]["enabled"] else "off"
            print(
                f"配置：K_bottom={meta['target_K_bottom']} | K_top={meta['target_K_top']} | "
                f"LDPC {rate} | {meta['modulation'].upper()} | "
                f"{evaluator.channel_label(meta['channel'], meta['rician_k_factor'])} | "
                f"SNR={meta['snr_db']:g} dB | CSCR={cscr}\n"
                f"实际传输比：{ratio:.8f}（约 1/{1 / ratio:.4f}）",
                flush=True,
            )
        elif current_signature != signature:
            raise ValueError(f"Incompatible trial configuration for seed {seed}")
        metrics = report["metrics"]
        trials.append((metrics["psnr"], metrics["ms_ssim"]))
        print(
            f"seed={seed} | PSNR：{metrics['psnr']:.4f} dB | "
            f"MS-SSIM：{evaluator.format_ms_ssim(metrics['ms_ssim'])}",
            flush=True,
        )
    mean_psnr = math.fsum(trial[0] for trial in trials) / len(trials)
    mean_ms_ssim = math.fsum(trial[1] for trial in trials) / len(trials)
    print(
        f"平均 | PSNR：{mean_psnr:.4f} dB | "
        f"MS-SSIM：{evaluator.format_ms_ssim(mean_ms_ssim)}",
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--channel-seeds", type=int, nargs="+", default=[42])
    args, evaluator_args = parser.parse_known_args(argv)
    with quiet_evaluation():
        import eval_seq2seq_ds8_dualk_channel as entry
        from aggregate_seq2seq_channel_trials import (
            _compatibility_signature, normalize_channel_seeds,
        )
    try:
        seeds = normalize_channel_seeds(args.channel_seeds)
    except ValueError as error:
        parser.error(str(error))
    run_trials(entry.evaluator, evaluator_args, seeds, _compatibility_signature)


if __name__ == "__main__":
    main()
