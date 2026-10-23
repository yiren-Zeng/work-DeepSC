"""Evaluate the scale-0-only model through the existing digital link."""

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from fractions import Fraction
import os
from pathlib import Path
import sys

from config_first_branch import FirstBranchConfig
from evaluation.first_branch_images import evaluate_with_images


@contextmanager
def _quiet_diagnostics():
    """Silence Python and native-library output; restore streams before errors escape."""
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            with redirect_stdout(sink), redirect_stderr(sink):
                yield
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _print_summary(results, cfg, arguments):
    if not results:
        return
    codebooks = ",".join(map(str, cfg.NUM_EMBEDDINGS_LIST))
    if arguments.no_channel:
        print(f"K={codebooks} | No channel")
    else:
        diagnostics = next(iter(results.values()))["diagnostics"]
        stream = diagnostics["combined_stream"]
        rate = Fraction(diagnostics["ldpc"]["k"], diagnostics["ldpc"]["n"])
        cbr = Fraction(stream["channel_symbols"], stream["source_values"])
        print(
            f"K={codebooks} | {arguments.modulation.upper()} | LDPC={rate} | "
            f"CBR={cbr} ({float(cbr):.6f})"
        )
    for snr, result in results.items():
        prefix = "No channel" if snr == "no_channel" else f"SNR={snr} dB"
        print(
            f"{prefix} | PSNR={result['psnr']:.4f} dB | "
            f"MS-SSIM={result['ms_ssim']:.4f}"
        )


def main(argv=None):
    cfg = FirstBranchConfig()
    parser = argparse.ArgumentParser(
        description="Evaluate scale 0 (K=32) on Kodak through LDPC + AWGN."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--snrs", type=int, nargs="+", default=[6])
    parser.add_argument("--no-channel", action="store_true")
    parser.add_argument(
        "--image-output-dir",
        default=str(Path(__file__).resolve().parent / "generated_images" / "k32_first_branch"),
        help="Root directory for reconstruction PNGs, grouped by link settings.",
    )
    parser.add_argument(
        "--modulation", choices=["bpsk", "qpsk", "16qam"], default="bpsk"
    )
    parser.add_argument("--ldpc_n", type=int, default=256)
    parser.add_argument("--ldpc_k", type=float, default=0.5)
    parser.add_argument("--stream-packing", choices=["combined"], default="combined")
    arguments = parser.parse_args(argv)
    with _quiet_diagnostics():
        results = evaluate_with_images(
            checkpoint_path=arguments.checkpoint,
            test_snrs=arguments.snrs,
            image_output_dir=arguments.image_output_dir,
            no_channel=arguments.no_channel,
            modulation=arguments.modulation,
            ldpc_n=arguments.ldpc_n,
            ldpc_rate=arguments.ldpc_k,
            stream_packing=arguments.stream_packing,
            cfg=cfg,
        )
    _print_summary(results, cfg, arguments)
    return results


if __name__ == "__main__":
    main()
