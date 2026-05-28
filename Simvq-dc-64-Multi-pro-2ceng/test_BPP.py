"""Compatibility entry point for the old BPP test command."""

from evaluate_bpp import evaluate_bpp


test_bpp = evaluate_bpp


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calculate BPP for a SimVQ checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path; defaults to the best model.")
    args = parser.parse_args()
    evaluate_bpp(args.checkpoint)
