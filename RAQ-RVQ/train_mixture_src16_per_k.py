"""Independent RAQ-RVQ src16 training with per-K mixture sampling.

The src16 experiment intentionally reuses the proven src32 training loop. All
model sizes, target-K bounds, output paths, and resume paths come from the
src16-specific environment exported by its launcher.
"""

from train_mixture_src32_per_k import main


if __name__ == "__main__":
    main()
