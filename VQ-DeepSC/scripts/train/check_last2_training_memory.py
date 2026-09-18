"""Probe full-size training without loading or saving experiment checkpoints."""

import argparse
import json
import os
from pathlib import Path
import sys


def main(finish_barrier=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--codebooks", type=int, nargs=2, required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--min-free-mib", type=int, default=3072)
    parser.add_argument("--max-reserved-mib", type=int, default=None)
    args = parser.parse_args()
    if args.gpu < 0 or args.steps < 2 or args.min_free_mib < 1:
        parser.error("GPU must be nonnegative; steps >= 2; min-free-mib > 0")
    if args.max_reserved_mib is not None and args.max_reserved_mib < 1:
        parser.error("max-reserved-mib must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["VQ_DEEPSC_DEVICE"] = "cuda:0"
    os.environ["VQ_DEEPSC_NUM_EMBEDDINGS_LIST"] = "64,64," + ",".join(
        str(k) for k in args.codebooks
    )
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    import torch

    from config import Config
    from data.datasets import get_dataloader
    from losses.deepsc_loss import DeepSCLoss
    from train_last2_cascade import build_model, setup_seed

    cfg = Config()
    cfg.validate()
    setup_seed(42)
    device = torch.device("cuda:0")
    mib = 1024**2
    initial_free, total = torch.cuda.mem_get_info(device)
    # Leave room for existing jobs and non-PyTorch CUDA allocations.
    budget = initial_free - (args.min_free_mib + 512) * mib
    if args.max_reserved_mib is not None:
        budget = min(budget, args.max_reserved_mib * mib)
    if budget <= 0:
        raise RuntimeError("Insufficient free GPU memory for a safe probe")
    torch.cuda.set_per_process_memory_fraction(budget / total, device)
    print(json.dumps({"phase": "start", "gpu": args.gpu,
                      "codebooks": args.codebooks, "batch_size": cfg.MICRO_BATCH_SIZE,
                      "initial_free_mib": initial_free // mib,
                      "allocator_budget_mib": budget // mib}), flush=True)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, betas=cfg.BETAS)
    loss_fn = DeepSCLoss().to(device)
    loader = get_dataloader(cfg.TRAIN_DATASET_PATH, cfg.MICRO_BATCH_SIZE,
                            shuffle=True, mode="train", num_workers=cfg.NUM_WORKERS,
                            pin_memory=cfg.PIN_MEMORY)
    batches = iter(loader)
    min_free = initial_free
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for step in range(args.steps):
        images = next(batches).to(device, non_blocking=True)
        if images.shape[0] != cfg.MICRO_BATCH_SIZE:
            raise RuntimeError("Probe requires a full training batch")
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        reconstruction_loss, vq_loss = loss_fn(
            images, output["reconstructed_images"], output["vq_losses"]
        )
        (reconstruction_loss + vq_loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        min_free = min(min_free, torch.cuda.mem_get_info(device)[0])
        print(json.dumps({"phase": "train", "step": step + 1,
                          "reconstruction_loss": reconstruction_loss.item(),
                          "vq_loss": vq_loss.item(), "free_mib": min_free // mib}),
              flush=True)

    model.eval()
    with torch.no_grad():
        validation_output = model(images)
        loss_fn(images, validation_output["reconstructed_images"],
                validation_output["vq_losses"])
    torch.cuda.synchronize(device)
    if finish_barrier is not None:
        # Both models, Adam states and allocator caches stay resident here.
        finish_barrier.wait(timeout=60)
    min_free = min(min_free, torch.cuda.mem_get_info(device)[0])
    result = {"phase": "complete", "gpu": args.gpu, "codebooks": args.codebooks,
              "batch_size": cfg.MICRO_BATCH_SIZE,
              "peak_allocated_mib": torch.cuda.max_memory_allocated(device) // mib,
              "peak_reserved_mib": torch.cuda.max_memory_reserved(device) // mib,
              "min_observed_free_mib": min_free // mib,
              "passed": min_free >= args.min_free_mib * mib}
    if finish_barrier is not None:
        # Do not release either model until both sampled combined GPU usage.
        finish_barrier.wait(timeout=60)
    print(json.dumps(result), flush=True)
    if not result["passed"]:
        raise RuntimeError("GPU free memory fell below the required safety margin")


if __name__ == "__main__":
    main()
