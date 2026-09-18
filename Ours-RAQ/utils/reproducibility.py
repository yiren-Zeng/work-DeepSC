import os
import random
import numpy as np
import torch


def setup_seed(seed=42, deterministic=True):
    """
    Set random seeds for reproducibility.
    Note:
    - For PYTHONHASHSEED and CUBLAS_WORKSPACE_CONFIG, it is better to set them
      before launching Python in the shell script.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    print(f"[Info] Random seed set to {seed}, deterministic={deterministic}")


def seed_worker(_worker_id):
    """
    Seed each DataLoader worker for full reproducibility.

    Intended as ``worker_init_fn`` for ``torch.utils.data.DataLoader``.
    The worker id is passed automatically by the DataLoader; the actual
    per-worker seed is derived from ``torch.initial_seed()``, which PyTorch
    sets based on the main-process seed + worker id.

    Args:
        _worker_id: 0-based worker index assigned by DataLoader
                    (unused directly; seed is derived via torch.initial_seed).
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _reset_eval_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)