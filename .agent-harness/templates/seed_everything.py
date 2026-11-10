"""Central reproducibility helper. Call once at every entry point."""
import os
import random

import numpy as np


def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    return seed


def seed_worker(worker_id: int) -> None:
    """Pass as DataLoader(worker_init_fn=seed_worker, generator=g)."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
