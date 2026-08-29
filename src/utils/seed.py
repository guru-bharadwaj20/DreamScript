"""Phase 0.1.6 — determinism harness.

Every experiment in DreamScript is comparable only if repeated runs of the same script
produce byte-identical metrics. This module is the single place that pins randomness.

    from src.utils.seed import set_seed
    set_seed(42, deterministic=True)

`PYTHONHASHSEED` must be set *before* the interpreter starts to affect string hashing, so
`set_seed` re-execs the process once if it is unset. Call it as the first statement in an
entry point, before importing anything that samples randomness at import time.
"""

from __future__ import annotations

import os
import random
import sys

DEFAULT_SEED = 42


def ensure_hashseed(seed: int = DEFAULT_SEED) -> None:
    """Re-exec the interpreter once with PYTHONHASHSEED set, if it is not already."""
    if os.environ.get("PYTHONHASHSEED") == str(seed):
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.execv(sys.executable, [sys.executable] + sys.argv)


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True) -> int:
    """Seed python, numpy and torch; optionally force deterministic cuDNN kernels.

    Returns the seed so callers can log it. Safe to call when torch is not installed.
    """
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy is a hard dep, but keep the util importable regardless
        pass

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # cuBLAS needs this workspace config for deterministic matmuls on CUDA >= 10.2.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # noqa: BLE001 - older torch builds
            pass
    else:
        torch.backends.cudnn.benchmark = True

    return seed


def seed_worker(worker_id: int) -> None:
    """DataLoader `worker_init_fn` so each worker is seeded reproducibly."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    try:
        import numpy as np

        np.random.seed(worker_seed)
    except ImportError:
        pass
