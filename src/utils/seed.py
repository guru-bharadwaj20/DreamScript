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

import contextlib
import os
import random
import sys
import warnings

DEFAULT_SEED = 42


def hashseed_effective(seed: int = DEFAULT_SEED) -> bool:
    """Whether string hashing in *this* interpreter is actually pinned to `seed`.

    The distinction this function exists for: setting `PYTHONHASHSEED` in `os.environ` after the
    interpreter has started changes what child processes see and changes nothing at all about
    this one. `set_seed` did exactly that and returned as if it had worked.
    """
    return os.environ.get("PYTHONHASHSEED") == str(seed)


def ensure_hashseed(seed: int = DEFAULT_SEED) -> None:
    """Re-exec the interpreter once with PYTHONHASHSEED set, if it is not already.

    `sys.orig_argv` rather than `sys.argv`, and it is not cosmetic: under `python -m src.ingest`
    the latter is `['.../src/ingest/__main__.py', ...]`, so re-execing it runs the file as a
    script with no package context and every `from src...` import fails. `orig_argv` keeps the
    `-m src.ingest` form the caller actually typed.
    """
    if hashseed_effective(seed):
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    argv = list(getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv])
    os.execv(sys.executable, argv)


def set_seed(seed: int = DEFAULT_SEED, *, deterministic: bool = True, strict: bool = False) -> int:
    """Seed python, numpy and torch; optionally force deterministic cuDNN kernels.

    Returns the seed so callers can log it. Safe to call when torch is not installed.

    **It cannot pin string hashing and no longer implies that it can.** This used to be
    `os.environ.setdefault("PYTHONHASHSEED", ...)`, which after interpreter start affects child
    processes and nothing in this one - the fact stated in this module's own docstring, three
    lines above the code that ignored it. The assignment stays, because children do inherit it;
    what is new is that a run which is *not* hash-pinned says so instead of looking seeded.
    `ensure_hashseed` is the fix, and `utils.cli.main` calls it for every stage entry point.

    `strict` chooses what torch does when an operation has no deterministic implementation:
    raise (True) or warn and carry on (False, the default). Not a preference - several ops on
    this project's CUDA training path have no deterministic kernel, so `strict=True` refuses to
    train rather than training reproducibly. It is a parameter so a measurement that needs the
    stronger guarantee can ask for it and find out.
    """
    if not hashseed_effective(seed):
        warnings.warn(
            f"PYTHONHASHSEED is {os.environ.get('PYTHONHASHSEED')!r}, not {seed!r}: string "
            "hashing in this process is not pinned and anything ordered by it may differ "
            "between runs. Call src.utils.seed.ensure_hashseed() from the entry point.",
            RuntimeWarning,
            stacklevel=2,
        )
    # Still set, so subprocesses this run spawns inherit it even though this one cannot use it.
    os.environ["PYTHONHASHSEED"] = str(seed)
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
        with contextlib.suppress(Exception):  # older torch builds lack this switch
            torch.use_deterministic_algorithms(True, warn_only=not strict)
    else:
        torch.backends.cudnn.benchmark = True

    return seed
