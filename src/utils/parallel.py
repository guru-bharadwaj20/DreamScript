"""Cross-cutting - running independent work across cores.

Phase 3's evaluation harnesses are embarrassingly parallel: every image is scored on its own,
and this machine has 32 cores. One helper, so that the choice of backend and the worker count
are made in one place and every harness reports the same way.

**Determinism is preserved.** Results come back in submission order, never completion order, so
a parallel run and a serial run produce byte-identical output. Anything whose result depends on
the order it finished is a bug, not a speed-up, and `tests/test_parallel.py` checks it.

The default leaves one core free. A machine pinned at 100% has no headroom for the interactive
work that is usually happening alongside a long evaluation.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Sequence
from typing import Any, TypeVar

from joblib import Parallel, delayed

T = TypeVar("T")
R = TypeVar("R")

#: Below this many items the process pool costs more to start than the work it saves.
MIN_ITEMS_FOR_PARALLEL = 8


def workers(requested: int | None = None) -> int:
    """How many processes to use: `requested`, or every core but one."""
    if requested is not None and requested > 0:
        return requested
    return max(1, (os.cpu_count() or 2) - 1)


def pmap(
    function: Callable[[T], R],
    items: Sequence[T] | Iterable[T],
    *,
    n_jobs: int | None = None,
    prefer: str = "processes",
    desc: str | None = None,
) -> list[R]:
    """`[function(x) for x in items]`, spread across cores, in the original order.

    `prefer="threads"` is the right choice for OpenCV-heavy work, which releases the GIL and
    avoids the cost of pickling images to another process.
    """
    items = list(items)
    jobs = workers(n_jobs)
    if jobs == 1 or len(items) < MIN_ITEMS_FOR_PARALLEL:
        return [function(item) for item in items]
    if desc:
        print(f"  {desc}: {len(items)} items across {jobs} workers")
    return list(Parallel(n_jobs=jobs, prefer=prefer)(delayed(function)(item) for item in items))


def pstarmap(
    function: Callable[..., R],
    argument_sets: Sequence[tuple[Any, ...]],
    *,
    n_jobs: int | None = None,
    prefer: str = "processes",
) -> list[R]:
    """As `pmap`, for a function taking several arguments."""
    argument_sets = list(argument_sets)
    jobs = workers(n_jobs)
    if jobs == 1 or len(argument_sets) < MIN_ITEMS_FOR_PARALLEL:
        return [function(*arguments) for arguments in argument_sets]
    return list(
        Parallel(n_jobs=jobs, prefer=prefer)(
            delayed(function)(*arguments) for arguments in argument_sets
        )
    )
