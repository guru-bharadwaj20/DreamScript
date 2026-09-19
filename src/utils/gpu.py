"""Make this card's failure mode an exception instead of a silent four-times slowdown.

    from src.utils import gpu
    gpu.cap()            # before the first allocation in any process that loads a model

## The problem this exists for

The card is 24 GiB and shared, and it runs under Windows' WDDM driver model. **WDDM does not
raise when a process asks for more VRAM than is free - it pages the excess into host RAM and
keeps going.** Nothing fails. `nvidia-smi` keeps reporting 100% utilisation, because the SMs are
busy waiting on PCIe, and the only visible symptom is power draw: a healthy run on this card
sits at 180-195 W, and a spilling one at 60-95 W while producing a quarter of the work.

That has now cost this project two long runs. `src.ocr.ownlearn` reserved 23.19 GiB and sat at
24.06 GiB drawing 60-99 W for over an hour, and a second attempt reached 21.9 GiB and oscillated
between 57 and 173 W. Both were *slower* than the smaller configuration, and neither raised
anything. A budget computed in advance - `ownlearn.plan_workers` does compute one - is not
enough on its own, because what overruns it is the caching allocator growing across phases, not
the batch that was planned.

## What this does instead

`torch.cuda.set_per_process_memory_fraction` makes the PyTorch allocator refuse to grow past a
share of the card. Past the cap the process raises `torch.cuda.OutOfMemoryError`, which is a
stack trace naming the line that asked for too much - something a log shows and a person can
act on - rather than an hour of quiet paging.

The cap is a share of the *total* card rather than of what is free, deliberately: the fraction
has to be decided before anything is allocated, other tenants come and go while a long job runs,
and a cap that moved with them would not be a cap.

**This does not make the process a good neighbour, and is not meant to.** Other accounts share
this card, and the rule that matters there is to wait for them rather than crowd them. This only
guarantees that when this process is wrong about its own size, it says so.
"""

from __future__ import annotations

import os

#: Share of the 24 GiB card one process may reserve. 20 GiB, leaving the margin WDDM needs
#: before it starts paging - the same cap `ownlearn.plan_workers` budgets against.
FRACTION = 20.0 / 24.0


def cap(fraction: float = FRACTION) -> bool:
    """Bound this process's VRAM. Returns whether a cap was actually applied.

    Safe to call more than once and safe to call with no CUDA device, so it can sit at the top
    of any entry point that might load a model without the caller testing for a card first.
    """
    try:
        import torch
    except ImportError:
        return False
    if not torch.cuda.is_available():
        return False

    # Segments that can grow and shrink in place fragment far less across a job whose phases
    # allocate differently - which is exactly the shape that overran the planned budget here,
    # a recognition pass followed by a readability pass with a different batch.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    try:
        torch.cuda.set_per_process_memory_fraction(float(fraction))
    except (RuntimeError, ValueError):
        return False
    return True


def headroom_gib() -> float:
    """Free VRAM on the card right now, in GiB. 0.0 when there is no card."""
    try:
        import torch
    except ImportError:
        return 0.0
    if not torch.cuda.is_available():
        return 0.0
    free, _total = torch.cuda.mem_get_info()
    return float(free) / 2**30
