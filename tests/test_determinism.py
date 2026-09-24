"""Phase 0.1.6 acceptance test — seeding produces identical results across runs."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.seed import set_seed


def test_set_seed_repeats_python_numpy_torch():
    def sample():
        set_seed(1234, deterministic=True)
        return (
            random.random(),
            float(np.random.rand()),
            float(torch.randn(3).sum()),
        )

    assert sample() == sample()


def test_different_seeds_differ():
    set_seed(1)
    a = float(torch.randn(8).sum())
    set_seed(2)
    b = float(torch.randn(8).sum())
    assert a != b


def test_deterministic_flags_are_set():
    set_seed(7, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_training_run_is_reproducible():
    """Two identical short training runs must reach exactly the same loss."""
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def train() -> float:
        set_seed(99, deterministic=True)
        x = torch.randn(128, 16, device=device)
        y = (x.sum(dim=1, keepdim=True) > 0).float()
        model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(30):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        return float(loss.item())

    assert train() == train()


# --- the pin that only the checker ever applied (audit 13) ------------------------------------


def test_set_seed_says_so_when_string_hashing_is_not_pinned(monkeypatch, recwarn):
    """It used to `os.environ.setdefault("PYTHONHASHSEED", ...)` and return, which after
    interpreter start changes what children see and nothing about this process - the fact this
    module's own docstring states three lines above the code that ignored it."""
    from src.utils.seed import hashseed_effective, set_seed

    monkeypatch.setenv("PYTHONHASHSEED", "random")
    assert not hashseed_effective(1234)
    set_seed(1234)
    assert any(issubclass(w.category, RuntimeWarning) for w in recwarn)

    monkeypatch.setenv("PYTHONHASHSEED", "1234")
    assert hashseed_effective(1234)


def test_set_seed_still_exports_the_value_for_children(monkeypatch):
    from src.utils.seed import set_seed

    monkeypatch.setenv("PYTHONHASHSEED", "7")
    set_seed(1234)
    assert os.environ["PYTHONHASHSEED"] == "1234"


def test_ensure_hashseed_returns_without_execing_when_already_pinned(monkeypatch):
    import src.utils.seed as seed_module

    monkeypatch.setenv("PYTHONHASHSEED", "42")
    monkeypatch.setattr(
        seed_module.os, "execv", lambda *a: pytest.fail("re-execed a pinned interpreter")
    )
    seed_module.ensure_hashseed(42)


def test_ensure_hashseed_re_execs_the_command_that_was_typed(monkeypatch):
    """`sys.argv` under `python -m src.ingest` is the path to `__main__.py`, and re-execing that
    runs it as a script with no package context, so every `from src...` import fails.
    `sys.orig_argv` keeps the `-m` form."""
    import src.utils.seed as seed_module

    captured: list[list[str]] = []
    monkeypatch.setenv("PYTHONHASHSEED", "random")
    monkeypatch.setattr(sys, "orig_argv", [sys.executable, "-m", "src.ingest", "--print-config"])
    monkeypatch.setattr(seed_module.os, "execv", lambda _exe, argv: captured.append(list(argv)))
    seed_module.ensure_hashseed(42)
    assert captured == [[sys.executable, "-m", "src.ingest", "--print-config"]]
    assert os.environ["PYTHONHASHSEED"] == "42"


def test_the_entry_point_contract_pins_the_hash_seed():
    """`ensure_hashseed` had exactly two references and both were in
    `scripts/determinism_check.py`, so the one process that checked determinism was the only
    process that had it."""
    import inspect

    from src.utils import cli

    source = inspect.getsource(cli.main)
    assert "ensure_hashseed(seed)" in source
    assert source.index("ensure_hashseed(seed)") < source.index("set_seed(seed")


def test_strict_chooses_whether_a_nondeterministic_op_raises_or_warns():
    import inspect

    from src.utils.seed import set_seed

    assert "strict" in inspect.signature(set_seed).parameters
    assert "warn_only=not strict" in inspect.getsource(set_seed)


def test_set_seeds_global_switches_do_not_leak_into_the_next_test():
    """The flags `set_seed(deterministic=True)` sets are process-global, and one test setting
    them broke another ten modules later: CUDA graph capture cannot run under deterministic
    algorithms, so `test_captured_update_is_bit_identical_to_the_eager_update` passed alone and
    failed in the suite with "operation failed due to a previous error during capture". The
    autouse fixture in conftest snapshots and restores them; this asserts it is still there."""
    source = (Path(__file__).parent / "conftest.py").read_text(encoding="utf-8")
    assert "_torch_global_flags_are_restored" in source
    assert "use_deterministic_algorithms(before[2]" in source
