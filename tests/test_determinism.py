"""Phase 0.1.6 acceptance test — seeding produces identical results across runs."""

from __future__ import annotations

import random

import numpy as np
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
