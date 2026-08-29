"""Phase 0.1.6 acceptance check — two runs of one script give identical metrics.

Trains a small torch model on synthetic data (CPU and, when available, CUDA), plus a
numpy/random/hash sample, and prints a metrics dict. Running it twice must produce the
exact same numbers; `--compare` runs it twice in-process and asserts that.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.seed import ensure_hashseed, set_seed  # noqa: E402


def run_workload(seed: int = 42) -> dict[str, object]:
    set_seed(seed, deterministic=True)

    import random

    import numpy as np

    metrics: dict[str, object] = {
        "python_random": random.random(),
        "numpy_random": float(np.random.rand()),
        "hash_of_str": hash("dreamscript"),
    }

    import torch
    import torch.nn as nn

    def train(device: str) -> float:
        set_seed(seed, deterministic=True)
        x = torch.randn(256, 32, device=device)
        y = (x.sum(dim=1, keepdim=True) > 0).float()
        model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 1)).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        return float(loss.item())

    metrics["cpu_final_loss"] = train("cpu")
    if torch.cuda.is_available():
        metrics["cuda_final_loss"] = train("cuda")
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--compare",
        action="store_true",
        help="run this script twice as a subprocess and assert identical output",
    )
    args = ap.parse_args()

    # PYTHONHASHSEED is read once at interpreter start, so re-exec ourselves if it is unset.
    # Without this, str hashing (and anything ordered by it) differs between runs.
    ensure_hashseed(args.seed)

    if args.compare:
        cmd = [sys.executable, __file__, "--seed", str(args.seed)]
        first = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        second = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        print("run 1:", first.strip())
        print("run 2:", second.strip())
        same = first == second
        print("-" * 60)
        print("RESULT:", "identical - deterministic" if same else "DIFFERENT - not deterministic")
        return 0 if same else 1

    print(json.dumps(run_workload(args.seed), sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
