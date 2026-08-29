"""Entry point: python -m src.preprocess --config configs/preprocess.yaml [key=value ...]"""

from __future__ import annotations

import sys

from omegaconf import DictConfig

from src.utils.cli import StageNotImplemented, main
from src.utils.logging import Run


def run(cfg: DictConfig, active: Run) -> int:
    raise StageNotImplemented("stage not implemented yet; see plan.md for the owning phase")


if __name__ == "__main__":
    sys.exit(main("src.preprocess", run))
