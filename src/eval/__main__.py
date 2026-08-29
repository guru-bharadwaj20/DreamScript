"""Entry point: python -m src.eval --config configs/eval.yaml [key=value ...]"""

from __future__ import annotations

import sys

from omegaconf import DictConfig

from src.utils.cli import StageNotImplemented, main


def run(cfg: DictConfig) -> int:
    raise StageNotImplemented("stage not implemented yet; see plan.md for the owning phase")


if __name__ == "__main__":
    sys.exit(main("src.eval", run))
