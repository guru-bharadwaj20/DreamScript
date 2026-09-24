"""Phase 0.2.3 — the shared entry-point contract.

Every pipeline package exposes the same command line:

    python -m src.<package> --config configs/<package>.yaml [key.path=value ...]

`main()` below parses those arguments, loads and composes the config, seeds every RNG from
it, opens a run directory under `experiments/`, and hands both to the package's
`run(cfg, run)`. Packages whose stage is not implemented
yet raise `StageNotImplemented`, which prints the resolved config and exits non-zero — so the
contract is testable from Phase 0 onward instead of after the stage is written.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable

from omegaconf import DictConfig, OmegaConf

from src.utils.config import add_config_args, load_config
from src.utils.logging import Run, start_run
from src.utils.seed import ensure_hashseed, set_seed


class StageNotImplemented(NotImplementedError):
    """Raised by a package whose pipeline stage has not been built yet."""


def main(
    package: str,
    run: Callable[[DictConfig, Run], int | None],
    argv: list[str] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog=f"python -m {package}",
        description=f"{package} stage of the DreamScript pipeline",
    )
    add_config_args(parser, default=f"configs/{package.split('.')[-1]}.yaml")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="resolve the config, print it, and exit without running the stage",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed = int(cfg.get("seed", 42))
    # Before `set_seed`, and it may not return: `PYTHONHASHSEED` is read once at interpreter
    # start, so pinning it means re-execing. Every stage entry point goes through this function,
    # and until now none of them pinned it - `ensure_hashseed` existed and its only two callers
    # were both inside `scripts/determinism_check.py`, so the one process that checked
    # determinism was the only process that had it.
    ensure_hashseed(seed)
    set_seed(seed, deterministic=cfg.get("deterministic", True))

    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0

    active = start_run(cfg)
    try:
        code = int(run(cfg, active) or 0)
        active.finish("ok" if code == 0 else f"exit-{code}")
        return code
    except StageNotImplemented as exc:
        active.log.warning("%s: %s", package, exc)
        # Discarded, not finished. A stage that refuses on its first line produced nothing, and
        # writing a timestamped directory for it on every invocation is how `experiments/` filled
        # with runs that recorded only that nothing ran.
        active.discard()
        print(f"config resolved from {cfg._config_path}:", file=sys.stderr)
        print(OmegaConf.to_yaml(cfg, resolve=True), file=sys.stderr)
        return 2
    except Exception:
        active.log.exception("%s failed", package)
        active.finish("error")
        raise
