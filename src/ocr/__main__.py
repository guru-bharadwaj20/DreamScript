"""Entry point: python -m src.ocr [command] [args ...]

    python -m src.ocr                     # the commands in this package
    python -m src.ocr <command> [args]    # run one of them
    python -m src.ocr --print-config      # the Phase 0.2.3 config contract, unchanged

This used to be a `run()` that raised `StageNotImplemented`, so the command printed a resolved
config and exited 2 while the package's own modules - each with a working `main()` - were
reachable only by knowing their names. `src.utils.stage` routes to them; the config contract is
untouched and still reached by any first argument beginning with `-`.
"""

from __future__ import annotations

import sys

from src.utils.stage import dispatch

if __name__ == "__main__":
    sys.exit(dispatch("src.ocr"))
