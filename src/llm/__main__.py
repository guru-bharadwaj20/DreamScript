"""Entry point: python -m src.llm [command] [args ...]

    python -m src.llm                     # the commands in this package
    python -m src.llm <command> [args]    # run one of them

Seven packages had runnable modules and no entry point, which is how a module comes to be
"imported by nothing, named in no config, no doc and no workflow" while still being the thing
that produced a report in `reports/`. Reachable by name is the difference between code that is
dead and code that is simply not on an import path.
"""

from __future__ import annotations

import sys

from src.utils.stage import dispatch

if __name__ == "__main__":
    sys.exit(dispatch("src.llm"))
