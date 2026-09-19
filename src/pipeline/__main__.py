"""`python -m src.pipeline ...` - see `src.pipeline.cli`."""

import sys

from src.pipeline.cli import run

if __name__ == "__main__":
    sys.exit(run())
