"""Phase 0.3.3 — the experiment naming convention, enforced in code.

Run names follow `<phase>-<model>-<variant>-<seed>`:

    p5-knn-k7-s42          Phase 5, KNN, k=7, seed 42
    p9-yolov8-1024px-s42   Phase 9, YOLOv8 detector, 1024px input, seed 42
    p12-qwen7b-lora32-s1   Phase 12, Qwen 7B base, LoRA rank 32, seed 1

Prose in `docs/conventions.md` explains why; this module is what makes it checkable, so a
malformed name fails a test instead of quietly producing an unfindable run directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# phase: p<number> with an optional sub-phase, e.g. p7 or p7.3
# model/variant: lowercase alphanumerics, may contain internal dots (e.g. 0.5x)
# seed: s<number>
RUN_NAME_RE = re.compile(
    r"^p(?P<phase>\d+(?:\.\d+)?)"
    r"-(?P<model>[a-z0-9]+(?:[a-z0-9.]*[a-z0-9])?)"
    r"-(?P<variant>[a-z0-9]+(?:[a-z0-9.]*[a-z0-9])?)"
    r"-s(?P<seed>\d+)$"
)

RUN_DIR_RE = re.compile(r"^(?P<stamp>\d{8}-\d{6}(?:-\d+)?)_(?P<name>.+)$")


class NamingError(ValueError):
    """Raised when a run name does not follow the convention."""


@dataclass(frozen=True)
class RunName:
    phase: str
    model: str
    variant: str
    seed: int

    def __str__(self) -> str:
        return f"p{self.phase}-{self.model}-{self.variant}-s{self.seed}"


def make_run_name(phase: str | float, model: str, variant: str, seed: int) -> str:
    """Build a conventional run name, validating each part."""
    name = f"p{phase}-{model.lower()}-{variant.lower()}-s{int(seed)}"
    parse_run_name(name)  # raises if the pieces do not fit the convention
    return name


def parse_run_name(name: str) -> RunName:
    m = RUN_NAME_RE.match(name)
    if not m:
        raise NamingError(
            f"run name {name!r} does not match <phase>-<model>-<variant>-<seed>, "
            "e.g. 'p5-knn-k7-s42' (see docs/conventions.md)"
        )
    return RunName(phase=m["phase"], model=m["model"], variant=m["variant"], seed=int(m["seed"]))


def is_valid_run_name(name: str) -> bool:
    return RUN_NAME_RE.match(name) is not None


def parse_run_dir(dirname: str) -> tuple[str, str]:
    """Split `20260829-090444_p5-knn-k7-s42` into its timestamp and run name."""
    m = RUN_DIR_RE.match(dirname)
    if not m:
        raise NamingError(f"{dirname!r} is not a <timestamp>_<run_name> directory")
    return m["stamp"], m["name"]
