"""Phase 13.2 - the typed values that pass between stages.

Every stage takes one of these and returns one of these, so a stage can be run, cached and
tested without the stage before it.

They are **pydantic dataclasses**, not `BaseModel`s: `StageReport("detect", ok=True, ...)` is
constructed positionally throughout the pipeline, which a `BaseModel` forbids, and
`pydantic.dataclasses.dataclass` gives the same validation without changing a single call site.

The validation is deliberately shallow on payloads and strict on structure. `Outcome.value` is
`Any` because what crosses that boundary is a numpy array, an IR dict or a code string, each
already validated where it is produced - `schemas/ir.schema.json` for the IR, 12.1.1's schema for
a pair - and re-describing those schemas here would restate them in a second language without
checking anything new. What *is* worth enforcing is the shape of the report itself: `seconds` and
`confidence` are floats, `ok`/`cached`/`degraded` are bools, `stages` is a list of `StageReport`.
Those are the fields the timing table and the failure message are built from, and a stage that
wrote a string into `seconds` would produce a table that sorts wrongly and a budget check that
silently compares a string to 10.0.

What this does buy is that **a stage's failure is a value, not an exception** (13.8). `Outcome`
carries either a payload or a reason, so a pipeline that cannot read a page still returns a
`Result` saying which stage stopped and why, and the caller never sees a traceback from inside
a detector.
"""

from __future__ import annotations

import time
from dataclasses import field
from typing import Any, Generic, TypeVar

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

#: Payloads crossing these boundaries are numpy arrays and IR dicts, which pydantic cannot
#: describe and does not need to; `arbitrary_types_allowed` lets them through untouched while
#: the report's own fields stay validated.
_CONFIG = ConfigDict(arbitrary_types_allowed=True)

T = TypeVar("T")

#: The five diagram types the project routes on, plus the honest "we do not know".
DIAGRAM_TYPES = ("flowchart", "state_machine", "er_diagram", "wireframe", "circuit")
UNKNOWN = "unknown"


@dataclass(slots=True, config=_CONFIG)
class Outcome(Generic[T]):
    """A stage's answer: a value, or a reason there is none. Never both."""

    value: T | None = None
    reason: str = ""
    confidence: float = 1.0
    degraded: bool = False

    @property
    def ok(self) -> bool:
        return self.value is not None

    @classmethod
    def failed(cls, reason: str) -> Outcome[T]:
        return cls(value=None, reason=reason)

    @classmethod
    def fallback(cls, value: T, reason: str, confidence: float = 0.0) -> Outcome[T]:
        """A value produced by a lower rung of 13.4's chain, marked as such."""
        return cls(value=value, reason=reason, confidence=confidence, degraded=True)


@dataclass(slots=True, config=_CONFIG)
class StageReport:
    """What one stage did, for the timing table (13.7) and the failure message (13.8)."""

    name: str
    ok: bool
    seconds: float
    cached: bool = False
    degraded: bool = False
    reason: str = ""
    confidence: float = 1.0
    detail: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "stage": self.name,
            "ok": self.ok,
            "seconds": round(self.seconds, 3),
            "cached": self.cached,
            "degraded": self.degraded,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            **self.detail,
        }


@dataclass(slots=True, config=_CONFIG)
class Result:
    """Everything one page produced, whether or not it got all the way to code."""

    source: str
    diagram_type: str = UNKNOWN
    ir: dict[str, Any] | None = None
    ir_text: str = ""
    traversal: list[str] = field(default_factory=list)
    code: str = ""
    language: str = ""
    stages: list[StageReport] = field(default_factory=list)
    needs_confirmation: bool = False
    stopped_at: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.code) and not self.stopped_at

    @property
    def seconds(self) -> float:
        return round(sum(s.seconds for s in self.stages), 3)

    def timing_table(self) -> list[dict[str, Any]]:
        """13.7's per-stage table, in execution order."""
        return [s.as_row() for s in self.stages]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "diagram_type": self.diagram_type,
            "ir": self.ir,
            "ir_text": self.ir_text,
            "traversal": self.traversal,
            "code": self.code,
            "language": self.language,
            "ok": self.ok,
            "seconds": self.seconds,
            "needs_confirmation": self.needs_confirmation,
            "stopped_at": self.stopped_at,
            "stages": self.timing_table(),
        }


class Timer:
    """Wall time for one stage, whether it returned or raised."""

    __slots__ = ("seconds", "_started")

    def __enter__(self) -> Timer:
        self._started = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *_exc: object) -> None:
        self.seconds = time.perf_counter() - self._started
