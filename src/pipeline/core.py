"""Phase 13.1 / 13.5 / 13.7 / 13.8 - the pipeline object, its gate, its clock and its failures.

    from src.pipeline import DreamScriptPipeline
    result = DreamScriptPipeline().run("page.png")
    result.code, result.diagram_type, result.timing_table()

## One entry point, eight stages

    load       the page's pixels, and its identity for the cache
    detect     9.1's YOLO over the page - boxes and their classes
    classify   13.3's prior over those classes -> diagram type and a probability
    assemble   10.1's tracer -> an IR `Diagram`
    traverse   7.3.3's reading-order DFS -> the node order the emitter and the model see
    serialise  12.1.2's compact IR text
    generate   12.2's fine-tuned adapter, or 12.1.6's emitter when no model is served
    verify     12.3.1's parse check on whatever came back

Each returns an `Outcome`, so a stage that cannot answer returns a reason instead of raising
(13.8). `run` stops at the first stage with no value and reports `stopped_at`, and the stages
that did run keep their timings, so a failure is still a measurement.

## The gate (13.5)

`confidence_floor` is the probability below which the pipeline refuses to guess a diagram type.
Below it the result comes back with `needs_confirmation=True` and no code, because every stage
after routing is *type-specific*: emitting Python for what is actually an ER diagram produces a
confident, runnable, wrong answer, which is worse than asking. The floor defaults to 0.60 and is
a constructor argument rather than a constant because the right value depends on whether a human
is present to answer.

## The clock (13.7)

Every stage is timed and the table is on the result. The plan's budget is < 10 s end to end on
GPU; what that costs is reported per stage rather than as a single number, because the two
expensive stages - the detector and the language model - are the two a deployment would move.

## The observer (16.1.2)

    def watch(event): print(event["event"], event.get("stage"))
    DreamScriptPipeline().run("page.png", observer=watch)

The same table, streamed as it is built rather than handed over at the end. It exists because of
where this pipeline is now used: a phone, over a mobile link, on a page that takes seconds. A
spinner for three seconds and a spinner for a stalled request look identical, and 16.1.2's whole
argument is that the wait has to be legible.

It is an *observation*, not a hook. The observer cannot change the run, its return value is
ignored, and an exception from it is swallowed - because the caller on the other end of it is a
socket belonging to a phone that can go into a lift, and a broken progress stream must never be
the reason a page is lost. `stage_finished` carries `StageReport.as_row()` in full, so the stream
and `timing_table()` cannot tell the client two different stories.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.pipeline.cache import StageCache, digest_obj, env_key, image_key
from src.pipeline.contracts import UNKNOWN, Outcome, Result, StageReport, Timer

#: Where a served adapter answers, when one is running. 13.4's chain is `model -> emitter` and
#: the model rung had **no caller in the repo**: `generate.emit` took a `model_url` nobody passed
#: and `_from_model` was forty lines nothing could reach, so the chain was a chain of one and
#: every measurement of it measured the emitter.
#:
#: The environment rather than only a constructor argument, because the whole point is that a
#: deployment decides this: the container that has llama.cpp beside it sets it, the CI image that
#: does not leaves it unset and gets the emitter. It is in `cache.ENV_KEYS` for the obvious
#: reason - it changes what the generate stage returns.
MODEL_URL_ENV = "DREAMSCRIPT_MODEL_URL"

#: Below this routing probability the pipeline asks rather than guesses (13.5).
CONFIDENCE_FLOOR = 0.60

#: Target for the whole run, from the plan's 13.7 line.
LATENCY_BUDGET_S = 10.0

#: The stages `run` executes, in order, for a page that gets all the way to code.
#:
#: Named here for 16.1.2 rather than left implicit in the body of `run`. A phone waiting three
#: seconds needs to show *seven greyed-out rows that fill in*, not a spinner, and it cannot draw
#: rows it has to guess: an observer that reported only "stage 4 finished" would make the client
#: hardcode the list and drift from this module the first time a stage is added. `load` is not in
#: the list because it only ever appears as a `StageReport` when it fails - a page whose pixels
#: cannot be read never reaches stage one.
STAGES: tuple[str, ...] = (
    "detect",
    "classify",
    "assemble",
    "traverse",
    "serialise",
    "generate",
    "verify",
)

#: A progress observer (16.1.2): called with one event dict per notification, return value ignored.
#:
#: The events are `run_started` (carrying `stages`, so the client draws the list rather than
#: guessing it), `stage_started`, `stage_finished` (carrying `StageReport.as_row()` entire, so the
#: stream and the final table cannot disagree) and `run_finished`. Deliberately plain dicts: every
#: consumer of this is about to serialise them as JSON over a socket, and a dataclass here would be
#: converted at every call site.
Observer = Callable[[dict[str, Any]], None]


class DreamScriptPipeline:
    """Phases 9 through 12 composed into one call."""

    def __init__(
        self,
        *,
        cache: StageCache | None = None,
        confidence_floor: float = CONFIDENCE_FLOOR,
        generate: bool = True,
        read_text: bool = False,
        model_url: str | None = None,
    ) -> None:
        self.cache = cache if cache is not None else StageCache()
        self.confidence_floor = confidence_floor
        self.generate_code = generate
        self.read_text = read_text
        self.model_url = (model_url or os.environ.get(MODEL_URL_ENV, "")).strip() or None
        self.config_key = digest_obj(
            {
                "confidence_floor": confidence_floor,
                "generate": generate,
                "read_text": read_text,
                # Which rung answered is part of the answer, so it is part of the key.
                "model_url": self.model_url,
                # The three constructor arguments were the whole key, and they are not the whole
                # configuration: `assemble.s5` takes three switches from the environment and
                # three more environment variables name the checkpoints that produce the text
                # and the edges. Flipping one of those and re-running returned the previous
                # answer from a warm cache. See `cache.ENV_KEYS`.
                "env": env_key(),
            }
        )

    # -- the driver ---------------------------------------------------------------------

    def run(self, image: str | Path, *, observer: Observer | None = None) -> Result:
        """One page, all the way to code, or as far as it gets.

        `observer`, added by 16.1.2, is called as each stage starts and finishes so a caller can
        stream progress instead of waiting for the table at the end. It changes nothing about what
        `run` returns or how long it takes: the timing table is still assembled the same way, and a
        run with no observer takes the same path it always did.

        It is an argument rather than a constructor option because a *pipeline* is long-lived and
        shared - the model server builds one and reuses it - while an observer belongs to one
        request. Storing it on the instance would leak one caller's stream into the next caller's
        run, which under any concurrency at all is the wrong answer sent to the wrong phone.
        """
        path = Path(image)
        result = Result(source=str(path))
        self._notify(
            observer, {"event": "run_started", "stages": list(STAGES), "source": str(path)}
        )
        try:
            key = image_key(path)
        except OSError as error:
            report = StageReport("load", ok=False, seconds=0.0, reason=f"unreadable: {error}")
            result.stages.append(report)
            result.stopped_at = "load"
            self._notify(observer, {"event": "stage_finished", "index": 0, **report.as_row()})
            return result

        boxes = self._stage(result, "detect", key, lambda: self._detect(path), observer=observer)
        if boxes is None:
            return self._stopped(result, observer)

        routed = self._stage(
            result, "classify", key, lambda: self._classify(boxes), cache=False, observer=observer
        )
        if routed is None:
            return self._stopped(result, observer)
        result.diagram_type, probability = routed
        if probability < self.confidence_floor or result.diagram_type == UNKNOWN:
            # 13.5: every stage after this one is type-specific, so a low-confidence type is a
            # question, not a guess.
            result.needs_confirmation = True
            result.stopped_at = "classify"
            self._notify(
                observer,
                {
                    "event": "run_finished",
                    "stopped_at": "classify",
                    "needs_confirmation": True,
                    "reason": f"routing probability {probability:.2f} is below the "
                    f"{self.confidence_floor:.2f} floor",
                },
            )
            return result

        diagram = self._stage(
            result,
            "assemble",
            key,
            lambda: self._assemble(path, boxes, result.diagram_type),
            observer=observer,
        )
        if diagram is None:
            return self._stopped(result, observer)
        result.ir = diagram

        order = self._stage(
            result, "traverse", key, lambda: self._traverse(diagram), observer=observer
        )
        if order is None:
            return self._stopped(result, observer)
        result.traversal = order

        text = self._stage(
            result, "serialise", key, lambda: self._serialise(diagram, order), observer=observer
        )
        if text is None:
            return self._stopped(result, observer)
        result.ir_text = text

        code = self._stage(
            result,
            "generate",
            key,
            lambda: self._generate(diagram, order, result.diagram_type),
            observer=observer,
        )
        if code is None:
            return self._stopped(result, observer)
        result.code, result.language = code

        self._stage(
            result,
            "verify",
            key,
            lambda: self._verify(result.code, result.diagram_type),
            observer=observer,
        )
        self._notify(observer, {"event": "run_finished", "stopped_at": result.stopped_at or None})
        return result

    # -- the observer (16.1.2) ----------------------------------------------------------

    @staticmethod
    def _notify(observer: Observer | None, event: dict[str, Any]) -> None:
        """Hand one event to the observer, and never let it end the run.

        13.8's rule is that no stage may raise through `run`, and an observer is a stage's worst
        neighbour: it is a client's callback, it writes to a queue or a socket, and the socket
        belongs to a phone that can walk into a lift mid-request. A broken pipe on the progress
        stream must not lose the page - the result is still computed, still returned and still
        stored, and the person can reload and read it by its id.
        """
        if observer is None:
            return
        # `Exception` and not something narrower, deliberately. The observer is a caller's callback
        # writing to a socket, and there is no useful list of what a socket can raise.
        with contextlib.suppress(Exception):
            observer(event)

    def _stopped(self, result: Result, observer: Observer | None) -> Result:
        """Close the stream on an early return. A client must never wait for a frame that is
        not coming: `_stage` already reported *which* stage failed and why, and this says that the
        run is over rather than merely quiet."""
        self._notify(observer, {"event": "run_finished", "stopped_at": result.stopped_at or None})
        return result

    def _stage(
        self,
        result: Result,
        name: str,
        key: str,
        work: Callable[[], Outcome[Any]],
        *,
        cache: bool = True,
        observer: Observer | None = None,
    ) -> Any:
        """Run one stage: cache, time, record, and turn a failure into a stop rather than a raise."""
        index = STAGES.index(name) + 1 if name in STAGES else 0
        self._notify(observer, {"event": "stage_started", "stage": name, "index": index})
        cache_key = f"{key}-{self.config_key}"
        if cache:
            stored = self.cache.get(name, cache_key)
            if stored is not None:
                report = StageReport(name, ok=True, seconds=0.0, cached=True)
                result.stages.append(report)
                # Reported like any other finish, `cached: true` and all. A stage that answers in
                # 0.0 s because the cache had it is a fact the client should be able to show -
                # a second run of the same page looking instant is the cache working, not a bug.
                self._notify(
                    observer, {"event": "stage_finished", "index": index, **report.as_row()}
                )
                return stored

        with Timer() as timer:
            try:
                outcome: Outcome[Any] = work()
            except Exception as error:  # noqa: BLE001 - 13.8: no stage may raise through `run`
                outcome = Outcome.failed(f"{type(error).__name__}: {error}")

        report = StageReport(
            name,
            ok=outcome.ok,
            seconds=timer.seconds,
            degraded=outcome.degraded,
            reason=outcome.reason,
            confidence=outcome.confidence,
        )
        result.stages.append(report)
        self._notify(observer, {"event": "stage_finished", "index": index, **report.as_row()})
        if not outcome.ok:
            result.stopped_at = name
            return None
        if cache:
            self.cache.put(name, cache_key, outcome.value)
        return outcome.value

    # -- the stages ---------------------------------------------------------------------

    def _detect(self, path: Path) -> Outcome[list[dict]]:
        from src.pipeline.vision import detect_boxes

        # Arrowheads are kept here and dropped inside `assemble_diagram`: 13.3's prior is
        # fitted on the full histogram, and the arrowhead count is its strongest feature.
        boxes = detect_boxes(path, keep_arrowheads=True)
        if not boxes:
            return Outcome.failed("the detector found nothing on this page")
        return Outcome(value=boxes)

    def _classify(self, boxes: list[dict]) -> Outcome[tuple[str, float]]:
        from src.pipeline.routing import route

        kind, probability = route(boxes)
        if kind == UNKNOWN:
            return Outcome(value=(UNKNOWN, 0.0), confidence=0.0)
        return Outcome(value=(kind, probability), confidence=probability)

    def _assemble(self, path: Path, boxes: list[dict], kind: str = "") -> Outcome[dict]:
        from src.pipeline.vision import assemble_diagram

        diagram = assemble_diagram(path, boxes, read_text=self.read_text, kind=kind)
        if not diagram.get("nodes"):
            return Outcome.failed("no nodes survived assembly")
        return Outcome(value=diagram)

    def _traverse(self, diagram: dict) -> Outcome[list[str]]:
        from src.parse.sequences import traversal

        order, _back = traversal(diagram)
        if not order:
            return Outcome.failed("the diagram has no reachable reading order")
        return Outcome(value=list(order))

    def _serialise(self, diagram: dict, order: list[str]) -> Outcome[str]:
        from src.codegen.serialise import serialise

        return Outcome(value=serialise(diagram, order))

    def _generate(self, diagram: dict, order: list[str], kind: str) -> Outcome[tuple[str, str]]:
        from src.pipeline.generate import emit, language_for

        if not self.generate_code:
            return Outcome.failed("code generation disabled for this run")
        try:
            language = language_for(kind)
        except KeyError:
            # The same answer `emit` gives for the same gap, one line earlier. `language_for`
            # used to default to "python" for a type with no emitter, which turned a missing
            # generator into a confident Python file.
            return Outcome.failed(f"no target generator for {kind!r}")
        return emit(diagram, order, kind, language, model_url=self.model_url)

    def _verify(self, code: str, kind: str) -> Outcome[dict]:
        """12.3.1's parse gate, routed by *diagram type* - `verify` picks the checker from it."""
        from src.codegen.targets import verify

        ok, detail = verify(code, kind)
        if not ok:
            return Outcome.fallback({"parses": False, "detail": detail}, detail, confidence=0.0)
        return Outcome(value={"parses": True, "detail": detail})

    # -- the budget ---------------------------------------------------------------------

    @staticmethod
    def within_budget(result: Result, budget: float = LATENCY_BUDGET_S) -> bool:
        return result.seconds < budget
