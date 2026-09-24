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
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.pipeline.cache import StageCache, digest_obj, env_key, image_key
from src.pipeline.contracts import UNKNOWN, Outcome, Result, StageReport, Timer

#: Below this routing probability the pipeline asks rather than guesses (13.5).
CONFIDENCE_FLOOR = 0.60

#: Target for the whole run, from the plan's 13.7 line.
LATENCY_BUDGET_S = 10.0


class DreamScriptPipeline:
    """Phases 9 through 12 composed into one call."""

    def __init__(
        self,
        *,
        cache: StageCache | None = None,
        confidence_floor: float = CONFIDENCE_FLOOR,
        generate: bool = True,
        read_text: bool = False,
    ) -> None:
        self.cache = cache if cache is not None else StageCache()
        self.confidence_floor = confidence_floor
        self.generate_code = generate
        self.read_text = read_text
        self.config_key = digest_obj(
            {
                "confidence_floor": confidence_floor,
                "generate": generate,
                "read_text": read_text,
                # The three constructor arguments were the whole key, and they are not the whole
                # configuration: `assemble.s5` takes three switches from the environment and
                # three more environment variables name the checkpoints that produce the text
                # and the edges. Flipping one of those and re-running returned the previous
                # answer from a warm cache. See `cache.ENV_KEYS`.
                "env": env_key(),
            }
        )

    # -- the driver ---------------------------------------------------------------------

    def run(self, image: str | Path) -> Result:
        """One page, all the way to code, or as far as it gets."""
        path = Path(image)
        result = Result(source=str(path))
        try:
            key = image_key(path)
        except OSError as error:
            result.stages.append(
                StageReport("load", ok=False, seconds=0.0, reason=f"unreadable: {error}")
            )
            result.stopped_at = "load"
            return result

        boxes = self._stage(result, "detect", key, lambda: self._detect(path))
        if boxes is None:
            return result

        routed = self._stage(result, "classify", key, lambda: self._classify(boxes), cache=False)
        if routed is None:
            return result
        result.diagram_type, probability = routed
        if probability < self.confidence_floor or result.diagram_type == UNKNOWN:
            # 13.5: every stage after this one is type-specific, so a low-confidence type is a
            # question, not a guess.
            result.needs_confirmation = True
            result.stopped_at = "classify"
            return result

        diagram = self._stage(
            result, "assemble", key, lambda: self._assemble(path, boxes, result.diagram_type)
        )
        if diagram is None:
            return result
        result.ir = diagram

        order = self._stage(result, "traverse", key, lambda: self._traverse(diagram))
        if order is None:
            return result
        result.traversal = order

        text = self._stage(result, "serialise", key, lambda: self._serialise(diagram, order))
        if text is None:
            return result
        result.ir_text = text

        code = self._stage(
            result, "generate", key, lambda: self._generate(diagram, order, result.diagram_type)
        )
        if code is None:
            return result
        result.code, result.language = code

        self._stage(result, "verify", key, lambda: self._verify(result.code, result.diagram_type))
        return result

    def _stage(
        self,
        result: Result,
        name: str,
        key: str,
        work: Callable[[], Outcome[Any]],
        *,
        cache: bool = True,
    ) -> Any:
        """Run one stage: cache, time, record, and turn a failure into a stop rather than a raise."""
        cache_key = f"{key}-{self.config_key}"
        if cache:
            stored = self.cache.get(name, cache_key)
            if stored is not None:
                result.stages.append(StageReport(name, ok=True, seconds=0.0, cached=True))
                return stored

        with Timer() as timer:
            try:
                outcome: Outcome[Any] = work()
            except Exception as error:  # noqa: BLE001 - 13.8: no stage may raise through `run`
                outcome = Outcome.failed(f"{type(error).__name__}: {error}")

        result.stages.append(
            StageReport(
                name,
                ok=outcome.ok,
                seconds=timer.seconds,
                degraded=outcome.degraded,
                reason=outcome.reason,
                confidence=outcome.confidence,
            )
        )
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
        return emit(diagram, order, kind, language)

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
