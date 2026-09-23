"""Phase 13.8 - the pipeline must not raise, whatever it is handed.

13.8's promise is that *every stage fails soft with an actionable message*, and its definition of
done is "no unhandled exception in fuzz test". The other suite checks that with fixed examples;
this one checks it with generated ones, because the interesting inputs are the ones nobody
thought to write down - a PNG header with no body, a box whose width is NaN, a detector that
returns a string where a dict belongs.

## What is fuzzed, and what is stubbed

The **detector is stubbed throughout**. Fuzzing it would mean loading YOLO once per example and
would measure ultralytics rather than this pipeline's error handling, and the property under test
is what `DreamScriptPipeline` does with whatever comes back - including a detector that raises,
returns nothing, or returns nonsense with the right shape.

Three layers get generated input:

    files     arbitrary bytes written to a `.png`, including empty and truncated-header files
    boxes     detector output with missing keys, wrong types, NaN/inf geometry, absurd sizes
    contracts `Outcome` / `Result` / `StageReport` built from arbitrary values

The invariant is the same at every layer: **`run` returns a `Result`**. It may be a `Result` that
stopped at the first stage with a reason, which is a correct answer to a corrupt page; what it may
never be is a traceback escaping into the caller.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.pipeline.cache import StageCache
from src.pipeline.contracts import Outcome, Result, StageReport
from src.pipeline.core import DreamScriptPipeline

#: Fuzzing writes a file per example, so the deadline has to allow for disk, and the pipeline
#: reuses its own cache directory between examples by design.
SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)


def _real_png(directory, size=(320, 240)):
    """A genuinely decodable page, so the fuzz reaches past `assemble`.

    With a header-only file `cv2.imread` returns None and every example stops at the same stage,
    which tests one branch sixty times. A real image lets malformed *boxes* be the variable.
    """
    import cv2

    rng = np.random.default_rng(0)
    page = directory / "page.png"
    cv2.imwrite(str(page), rng.integers(0, 255, (size[1], size[0]), dtype=np.uint8))
    return page


def _pipeline(monkeypatch, boxes):
    """A pipeline whose detector returns `boxes` - or raises, if `boxes` is an exception."""

    def fake_detect(path, min_score=0.25):
        if isinstance(boxes, BaseException):
            raise boxes
        return boxes

    monkeypatch.setattr("src.pipeline.vision.detect_boxes", fake_detect)
    return DreamScriptPipeline(cache=StageCache(enabled=False))


# -- layer 1: the bytes on disk ----------------------------------------------------------


@SETTINGS
@given(payload=st.binary(min_size=0, max_size=2048))
def test_arbitrary_file_bytes_never_raise(tmp_path_factory, monkeypatch, payload):
    page = tmp_path_factory.mktemp("fuzz") / "page.png"
    page.write_bytes(payload)
    pipeline = _pipeline(monkeypatch, [])
    result = pipeline.run(page)
    assert isinstance(result, Result)
    # Nothing was detectable, so it must have stopped and said so rather than inventing an IR.
    assert result.stopped_at
    assert result.ir is None


def test_a_missing_file_stops_at_load(tmp_path, monkeypatch):
    pipeline = _pipeline(monkeypatch, [])
    result = pipeline.run(tmp_path / "does-not-exist.png")
    assert result.stopped_at == "load"
    assert "unreadable" in result.stages[0].reason


def test_a_directory_in_place_of_a_page_stops_at_load(tmp_path, monkeypatch):
    pipeline = _pipeline(monkeypatch, [])
    result = pipeline.run(tmp_path)
    assert result.stopped_at == "load"


# -- layer 2: whatever the detector returns ----------------------------------------------

#: Geometry a detector should never emit but might: NaN, inf, negatives, absurd magnitudes.
NASTY_NUMBERS = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.integers(min_value=-(10**9), max_value=10**9),
    st.just(0),
)

BOXES = st.lists(
    st.fixed_dictionaries(
        {},
        optional={
            "id": st.one_of(st.text(max_size=8), st.integers(), st.none()),
            "bbox": st.one_of(
                st.lists(NASTY_NUMBERS, min_size=0, max_size=6),
                st.text(max_size=5),
                st.none(),
            ),
            "cls": st.one_of(st.text(max_size=12), st.integers(), st.none()),
            "score": NASTY_NUMBERS,
        },
    ),
    max_size=6,
)


@SETTINGS
@given(boxes=BOXES)
def test_malformed_detector_output_never_raises(tmp_path_factory, monkeypatch, boxes):
    page = _real_png(tmp_path_factory.mktemp("fuzz"))
    pipeline = _pipeline(monkeypatch, boxes)
    result = pipeline.run(page)
    assert isinstance(result, Result)
    # Either it stopped somewhere with a reason, or it got all the way to code. Never both, and
    # never a traceback.
    assert result.stopped_at or result.ok or result.needs_confirmation
    for stage in result.stages:
        assert isinstance(stage.seconds, float)
        assert stage.ok or stage.reason or stage.name == "verify"


@SETTINGS
@given(
    error=st.sampled_from(
        [
            RuntimeError("the card fell out"),
            ValueError("bad weights"),
            KeyError("cls"),
            MemoryError(),
            OSError("device busy"),
        ]
    )
)
def test_a_detector_that_raises_becomes_a_stop(tmp_path_factory, monkeypatch, error):
    page = tmp_path_factory.mktemp("fuzz") / "page.png"
    page.write_bytes(b"\x89PNG\r\n\x1a\n")
    pipeline = _pipeline(monkeypatch, error)
    result = pipeline.run(page)
    assert result.stopped_at == "detect"
    assert type(error).__name__ in result.stages[-1].reason


# -- layer 3: the contracts themselves ---------------------------------------------------


@SETTINGS
@given(
    name=st.text(max_size=20),
    seconds=st.floats(min_value=0, max_value=1e6, allow_nan=False, allow_infinity=False),
    reason=st.text(max_size=50),
)
def test_a_stage_report_always_renders_a_row(name, seconds, reason):
    row = StageReport(name, ok=False, seconds=seconds, reason=reason).as_row()
    assert row["stage"] == name
    assert isinstance(row["seconds"], float)


@SETTINGS
@given(value=st.one_of(st.none(), st.integers(), st.text(), st.lists(st.integers())))
def test_an_outcome_is_ok_exactly_when_it_has_a_value(value):
    outcome: Outcome = Outcome(value=value)
    assert outcome.ok == (value is not None)


@SETTINGS
@given(
    stages=st.lists(
        st.builds(
            StageReport,
            st.text(max_size=10),
            st.booleans(),
            st.floats(min_value=0, max_value=1000, allow_nan=False, allow_infinity=False),
        ),
        max_size=8,
    )
)
def test_a_result_totals_its_stages_whatever_they_are(stages):
    result = Result(source="x.png")
    result.stages = stages
    assert result.seconds == pytest.approx(round(sum(s.seconds for s in stages), 3))
    assert len(result.timing_table()) == len(stages)
