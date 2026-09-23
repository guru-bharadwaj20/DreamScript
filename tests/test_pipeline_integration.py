"""Phase 13.10 - the pipeline end to end, and the contracts that let it fail without raising.

Two halves, because they cost very different things:

* the **contract** tests run anywhere. They pin the behaviour 13.8 promises - a stage that
  cannot answer returns a reason rather than an exception, `run` stops at that stage, and the
  stages that did run keep their timings - by driving the pipeline with stubs instead of models.
* the **golden set** tests are marked `gpu` and `slow`. They run the real detector and the real
  emitter over 25 held-out test pages (`tests/fixtures/p13_golden.json`, stratified 8 fa_bresler
  / 17 hdbpmn in the proportion of the test split itself) and assert the things a deployment
  cares about: every page produces either code or a reason, the median page stays inside 13.7's
  budget, and a second run over the same pages is faster because of 13.6's cache.

The golden pages are all from the **test** split, which the detector never trained on and the
router was neither fitted nor selected on, so a number here is a held-out number.
"""

from __future__ import annotations

import json

import pytest

from src.pipeline import cli
from src.pipeline.cache import StageCache, digest_obj, image_key
from src.pipeline.contracts import UNKNOWN, Outcome, Result, StageReport
from src.pipeline.core import LATENCY_BUDGET_S, DreamScriptPipeline
from src.utils.config import ROOT

GOLDEN = ROOT / "tests" / "fixtures" / "p13_golden.json"


# -- 13.2 / 13.8: a failure is a value ---------------------------------------------------


def test_an_outcome_is_either_a_value_or_a_reason_never_both():
    assert Outcome(value={"a": 1}).ok
    assert not Outcome.failed("nothing to see").ok
    assert Outcome.failed("nothing to see").reason == "nothing to see"


def test_a_fallback_carries_its_value_and_admits_it_is_one():
    outcome = Outcome.fallback(("code", "python"), "emitter answered")
    assert outcome.ok
    assert outcome.degraded
    assert outcome.value == ("code", "python")


def test_an_unreadable_page_stops_at_load_and_does_not_raise(tmp_path):
    result = DreamScriptPipeline().run(tmp_path / "no-such-page.png")
    assert result.stopped_at == "load"
    assert not result.ok
    assert "unreadable" in result.stages[0].reason


def test_a_stage_that_raises_becomes_a_stop_not_a_traceback():
    pipeline = DreamScriptPipeline()
    result = Result(source="x.png")
    value = pipeline._stage(
        result,
        "detect",
        "k",
        lambda: (_ for _ in ()).throw(RuntimeError("card fell out")),
        cache=False,
    )
    assert value is None
    assert result.stopped_at == "detect"
    assert "RuntimeError: card fell out" in result.stages[-1].reason


def test_the_timing_table_keeps_the_stages_that_did_run():
    result = Result(source="x.png")
    result.stages = [
        StageReport("detect", ok=True, seconds=1.5),
        StageReport("classify", ok=False, seconds=0.25, reason="no idea"),
    ]
    rows = result.timing_table()
    assert [r["stage"] for r in rows] == ["detect", "classify"]
    assert result.seconds == 1.75
    assert rows[1]["reason"] == "no idea"


# -- 13.5: the confidence gate -----------------------------------------------------------


def test_a_low_confidence_route_asks_rather_than_guessing(monkeypatch):
    pipeline = DreamScriptPipeline(confidence_floor=0.9)
    monkeypatch.setattr(pipeline, "_detect", lambda path: Outcome(value=[{"cls": "circle"}]))
    monkeypatch.setattr(
        pipeline, "_classify", lambda boxes: Outcome(value=("flowchart", 0.51), confidence=0.51)
    )
    result = pipeline.run(GOLDEN)  # any readable file; detection is stubbed
    assert result.needs_confirmation
    assert result.stopped_at == "classify"
    assert result.code == ""


def test_an_unknown_type_is_never_forced_into_a_guess(monkeypatch):
    pipeline = DreamScriptPipeline()
    monkeypatch.setattr(pipeline, "_detect", lambda path: Outcome(value=[{"cls": "freeform"}]))
    monkeypatch.setattr(
        pipeline, "_classify", lambda boxes: Outcome(value=(UNKNOWN, 0.0), confidence=0.0)
    )
    result = pipeline.run(GOLDEN)
    assert result.needs_confirmation
    assert result.diagram_type == UNKNOWN


# -- 13.6: the cache ---------------------------------------------------------------------


def test_the_cache_keys_on_bytes_not_on_the_name(tmp_path):
    one, two = tmp_path / "a.png", tmp_path / "b.png"
    one.write_bytes(b"same pixels")
    two.write_bytes(b"same pixels")
    assert image_key(one) == image_key(two)


def test_a_different_configuration_is_a_miss_not_a_wrong_answer(tmp_path):
    cache = StageCache(directory=tmp_path)
    cache.put("detect", f"page-{digest_obj({'floor': 0.6})}", [{"cls": "circle"}])
    assert cache.get("detect", f"page-{digest_obj({'floor': 0.9})}") is None
    assert cache.get("detect", f"page-{digest_obj({'floor': 0.6})}") == [{"cls": "circle"}]


def test_a_disabled_cache_never_answers(tmp_path):
    cache = StageCache(directory=tmp_path, enabled=False)
    cache.put("detect", "k", [1, 2, 3])
    assert cache.get("detect", "k") is None


# -- 13.9: the batch CLI -----------------------------------------------------------------


def test_every_generated_language_has_a_file_extension():
    from src.pipeline.generate import LANGUAGES

    for language in set(LANGUAGES.values()):
        assert language in cli.SUFFIX, f"{language} would be written as .txt"


def test_a_directory_with_no_images_is_an_error_with_a_reason(tmp_path):
    with pytest.raises(SystemExit) as raised:
        cli.pages_under(tmp_path)
    assert "no images" in str(raised.value)


def test_a_single_file_target_is_just_that_file(tmp_path):
    page = tmp_path / "one.png"
    page.write_bytes(b"x")
    assert cli.pages_under(page) == [page]


def test_the_summary_counts_where_pages_stopped():
    results = [
        Result(source="a.png", code="print(1)", language="python", diagram_type="flowchart"),
        Result(source="b.png", stopped_at="detect"),
        Result(source="c.png", stopped_at="detect"),
    ]
    summary = cli.summarise(results, LATENCY_BUDGET_S)
    assert summary["pages"] == 3
    assert summary["produced_code"] == 1
    assert summary["stopped_at"] == {"detect": 2}
    assert summary["diagram_types"] == {"flowchart": 1}


def test_the_report_names_every_page_including_the_failures():
    results = [
        Result(source="a.png", code="print(1)", language="python", diagram_type="flowchart"),
        Result(source="b.png", stopped_at="assemble"),
    ]
    text = cli.report_markdown(cli.summarise(results, LATENCY_BUDGET_S), results)
    assert "a.png" in text and "b.png" in text
    assert "assemble" in text


def test_a_batch_that_produced_nothing_exits_non_zero(tmp_path, monkeypatch):
    # The detector is stubbed rather than loaded: this is about the CLI's exit code, and a page
    # the detector finds nothing on is exactly the case being exercised.
    monkeypatch.setattr("src.pipeline.vision.detect_boxes", lambda path, min_score=0.25: [])
    page = tmp_path / "page.png"
    page.write_bytes(b"not really a png")
    code = cli.run(["run", str(page), "--out", str(tmp_path / "out"), "--quiet"])
    assert code == 1
    summary = json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf-8"))
    assert summary["stopped_at"] == {"detect": 1}


# -- 13.10: the golden set ---------------------------------------------------------------


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    if not GOLDEN.is_file():
        pytest.skip("golden set not built")
    pages = json.loads(GOLDEN.read_text(encoding="utf-8"))
    missing = [p["name"] for p in pages if not (ROOT / p["image"]).is_file()]
    if missing:
        pytest.skip(f"golden images missing: {len(missing)}")
    return pages


@pytest.fixture(scope="module")
def golden_results(golden) -> list[Result]:
    pipeline = DreamScriptPipeline(cache=StageCache(enabled=False))
    return [pipeline.run(page["image"]) for page in golden]


@pytest.mark.gpu
@pytest.mark.slow
def test_the_golden_set_is_the_shape_it_claims(golden):
    assert len(golden) == 25
    sources = {p["source"] for p in golden}
    assert sources == {"fa_bresler", "hdbpmn"}


@pytest.mark.gpu
@pytest.mark.slow
def test_every_golden_page_produces_code_or_a_reason(golden_results):
    for result in golden_results:
        assert (
            result.ok or result.stopped_at or result.needs_confirmation
        ), f"{result.source} returned neither code nor a reason"


@pytest.mark.gpu
@pytest.mark.slow
def test_generated_code_parses_when_it_is_produced(golden_results):
    produced = [r for r in golden_results if r.ok]
    assert produced, "no golden page produced code at all"
    verified = [r for r in produced if r.stages[-1].name == "verify" and r.stages[-1].ok]
    assert (
        len(verified) / len(produced) >= 0.95
    ), f"only {len(verified)}/{len(produced)} generated programs parse"


@pytest.mark.gpu
@pytest.mark.slow
def test_the_median_page_stays_inside_the_latency_budget(golden_results):
    seconds = sorted(r.seconds for r in golden_results)
    median = seconds[len(seconds) // 2]
    assert median < LATENCY_BUDGET_S, f"median page took {median}s against {LATENCY_BUDGET_S}s"


@pytest.mark.gpu
@pytest.mark.slow
def test_a_warm_cache_run_is_faster_than_a_cold_one(golden, tmp_path):
    pipeline = DreamScriptPipeline(cache=StageCache(directory=tmp_path))
    pages = [p["image"] for p in golden[:5]]

    cold = [pipeline.run(page) for page in pages]
    warm = [pipeline.run(page) for page in pages]

    assert sum(r.seconds for r in warm) < sum(r.seconds for r in cold)
    assert any(s.cached for r in warm for s in r.stages), "nothing was served from the cache"


@pytest.mark.gpu
@pytest.mark.slow
def test_the_emitter_answers_when_no_model_is_served(golden_results):
    """13.4: with nothing served, every program came from 12.1.6's emitter and says so."""
    produced = [r for r in golden_results if r.ok]
    generate = [next(s for s in r.stages if s.name == "generate") for r in produced]
    assert all(s.degraded for s in generate), "a generate stage claimed a model it did not use"


def test_a_cache_entry_written_by_different_code_is_a_miss(tmp_path, monkeypatch):
    """The cache must not answer with results the current code would not produce.

    Found by accident and worth a test: the key was the image plus a config hash of three
    constructor arguments, so wiring the S5 assembly stages in changed what `assemble` returns
    for every page while a warm cache went on serving the old IR - a state machine that had been
    rebuilt with real states and transitions still came back with one merged state and none.
    """
    from src.pipeline import cache as cache_module

    store = cache_module.StageCache(directory=tmp_path)
    store.put("assemble", "k1", {"nodes": [1, 2]})
    assert store.get("assemble", "k1") == {"nodes": [1, 2]}

    monkeypatch.setattr(cache_module, "code_key", lambda: "0000000000000000")
    fresh = cache_module.StageCache(directory=tmp_path)
    assert fresh.get("assemble", "k1") is None
    assert fresh.stats() == {"hits": 0, "misses": 1}


def test_the_code_key_is_stable_within_a_process():
    from src.pipeline.cache import code_key

    assert code_key() == code_key()
    assert len(code_key()) == 16
