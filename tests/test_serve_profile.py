"""Phase 16.1.5 - the CPU fallback profile.

Running the profile costs a GPU pass and a CPU pass and is not a unit test; the measurement lives in
`reports/cpu_fallback.json` and this file checks the two things that could make that artefact a lie.

**The arithmetic**, driven with recorded payloads rather than by running models: cold separated from
warm, medians over the warm passes only, no percentile invented from five samples, and the budget
checked against both the median and the worst page.

**The artefact**, if it is present: that it was produced with the cache off, that its CPU arm really
had no device, and that the numbers in it are internally consistent. A committed report whose own
`within_budget` flag disagrees with its own medians is worse than no report.
"""

from __future__ import annotations

import json

import pytest

from src.serve import profile
from src.utils.config import ROOT


def payload(gpu: bool, totals: list[list[float]]) -> dict:
    """A recorded child payload. `totals[i]` is pass `i`; pass 0 is the warm-up."""
    runs = []
    for index, batch in enumerate(totals):
        for page, seconds in enumerate(batch):
            runs.append(
                {
                    "page": f"p{page}.png",
                    "repeat": index,
                    "cold": index == 0,
                    "seconds": seconds,
                    "ok": True,
                    "stopped_at": None,
                    "diagram_type": "flowchart",
                    "degraded": False,
                    # Two stages, so a ratio is computable and a zero-median stage is exercised.
                    "stages": {
                        "detect": round(seconds * 0.01, 3),
                        "assemble": round(seconds * 0.98, 3),
                    },
                    "reasons": {},
                }
            )
    return {
        "device": {
            "torch": "2.5.1+cu124",
            "cuda_available": gpu,
            "device_count": 1 if gpu else 0,
            "cuda_visible_devices": "<unset>" if gpu else "-1",
            "device_name": "NVIDIA RTX 4500 Ada Generation" if gpu else None,
            "cpu_count": 32,
            "torch_threads": 1 if gpu else 8,
        },
        "runs": runs,
    }


GPU = payload(True, [[11.43, 3.07, 0.42, 0.13, 0.21], [0.29, 0.24, 0.42, 0.12, 0.21]])
CPU = payload(False, [[18.8, 9.5, 18.0, 2.3, 3.6], [10.7, 7.6, 18.1, 2.3, 3.5]])


# == the arithmetic ===========================================================================


def test_the_whole_first_pass_is_warm_up_not_only_its_first_page():
    """The split that was wrong. Pass one read 11.43, 3.07, 0.42, 0.13, 0.21 - four of those five
    were being counted as steady state, and the 3.07 became a p95."""
    summary = profile.summarise(GPU)
    assert summary["warm"]["n"] == 5, "a second pass of five pages is five warm samples"
    assert summary["warm"]["max"] == 0.42, "no warm-up run may appear in the warm maximum"
    assert summary["cold_seconds"] == 11.43


def test_the_warm_up_pass_total_is_kept_because_the_first_user_pays_it():
    summary = profile.summarise(GPU)
    assert summary["warmup_pass_seconds"] == pytest.approx(15.26, abs=0.01)


def test_no_percentile_is_invented_from_five_samples():
    """Reporting a maximum under a percentile's name is how a small sample becomes a distribution.

    Checked on the per-stage blocks too, not only end to end. They kept a `p95` after the top-level
    one was removed, and in the committed report it equalled the maximum in every stage of both arms
    - which is as clear a demonstration as there could be of what it actually was.
    """
    summary = profile.summarise(GPU)
    assert set(summary["warm"]) == {"n", "median", "min", "max"}
    for stage, stats in summary["stages"].items():
        assert "p95" not in stats, stage
        assert set(stats) == {"n", "median", "min", "max"}, stage


def test_the_budget_is_checked_against_the_median_and_against_the_worst_page():
    """On CPU the two disagree, and the disagreement is the finding."""
    cpu = profile.summarise(CPU)
    assert cpu["within_budget"] is True, "a median page of 7.6s is inside a 10s budget"
    assert cpu["worst_within_budget"] is False, "and an 18.1s page is not"


def test_a_stage_the_gpu_finished_instantly_gets_no_ratio_rather_than_a_fabricated_one():
    """Dividing by 0.0, or by an arbitrary floor, would put a number where there is not one."""
    flat = payload(True, [[1.0] * 5, [1.0] * 5])
    for run in flat["runs"]:
        run["stages"] = {"detect": 0.0, "assemble": 1.0}
    comparison = profile.compare(
        {
            "gpu": {"arm": "gpu", "ok": True, **profile.summarise(flat)},
            "cpu": {"arm": "cpu", "ok": True, **profile.summarise(CPU)},
        }
    )
    assert comparison["per_stage"]["detect"] is None
    assert comparison["per_stage"]["assemble"] > 1


def test_the_end_to_end_ratio_is_the_warm_medians():
    comparison = profile.compare(
        {
            "gpu": {"arm": "gpu", "ok": True, **profile.summarise(GPU)},
            "cpu": {"arm": "cpu", "ok": True, **profile.summarise(CPU)},
        }
    )
    gpu, cpu = profile.summarise(GPU), profile.summarise(CPU)
    assert comparison["end_to_end"] == pytest.approx(
        cpu["warm"]["median"] / gpu["warm"]["median"], abs=0.01
    )


def test_an_arm_that_did_not_run_is_not_compared_against():
    """Half a comparison is not a comparison, and a ratio against a missing arm is a fiction."""
    comparison = profile.compare({"gpu": {"arm": "gpu", "ok": False, "detail": "no CUDA here"}})
    assert comparison["available"] is False
    assert comparison["detail"]


def test_the_summary_records_that_the_cache_was_off():
    """`ENV_KEYS` does not include the device, so a warm cache would have handed the CPU arm the GPU
    arm's answers and the profile would have measured a dict lookup."""
    assert profile.summarise(CPU)["cache_enabled"] is False


def test_the_cache_key_still_does_not_include_the_device():
    """The reason the cache has to be off. If this ever changes, the profile can stop disabling it -
    and this test is where that would be noticed rather than in a silently faster CPU arm."""
    from src.pipeline.cache import ENV_KEYS

    assert not any("CUDA" in name for name in ENV_KEYS)


# == the child's own reporting ================================================================


def test_one_failing_device_probe_does_not_erase_the_probes_that_worked():
    """`facts["torch"]` was being overwritten by a handler wrapped around every probe, so a failing
    `get_device_name` made the report claim torch itself was unavailable."""
    facts = profile.device_facts()
    assert not str(facts["torch"]).startswith("unavailable"), facts["torch"]
    assert "cuda_available" in facts
    assert facts["cuda_visible_devices"] is not None


def test_the_child_reports_the_environment_it_is_in_rather_than_what_was_intended():
    facts = profile.device_facts()
    assert facts["python"]
    assert facts["cpu_count"]


def test_the_last_json_line_is_found_past_whatever_transformers_printed():
    """Importing the pipeline makes transformers print a model config to stdout unbidden."""
    noisy = '{\n  "model_type": "trocr",\n}\nsome warning\n{"runs": [], "device": {}}\n'
    assert json.loads(profile._last_json_line(noisy)) == {"runs": [], "device": {}}


def test_a_child_that_printed_no_json_is_an_error_not_a_crash():
    with pytest.raises(IndexError):
        profile._last_json_line("only warnings here\n")


def test_the_cpu_arm_hides_devices_with_minus_one_and_not_the_empty_string():
    """`""` leaves torch reporting `is_available() == True` with zero devices, and the detect stage
    then fails in 1.5 s instead of running slowly. Both values were checked directly; this pins the
    one that works."""
    source = (ROOT / "src" / "serve" / "profile.py").read_text(encoding="utf-8")
    assert 'env["CUDA_VISIBLE_DEVICES"] = "-1"' in source


# == the committed artefact ===================================================================

REPORT = ROOT / "reports" / "cpu_fallback.json"


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve profile` first")
def test_the_committed_report_was_produced_with_the_cache_off():
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    assert held["cache_enabled"] is False
    for arm in held["arms"].values():
        if arm.get("ok"):
            assert arm["cache_enabled"] is False


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve profile` first")
def test_the_cpu_arm_really_had_no_device():
    """The one claim that makes a CPU profile a CPU profile."""
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    cpu = held["arms"].get("cpu")
    if not (cpu and cpu.get("ok")):
        pytest.skip("the CPU arm did not run on this machine")
    assert cpu["device"]["cuda_available"] is False
    assert cpu["device"]["device_count"] == 0
    assert cpu["device"]["cuda_visible_devices"] == "-1"


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve profile` first")
def test_the_report_agrees_with_itself():
    """A committed report whose own budget flags disagree with its own medians is worse than none."""
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    budget = held["budget_s"]
    for name, arm in held["arms"].items():
        if not arm.get("ok"):
            continue
        warm = arm["warm"]
        assert warm["min"] <= warm["median"] <= warm["max"], name
        assert arm["within_budget"] == (warm["median"] <= budget), name
        assert arm["worst_within_budget"] == (warm["max"] <= budget), name


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve profile` first")
def test_every_run_in_the_report_says_which_rung_answered():
    """13.4, at the one place it would be easiest to drop: a latency measured with the emitter
    answering is not a latency for a deployment that serves the adapter."""
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    for arm in held["arms"].values():
        if not arm.get("ok"):
            continue
        assert arm["pages"], "an arm that ran has pages"
        for page in arm["pages"]:
            assert "degraded" in page


MARKDOWN = ROOT / "reports" / "cpu_fallback.md"


@pytest.mark.skipif(not MARKDOWN.is_file(), reason="run `python -m src.serve profile` first")
def test_the_markdown_is_generated_from_the_json_so_the_two_cannot_disagree():
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    text = MARKDOWN.read_text(encoding="utf-8")
    assert text.strip() == profile.markdown(held).strip(), (
        "reports/cpu_fallback.md is not what the current code would write from "
        "reports/cpu_fallback.json - regenerate it with `python -m src.serve profile`, or if only "
        "the prose changed, the two artefacts were committed out of step"
    )


@pytest.mark.skipif(not MARKDOWN.is_file(), reason="run `python -m src.serve profile` first")
def test_the_markdown_states_the_finding_rather_than_only_tabulating_it():
    text = MARKDOWN.read_text(encoding="utf-8")
    assert (
        "CUDA_VISIBLE_DEVICES=-1" in text
    ), "the -1 finding belongs in the report, not only in code"
    assert "slower end to end" in text
    assert "emitter" in text, "which rung answered has to be in the artefact"
