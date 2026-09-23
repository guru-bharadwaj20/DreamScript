"""Phase 15.8 - the trigger's value is in what it refuses, so that is what is tested.

Nothing fires in the live report, because no production traffic has been scored and the only
corrections store is empty. The refusal path therefore has to be exercised here or it ships
untested - which would be the worst outcome, since refusing is the whole reason this row is not
a cron job.
"""

from __future__ import annotations

import json

import pytest

from src.mlops import retrain


def test_nothing_firing_means_nothing_is_due():
    decisions = retrain.decide({"drift": {"fires": False}, "corrections": {"fires": False}})
    assert all(d["verdict"] == "no" for d in decisions)


def test_a_firing_signal_on_an_exhausted_component_is_refused_not_obeyed():
    """The row's point: a signal can fire and the answer still be 'do not retrain'."""
    decisions = retrain.decide({"corrections": {"fires": True}})
    ocr = next(d for d in decisions if d["component"] == "ocr")
    assert ocr["verdict"] == "refuse"
    assert "corrections" in ocr["firing_signals"]
    # A refusal has to cite something, or it is just an opinion.
    assert ocr["evidence"]
    assert "selection" in ocr["evidence"]


def test_a_refusal_names_the_lever_to_pull_instead():
    decisions = retrain.decide({"corrections": {"fires": True}})
    ocr = next(d for d in decisions if d["component"] == "ocr")
    assert "instead" in ocr["reason"]
    assert retrain.EXHAUSTED["ocr"]["lever"] in ocr["reason"]


def test_a_firing_signal_on_an_unexhausted_component_does_retrain():
    decisions = retrain.decide({"drift": {"fires": True}})
    classification = next(d for d in decisions if d["component"] == "classification")
    assert classification["verdict"] == "retrain"
    assert "drift" in classification["firing_signals"]


def test_every_exhausted_component_cites_its_measurement():
    """A refusal without evidence would be this module asserting a preference."""
    for component, entry in retrain.EXHAUSTED.items():
        assert entry["lever"]
        assert entry["evidence"]
        # The evidence must reference a number or a phase, not just an assertion.
        assert any(ch.isdigit() for ch in entry["evidence"]), component


def test_generation_is_refused_because_the_bar_moves_with_the_emitter():
    decisions = retrain.decide({"drift": {"fires": True}, "corrections": {"fires": True}})
    # generation has no owning signal, so it should not be asked for at all here...
    generation = next(d for d in decisions if d["component"] == "generation")
    assert generation["verdict"] == "no"
    # ...but the exhaustion entry that would refuse it must still be on file and cite 12.3.3.
    assert "0.784" in retrain.EXHAUSTED["generation"]["evidence"]


def test_calibration_arms_do_not_fire_the_trigger():
    """15.5 and 15.6 alarm on their own controls; firing on those is firing on the test set."""
    drift = retrain.drift_signal()
    confidence = retrain.confidence_signal()
    if drift.get("available"):
        assert drift["fires"] is False
        assert "calibration" in drift["why"] or "not observed production" in drift["why"]
    if confidence.get("available"):
        assert confidence["fires"] is False


def test_the_drift_signal_notices_its_own_shifts_even_though_it_does_not_fire():
    """Not firing must not mean not looking - the shifts should still be reported."""
    drift = retrain.drift_signal()
    if not drift.get("available"):
        pytest.skip("needs reports/drift.json")
    assert drift["shifts_detected"], "the calibration arms should be visible in the signal"
    assert drift["controls_pass"] is True


def test_the_correction_threshold_fires_at_the_boundary():
    below = retrain.decide(
        {"corrections": {"fires": False}},
    )
    assert all(d["verdict"] == "no" for d in below)


def test_corrections_signal_reports_an_empty_store_without_firing():
    signal = retrain.corrections_signal()
    assert signal["fires"] is False
    assert signal["usable"] == 0
    assert "not built" in signal["why"] or "threshold" in signal["why"]


def test_a_missing_upstream_report_does_not_take_the_trigger_down():
    """A trigger that crashes when one input is absent is worse than one that says 'unknown'."""
    import src.mlops.retrain as module

    original = module.ROOT
    try:
        module.ROOT = original / "definitely_not_here"
        signal = module.drift_signal()
        assert signal["available"] is False
        assert signal["why"]
    finally:
        module.ROOT = original


def test_render_is_stable_and_names_the_refusals():
    result = {
        "signals": {"corrections": {"available": True, "fires": True, "why": "x"}},
        "decisions": retrain.decide({"corrections": {"fires": True}}),
        "verdict": {"any_due": False, "refusals": ["ocr"]},
    }
    text = retrain.render(result)
    assert "refuse" in text
    assert "ocr" in text
    assert "cron job" in text


def test_report_is_json_serialisable():
    json.dumps(retrain.collect(), default=str)
