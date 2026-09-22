"""Phase 14.8 - a speedup number is only honest if its assumptions are reachable and swept.

These tests hold the model to that: every human-side parameter must be a declared field (so the
report can print it), the expected speedup must carry the failure rate rather than assume it
away, and the break-even figure must not be clamped into looking like a near-miss.
"""

from __future__ import annotations

from dataclasses import fields

from src.eval import humanbaseline as hb

SHAPE = {"diagrams": 162, "median_nodes": 14, "median_edges": 17, "median_characters": 2400}
MACHINE = {"median_s": 2.71, "pages": 25}


def test_every_human_side_assumption_is_a_named_parameter():
    names = {field.name for field in fields(hb.Model)}
    assert names == {
        "seconds_per_node",
        "seconds_per_edge",
        "chars_per_minute",
        "verification_factor",
        "review_seconds",
        "repair_seconds",
    }


def test_human_time_grows_with_the_drawing_and_with_the_program():
    model = hb.Model()
    small = model.human_seconds(5, 5, 500)
    more_nodes = model.human_seconds(50, 5, 500)
    more_code = model.human_seconds(5, 5, 5000)
    assert more_nodes > small
    assert more_code > small


def test_a_faster_typist_lowers_the_human_side_only():
    fast = hb.Model(chars_per_minute=400.0)
    slow = hb.Model(chars_per_minute=100.0)
    rates = {"end_to_end": 0.5}
    assert fast.human_seconds(10, 10, 2000) < slow.human_seconds(10, 10, 2000)
    assert (
        hb.compare(fast, SHAPE, MACHINE, rates)["machine_seconds_expected"]
        == hb.compare(slow, SHAPE, MACHINE, rates)["machine_seconds_expected"]
    )


def test_the_expected_speedup_charges_for_being_wrong():
    model = hb.Model()
    always_right = hb.compare(model, SHAPE, MACHINE, {"end_to_end": 1.0})
    often_wrong = hb.compare(model, SHAPE, MACHINE, {"end_to_end": 0.19})
    assert always_right["speedup_expected"] > often_wrong["speedup_expected"]
    assert always_right["machine_seconds_expected"] == always_right["machine_seconds_if_correct"]


def test_the_review_cost_is_charged_even_when_the_output_is_right():
    model = hb.Model(review_seconds=90.0)
    entry = hb.compare(model, SHAPE, MACHINE, {"end_to_end": 1.0})
    assert entry["machine_seconds_if_correct"] >= 90.0


def test_break_even_is_not_clamped_so_a_comfortable_margin_stays_visible():
    """Below zero means "wins even at a pass rate of zero", which is a result, not an error."""
    model = hb.Model(repair_seconds=240.0)
    entry = hb.compare(model, SHAPE, MACHINE, {"end_to_end": 0.19})
    assert entry["break_even_pass_rate"] < 0


def test_an_expensive_repair_can_erase_the_advantage():
    rates = {"end_to_end": 0.19}
    cheap = hb.compare(hb.Model(repair_seconds=60.0), SHAPE, MACHINE, rates)
    dear = hb.compare(hb.Model(repair_seconds=3000.0), SHAPE, MACHINE, rates)
    assert cheap["speedup_expected"] > dear["speedup_expected"]
    assert dear["speedup_expected"] < 1.0


def test_the_sensitivity_sweep_moves_only_the_parameter_it_names():
    arms = {arm["arm"]: arm for arm in hb.sensitivity(SHAPE, MACHINE, {"end_to_end": 0.19})}
    assert arms["default"]["parameters"]["chars_per_minute"] == 200.0
    assert arms["fast_typist_300cpm"]["parameters"]["chars_per_minute"] == 300.0
    assert arms["fast_typist_300cpm"]["parameters"]["repair_seconds"] == 240.0
    assert "repair_equals_rewrite" in arms


def test_the_report_says_that_no_human_was_timed():
    text = hb.render(
        {
            "human_study": "none was run",
            "corpus": SHAPE,
            "machine": MACHINE,
            "pass_rate": {"end_to_end": 0.19, "from_gold_ir": 0.784},
            "headline": hb.compare(hb.Model(), SHAPE, MACHINE, {"end_to_end": 0.19}),
            "sensitivity": hb.sensitivity(SHAPE, MACHINE, {"end_to_end": 0.19}),
            "protocol_for_a_real_study": ["10 participants"],
        }
    )
    assert "No human was timed" in text
    assert "10 participants" in text


def test_a_missing_machine_measurement_does_not_invent_one(monkeypatch, tmp_path):
    monkeypatch.setattr(hb, "CACHE_REPORT", tmp_path / "absent.json")
    entry = hb.machine_seconds()
    assert "note" in entry and "median_s" not in entry
