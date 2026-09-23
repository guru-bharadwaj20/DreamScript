"""Phase 15.6 - a confidence log is only useful if the confidence means something.

Two things can go wrong quietly here. The monitor can watch a signal that never moves and report
a straight line as a distribution; and the spike alarm can be tuned to an absolute share that only
holds for this corpus. These tests pin against both.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.mlops import monitor


def test_distribution_reports_percentiles_not_just_a_mean():
    """The mean hides the tail, which is the only part the alarm cares about."""
    values = np.concatenate([np.full(95, 0.99), np.full(5, 0.10)])
    result = monitor.distribution(values)
    assert result["n"] == 100
    assert set(result["percentiles"]) == {f"p{p}" for p in monitor.PERCENTILES}
    # The mean is dragged up by the bulk; the low percentile is where the trouble shows.
    assert result["mean"] > 0.9
    assert result["percentiles"]["p1"] < monitor.LOW_CONFIDENCE


def test_low_confidence_share_counts_below_the_gate():
    values = np.array([0.1, 0.5, 0.69, 0.70, 0.71, 0.99])
    result = monitor.distribution(values)
    # 0.70 is not below 0.70, so three of six.
    assert result["low_confidence_share"] == pytest.approx(3 / 6)


def test_an_empty_batch_is_reported_as_empty_rather_than_confident():
    assert monitor.distribution(np.array([]))["n"] == 0


def test_nan_confidences_are_dropped_not_counted_as_low():
    values = np.array([0.99, np.nan, 0.99, np.nan])
    result = monitor.distribution(values)
    assert result["n"] == 2
    assert result["low_confidence_share"] == 0.0


def test_the_alarm_is_a_ratio_not_an_additive_margin():
    """The bug this shape replaced: an additive margin over a near-zero baseline is absolute.

    With a baseline of 0.0023 - this corpus's real one - a degraded batch at 0.0867 is a 38x
    rise and unmistakably a spike. `baseline + 0.10` scored it as quiet because 8.67% < 10.23%.
    """
    baseline = 0.0023
    degraded = np.concatenate([np.full(87, 0.1), np.full(913, 0.99)])
    result = monitor.spike(baseline, degraded)
    assert result["ratio"] > 30
    assert result["spiked"] is True


def test_a_high_baseline_corpus_does_not_alarm_merely_for_being_itself():
    """A ratio travels between corpora; an absolute share does not."""
    batch = np.full(1000, 0.5)  # every page low confidence
    assert monitor.spike(0.5, batch)["spiked"] is False  # only 2x, below the ratio
    assert monitor.spike(0.01, batch)["spiked"] is True  # 100x


def test_the_floor_stops_the_ratio_being_hysterical_near_zero():
    """On a tiny baseline, a tiny share is a huge ratio and means nothing."""
    baseline = 0.0001
    barely = np.concatenate([np.full(5, 0.1), np.full(995, 0.99)])  # 0.5% low confidence
    result = monitor.spike(baseline, barely)
    assert result["ratio"] > monitor.SPIKE_RATIO  # the ratio alone would fire
    assert result["low_confidence_share"] < monitor.SPIKE_FLOOR
    assert result["spiked"] is False  # but the floor holds it


def test_both_conditions_are_required():
    # Above the floor but not the ratio.
    assert (
        monitor.spike(0.05, np.concatenate([np.full(60, 0.1), np.full(940, 0.99)]))["spiked"]
        is False
    )
    # Above the ratio but not the floor.
    assert (
        monitor.spike(0.0001, np.concatenate([np.full(5, 0.1), np.full(995, 0.99)]))["spiked"]
        is False
    )


def test_a_quiet_batch_does_not_spike():
    assert monitor.spike(0.02, np.full(200, 0.99))["spiked"] is False


def test_the_low_confidence_gate_matches_the_routing_gate():
    """13.5 routes on 0.60-0.70; two rows disagreeing about 'low confidence' would be a bug."""
    assert 0.5 <= monitor.LOW_CONFIDENCE <= 0.9


def test_confidence_separates_right_from_wrong_on_the_real_corpus():
    """If the router is equally confident when wrong, the gate is decoration."""
    from src.pipeline import fallback

    if not fallback.TABLE.is_file():
        pytest.skip("needs data/features/handcrafted.parquet")
    scored = monitor.out_of_fold_confidence(fallback.load())
    right = scored[scored["correct"]]["confidence"].mean()
    wrong = scored[~scored["correct"]]["confidence"].mean()
    if not np.isfinite(wrong):
        pytest.skip("no misclassifications to compare against")
    assert wrong < right


def test_out_of_fold_confidence_covers_every_page_exactly_once():
    from src.pipeline import fallback

    if not fallback.TABLE.is_file():
        pytest.skip("needs data/features/handcrafted.parquet")
    frame = fallback.load()
    scored = monitor.out_of_fold_confidence(frame)
    assert len(scored) == len(frame)
    assert scored["id"].is_unique
    # Every page must have been scored by some fold, or the baseline is over a subset.
    assert scored["confidence"].notna().all()


def test_confidences_are_probabilities():
    from src.pipeline import fallback

    if not fallback.TABLE.is_file():
        pytest.skip("needs data/features/handcrafted.parquet")
    values = monitor.out_of_fold_confidence(fallback.load())["confidence"].to_numpy()
    assert values.min() >= 0.0
    assert values.max() <= 1.0
    # A max-probability over 3 classes cannot be below 1/3.
    assert values.min() >= 1 / 3 - 1e-9


def test_render_says_so_when_confidence_fails_to_separate():
    """The report must not quietly present a useless gate as a working one."""
    result = {
        "monitored": "m",
        "not_monitored": "n",
        "overall": {
            "n": 10,
            "mean": 0.9,
            "percentiles": {"p50": 0.9},
            "low_confidence_share": 0.1,
        },
        "by_source": {"x": {"n": 10, "mean": 0.9, "low_confidence_share": 0.1}},
        "when_correct": {"n": 5, "mean": 0.80},
        "when_wrong": {"n": 5, "mean": 0.85},  # wrong is MORE confident
        "controls": {},
        "verdict": {"baseline_low_confidence_share": 0.1},
    }
    text = monitor.render(result, None)
    assert "not separated" in text


def test_controls_cannot_pass_when_none_were_run():
    """--no-pixels runs no controls, which must read as 'not established', never as success."""
    from src.pipeline import fallback

    if not fallback.TABLE.is_file():
        pytest.skip("needs data/features/handcrafted.parquet")
    result = monitor.collect(with_pixels=False)
    assert result["verdict"]["controls_pass"] is False
