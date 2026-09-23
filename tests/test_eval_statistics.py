"""Phase 14.10 - an interval is a claim about a procedure, so the procedure is what is tested.

Three things could make these bars decorative: a bootstrap that is not reproducible, an interval
computed over the wrong unit (characters rather than crops), and a seed spread invented where the
criterion never ran repeats. Each has a test.
"""

from __future__ import annotations

import json
import statistics

import pytest

from src.eval import statistics_ as stats


def test_the_bootstrap_is_reproducible_for_a_given_seed():
    items = [float(index % 7) for index in range(50)]
    first = stats.bootstrap(items, statistics.fmean, resamples=200, seed=11)
    second = stats.bootstrap(items, statistics.fmean, resamples=200, seed=11)
    assert first == second
    assert first != stats.bootstrap(items, statistics.fmean, resamples=200, seed=12)


def test_the_interval_brackets_the_point_estimate():
    items = [0.0] * 30 + [1.0] * 70
    entry = stats.bootstrap(items, statistics.fmean, resamples=500)
    assert entry["point"] == pytest.approx(0.7, abs=1e-9)
    assert entry["ci95"][0] <= entry["point"] <= entry["ci95"][1]
    assert entry["items"] == 100


def test_more_items_give_a_narrower_interval():
    small = stats.bootstrap([0.0, 1.0] * 15, statistics.fmean, resamples=400)
    large = stats.bootstrap([0.0, 1.0] * 300, statistics.fmean, resamples=400)
    assert large["half_width"] < small["half_width"]


def test_an_empty_sample_produces_no_interval():
    assert stats.bootstrap([], statistics.fmean) == {}


def test_the_resample_draws_whole_items():
    """The unit has to be the crop or the page; resampling inside an item would narrow the bar."""
    seen = []

    def statistic(sample):
        seen.append(sample)
        return 0.0

    items = [("truth", "prediction")] * 5
    stats.bootstrap(items, statistic, resamples=3, seed=1)
    for sample in seen:
        assert len(sample) == len(items)
        assert all(entry in items for entry in sample)


def test_seed_spread_is_read_from_the_repeats_not_estimated(tmp_path):
    path = tmp_path / "s1.json"
    path.write_text(
        json.dumps(
            {
                "accuracy": 0.9871,
                "target_accuracy": 0.92,
                "repeats": [{"accuracy": 0.9873}, {"accuracy": 0.9858}, {"accuracy": 0.9881}],
            }
        ),
        encoding="utf-8",
    )
    entry = stats._seeded(path, "accuracy", "accuracy")
    assert entry["seeds"] == 3
    assert entry["seed_min"] == 0.9858
    assert entry["seed_max"] == 0.9881
    assert entry["seed_std"] > 0


def test_a_criterion_without_repeats_says_so_rather_than_showing_a_bar(tmp_path):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"accuracy": 0.5}), encoding="utf-8")
    entry = stats._seeded(path, "accuracy", "accuracy")
    assert "seed_std" not in entry
    assert "no per-repeat values" in entry["seed_variance"]


def test_a_missing_artefact_is_a_note_not_a_number(tmp_path, monkeypatch):
    monkeypatch.setattr(stats, "ROOT", tmp_path)
    assert "note" in stats.s5_interval(10)
    assert "note" in stats.end_to_end_interval(10)


def test_render_separates_sampling_from_seed_uncertainty():
    result = {
        "protocol": {"sampling": "percentile bootstrap, 10 resamples", "seeds": "read"},
        "s1_classification": {
            "metric": "a",
            "point": 0.9,
            "target": 0.92,
            "seeds": 3,
            "seed_std": 0.001,
            "seed_min": 0.89,
            "seed_max": 0.91,
        },
        "s4_roles": {"metric": "b", "point": 0.8, "target": 0.8, "seed_variance": "single run"},
        "s3_ocr": {
            "metric": "c",
            "target": 0.15,
            "cer": {"point": 0.2, "ci95": [0.19, 0.21], "resamples": 400, "items": 2835},
            "seed_variance": "single run",
        },
        "s5_assembly": {
            "metric": "d",
            "target": 3.0,
            "median_ged": {"point": 13.0, "ci95": [9.0, 16.0]},
            "seed_variance": "single run",
        },
        "end_to_end": {
            "gold": {"point": 0.78, "ci95": [0.72, 0.85], "half_width": 0.06},
            "gold_structure": {"point": 0.65, "ci95": [0.58, 0.73], "half_width": 0.07},
            "gold_text": {"point": 0.23, "ci95": [0.17, 0.3], "half_width": 0.06},
            "predicted": {"point": 0.19, "ci95": [0.14, 0.25], "half_width": 0.06},
            "seed_variance": "deterministic",
        },
    }
    text = stats.render(result)
    assert "3 seeds" in text
    assert "single run" in text
    assert "do not overlap" in text
    assert "400 resamples" in text


def test_render_reports_overlapping_intervals_as_overlapping():
    result = {
        "protocol": {"sampling": "s", "seeds": "r"},
        "s1_classification": {},
        "s4_roles": {},
        "s3_ocr": {},
        "s5_assembly": {},
        "end_to_end": {
            "gold": {"point": 0.30, "ci95": [0.20, 0.40], "half_width": 0.1},
            "predicted": {"point": 0.25, "ci95": [0.15, 0.35], "half_width": 0.1},
        },
    }
    assert "overlap, so" in stats.render(result)
