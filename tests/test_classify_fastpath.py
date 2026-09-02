"""Phase 7.2 - the text fast path, timed against the embedding path stage by stage."""

from __future__ import annotations

import pytest

from src.classify import fastpath


def rows(*totals):
    """Rows whose four stages sum to the given end-to-end totals."""
    return [dict(zip(fastpath.STAGES, (t / 4,) * 4, strict=True)) for t in totals]


def test_the_four_stages_are_the_ones_being_compared():
    assert fastpath.STAGES == ("decode", "preprocess", "features", "predict")


def test_an_empty_run_reports_zero_pages_rather_than_raising():
    assert fastpath.summarise([], "text") == {"path": "text", "pages": 0}


def test_the_path_name_is_carried_through():
    assert fastpath.summarise(rows(4.0), "embedding")["path"] == "embedding"


def test_the_page_count_is_the_row_count():
    assert fastpath.summarise(rows(1.0, 2.0, 3.0), "text")["pages"] == 3


def test_the_end_to_end_total_is_the_sum_of_the_stages():
    assert fastpath.summarise(rows(8.0), "text")["end_to_end_median_ms"] == pytest.approx(8.0)


def test_every_stage_gets_a_median_and_a_p95():
    summary = fastpath.summarise(rows(4.0, 8.0), "text")
    assert set(summary["median_ms"]) == set(fastpath.STAGES)
    assert set(summary["p95_ms"]) == set(fastpath.STAGES)


def test_the_median_ignores_the_one_page_that_hit_a_collection():
    """Mean would let a single outlier define the latency of a request handler."""
    summary = fastpath.summarise(rows(4.0, 4.0, 4.0, 4.0, 400.0), "text")
    assert summary["end_to_end_median_ms"] == pytest.approx(4.0)


def test_the_p95_is_the_page_that_does_define_the_tail():
    summary = fastpath.summarise(rows(4.0, 4.0, 4.0, 4.0, 400.0), "text")
    assert summary["end_to_end_p95_ms"] > summary["end_to_end_median_ms"]


def test_predict_share_is_a_fraction_of_the_whole():
    summary = fastpath.summarise(rows(8.0), "text")
    assert summary["predict_share_of_total"] == pytest.approx(0.25)


def test_a_dominant_predict_stage_pushes_the_share_toward_one():
    heavy = [{"decode": 0.1, "preprocess": 0.1, "features": 0.1, "predict": 99.7}]
    assert fastpath.summarise(heavy, "text")["predict_share_of_total"] > 0.99


def test_the_page_budget_is_declared():
    assert fastpath.PAGES > 0
