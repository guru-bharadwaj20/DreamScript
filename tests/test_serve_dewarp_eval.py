"""Phase 16.2.3 - the dewarp measurement, and the two ways its artefact could be a lie.

The detector itself is tested in `app/frontend/src/lib/dewarp.test.ts` - 23 cases against synthetic
pages with known corners, in the language it is written in. This file is about the *measurement*:
the arithmetic that turns three pipeline runs into a comparison, and the committed report.

The measurement had a defect worth pinning. Its first run scored **every** dewarped page at node F1
0.00, because `geometric` matching is IoU over pixel coordinates and the dewarped arm is by
construction a different crop at a different size: it was being penalised for doing exactly what it
exists to do. `test_boxes_are_normalised_before_they_are_compared` is that regression.
"""

from __future__ import annotations

import json

import pytest

from src.serve import dewarp_eval
from src.utils.config import ROOT

REPORT = ROOT / "reports" / "dewarp.json"
MARKDOWN = ROOT / "reports" / "dewarp.md"


# == the arithmetic ===========================================================================


def test_boxes_are_normalised_before_they_are_compared():
    """The regression. A crop is not a different answer, and the metric must not read it as one."""
    ir = {"nodes": [{"id": "a", "bbox": [50, 25, 100, 50]}], "edges": []}
    out = dewarp_eval.normalise(ir, 200, 100)
    assert out["nodes"][0]["bbox"] == [0.25, 0.25, 0.5, 0.5]
    # The input is not mutated: the same IR is normalised against two different sizes in one sweep.
    assert ir["nodes"][0]["bbox"] == [50, 25, 100, 50]


def test_the_same_page_at_two_crops_normalises_to_the_same_boxes():
    """Which is the whole point: a page occupying a 400x300 frame and the same page cropped to
    200x150 put their boxes in the same place once each is divided by its own image."""
    big = {"nodes": [{"id": "a", "bbox": [100, 75, 200, 150]}], "edges": []}
    small = {"nodes": [{"id": "a", "bbox": [50, 37.5, 100, 75]}], "edges": []}
    assert dewarp_eval.normalise(big, 400, 300) == dewarp_eval.normalise(small, 200, 150)


def test_a_node_with_no_box_survives_normalisation():
    ir = {"nodes": [{"id": "a"}, {"id": "b", "bbox": None}], "edges": []}
    assert dewarp_eval.normalise(ir, 100, 100)["nodes"] == ir["nodes"]


def test_normalising_nothing_is_nothing():
    assert dewarp_eval.normalise(None, 100, 100) is None
    assert dewarp_eval.normalise({"nodes": []}, 0, 0) == {"nodes": []}


def test_png_size_is_read_from_the_header_without_a_decoder():
    """The report has to know each arm's dimensions and has no business importing OpenCV to
    find out."""
    width, height = dewarp_eval.image_size(ROOT / "tests" / "fixtures" / "flowchart.png")
    assert (width, height) == (320, 240)


def test_a_file_that_is_not_a_png_is_refused_rather_than_misread():
    with pytest.raises(ValueError):
        dewarp_eval.image_size(ROOT / "tests" / "fixtures" / "manifest.json")


def test_a_missing_arm_scores_nothing_rather_than_zero():
    """A page the pipeline could not read has no graph. Scoring that as 0.0 would put it in the
    median beside genuine misses and make the two indistinguishable."""
    assert dewarp_eval.compare(None, {"nodes": []}) is None
    assert dewarp_eval.compare({"nodes": []}, None) is None


def test_the_matching_rule_is_named_in_the_artefact():
    """10.2.5's own warning: a node F1 quoted without its matching rule is not reproducible."""
    assert dewarp_eval.MATCH in ("geometric", "text", "combined", "hungarian")


# == the committed report =====================================================================


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve dewarp_eval` first")
def test_the_report_says_its_skew_is_synthetic():
    """This repository has no skewed photographs. A report that did not say so would be read as
    an answer about real phone photographs, which it is not."""
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    assert held["synthetic_skew"] is True
    text = MARKDOWN.read_text(encoding="utf-8")
    assert "synthetic" in text.lower()
    assert "not an answer about real phone photographs" in text


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve dewarp_eval` first")
def test_the_report_states_what_the_correction_costs_as_well_as_what_it_buys():
    """A dewarp that improves the median while losing a page outright is a trade. The report is
    required to say so where the trade exists, not to bury it in a table."""
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    reached = held["summary"]["reached_code"]
    text = MARKDOWN.read_text(encoding="utf-8")
    if reached["dewarped"] < reached["photo"]:
        assert "costs something" in text, "the dewarp lost a page and the report does not say so"


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve dewarp_eval` first")
def test_straightening_beats_not_straightening_on_the_median():
    """The row's claim, as a number. If this ever fails, `mandatory, not optional` is wrong."""
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    photo = held["summary"]["photo_vs_flat"]
    dewarped = held["summary"]["dewarped_vs_flat"]
    assert dewarped["node_f1_median"] > photo["node_f1_median"]
    assert dewarped["ged_normalised_median"] < photo["ged_normalised_median"]


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve dewarp_eval` first")
def test_the_report_agrees_with_its_own_rows():
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    rows = held["rows"]
    assert held["summary"]["pages"] == len(rows)
    assert held["summary"]["detected"] == sum(1 for r in rows if r["detected"])
    for arm in ("source", "photo", "dewarped"):
        counted = sum(1 for r in rows if r["runs"].get(arm, {}).get("ok"))
        assert held["summary"]["reached_code"][arm] == counted, arm


@pytest.mark.skipif(not REPORT.is_file(), reason="run `python -m src.serve dewarp_eval` first")
def test_the_markdown_is_generated_from_the_json():
    held = json.loads(REPORT.read_text(encoding="utf-8"))
    assert MARKDOWN.read_text(encoding="utf-8").strip() == dewarp_eval.markdown(held).strip()
