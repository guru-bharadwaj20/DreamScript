"""Phase 3.3.3 - the failure gallery.

The corpus ranking is produced by `python -m src.preprocess.failures`; these tests pin the
signals, the diagnosis rule and the report, on constructed cases where the answer is known.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocess import failures as fl


def case(**overrides) -> fl.Case:
    fields = {
        "id": "page",
        "image": "data/raw/page.png",
        "empty_boxes": 0.0,
        "ink_share": 0.05,
        "fragmentation": 2.0,
        "shadow_left": 8.0,
        "crop_loss": 0.0,
        "boxes": 10,
    }
    fields.update(overrides)
    return fl.Case(**fields)


def test_a_healthy_page_fires_nothing():
    name, score, _ = fl.diagnose(case())
    assert name == "clean" and score == 0.0


def test_a_lost_shape_is_the_worst_signal():
    name, _, _ = fl.diagnose(case(empty_boxes=0.4, shadow_left=22.0))
    assert name == "ink_lost"


def test_each_signal_can_win():
    assert fl.diagnose(case(ink_share=0.001))[0] == "threshold_too_high"
    assert fl.diagnose(case(ink_share=0.30))[0] == "background_survived"
    assert fl.diagnose(case(fragmentation=12.0))[0] == "fragmented"
    assert fl.diagnose(case(shadow_left=50.0))[0] == "shadow_left"
    assert fl.diagnose(case(crop_loss=0.5))[0] == "crop_lost_content"
    assert fl.diagnose(case(off_page_ink=0.8))[0] == "off_page_background"


def test_the_score_is_the_worst_signal_not_the_sum():
    """One bad failure must outrank several mild oddities."""
    one_bad = case(fragmentation=20.0)
    several_mild = case(ink_share=0.18, shadow_left=21.0, fragmentation=5.2, crop_loss=0.05)
    assert fl.diagnose(one_bad)[1] > fl.diagnose(several_mild)[1]


def test_rule_residue_separates_a_grid_from_handwriting():
    grid = np.zeros((600, 600), np.uint8)
    for position in range(0, 600, 30):
        cv2.line(grid, (position, 0), (position, 599), 255, 2)
        cv2.line(grid, (0, position), (599, position), 255, 2)
    scribble = np.zeros((600, 600), np.uint8)
    rng = np.random.default_rng(0)
    for _ in range(60):
        x, y = rng.integers(50, 550, 2)
        cv2.circle(scribble, (int(x), int(y)), 8, 255, 2)
    assert fl.rule_residue(grid > 0) > 0.8
    assert fl.rule_residue(scribble > 0) < 0.1


def test_rule_residue_of_an_empty_mask_is_zero():
    assert fl.rule_residue(np.zeros((100, 100), bool)) == 0.0


def test_every_diagnosis_has_an_explanation():
    for name in fl.signals(case()):
        explained = fl._explain(case(diagnosis=name))
        assert explained, f"no explanation written for {name}"
    assert fl._explain(case(diagnosis="clean"))


def test_the_grid_explanation_only_appears_when_the_residue_says_so():
    grid_page = case(diagnosis="background_survived", ink_share=0.25, rule_residue=0.33)
    plain_page = case(diagnosis="background_survived", ink_share=0.25, rule_residue=0.02)
    assert "squared paper" in fl._explain(grid_page)
    assert "squared paper" not in fl._explain(plain_page)


def test_report_lists_every_gallery_case(tmp_path, monkeypatch):
    monkeypatch.setattr(fl, "REPORT", tmp_path / "preproc_failures.md")
    cases = [case(id=f"page{i}", fragmentation=6.0 + i) for i in range(5)]
    for item in cases:
        item.diagnosis, item.score, item.signals = fl.diagnose(item)
    path = fl.report(cases, cases[:3])
    text = path.read_text(encoding="utf-8")
    for item in cases[:3]:
        assert item.id in text
    assert "fragmented" in text
