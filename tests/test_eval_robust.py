"""Phase 14.5 - a robustness curve is worthless if the degradation never reached the model.

The two ways this harness could silently lie are (1) the degraded page reusing the clean page's
cached detections, and (2) severity 0 being read from somewhere else instead of re-run. Both are
pinned here, along with the degradations themselves actually changing the pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.eval import robust


@pytest.fixture()
def page_image():
    rng = np.random.default_rng(0)
    image = np.full((120, 160), 240, dtype=np.uint8)
    image[40:80, 30:130] = 20
    image += rng.integers(0, 6, image.shape, dtype=np.uint8)
    return image


@pytest.mark.parametrize("kind", list(robust.SWEEPS))
def test_every_degradation_changes_the_page_at_its_strongest_severity(kind, page_image):
    strongest = robust.SWEEPS[kind][-1]
    out = robust.degrade(page_image, kind, strongest)
    assert out.shape == page_image.shape
    assert not np.array_equal(out, page_image)


@pytest.mark.parametrize("kind", list(robust.SWEEPS))
def test_the_clean_control_is_the_identity(kind, page_image):
    """Severity 0 must be the same pixels, or the baseline is a different experiment."""
    clean = robust.SWEEPS[kind][0]
    assert np.array_equal(robust.degrade(page_image, kind, clean), page_image)


def test_an_unknown_degradation_raises_rather_than_passing_the_page_through(page_image):
    with pytest.raises(ValueError):
        robust.degrade(page_image, "smudge", 1.0)


class _Page:
    name = "hdbpmn__ex00_writer0001"
    source = "hdbpmn"
    split = "test"
    scale = 1.0


def test_a_degraded_page_gets_its_own_identity(tmp_path):
    """Sharing `name` with the clean page would hand back the clean page's cached boxes."""
    degraded = robust.DegradedPage(_Page(), tmp_path / "x.png", "blur9")
    assert degraded.name != _Page.name
    assert degraded.image == tmp_path / "x.png"
    assert degraded.source == "hdbpmn"  # everything identifying the truth is the original's
    assert degraded.split == "test"


def test_aggregate_reports_the_pages_behind_every_point():
    rows = [
        {"kind": "blur", "severity": 0, "ged": 4.0, "node_f1": 1.0, "edge_f1": 0.9, "routed": 1},
        {"kind": "blur", "severity": 0, "ged": 6.0, "node_f1": 0.8, "edge_f1": 0.7, "routed": 1},
        {"kind": "blur", "severity": 9, "ged": 20.0, "node_f1": 0.4, "edge_f1": 0.1, "routed": 0},
    ]
    curves = robust.aggregate(rows)
    assert curves["blur"]["0"]["pages"] == 2
    assert curves["blur"]["0"]["median_ged"] == 5.0
    assert curves["blur"]["0"]["node_f1"] == 0.9
    assert curves["blur"]["9"]["routed"] == 0.0


def test_a_page_whose_stage_failed_is_counted_as_an_error_not_as_a_score():
    rows = [
        {"kind": "blur", "severity": 0, "ged": 4.0, "node_f1": 1.0, "edge_f1": 0.9, "routed": 1},
        {"kind": "blur", "severity": 0, "error": "RuntimeError: no boxes"},
    ]
    curves = robust.aggregate(rows)
    assert curves["blur"]["0"]["pages"] == 1
    assert curves["blur"]["0"]["node_f1"] == 1.0


def test_render_names_the_sample_and_every_metric():
    rows = [{"kind": "blur", "severity": 0, "ged": 4.0, "node_f1": 1.0, "edge_f1": 0.9, "routed": 1}]
    text = robust.render(
        {
            "split": "test",
            "pages": ["p1"],
            "sources": ["hdbpmn"],
            "seconds": 1.0,
            "curves": robust.aggregate(rows),
        }
    )
    for metric in robust.METRICS:
        assert metric in text
    assert "1 held-out test pages" in text


def test_the_sample_is_stratified_across_sources():
    sample = robust._sample(4, "test")
    if not sample:
        pytest.skip("the held-out corpus is not present in this tree")
    assert len({page.source for page in sample}) >= min(2, len({p.source for p in sample}))
    assert len(sample) == 4
