"""Phase 9.1.5 - the tiling geometry and the merge, which are what can be wrong silently."""

from __future__ import annotations

import numpy as np
import pytest

from src.detect import tiling

# -- the window grid ------------------------------------------------------------------------


def test_the_windows_cover_the_whole_page():
    width, height, tile = 1500, 900, 640
    covered = np.zeros((height, width), dtype=bool)
    for x, y in tiling.tiles(width, height, tile, 0.25):
        covered[y : y + tile, x : x + tile] = True
    assert covered.all()


def test_the_last_window_is_pinned_to_the_page_edge():
    """Otherwise the right and bottom strips are covered by fewer windows than the middle."""
    windows = tiling.tiles(1500, 900, 640, 0.25)
    assert max(x for x, _ in windows) == 1500 - 640
    assert max(y for _, y in windows) == 900 - 640


def test_a_page_smaller_than_a_tile_yields_one_window():
    assert tiling.tiles(300, 200, 640, 0.25) == [(0, 0)]


def test_more_overlap_means_more_windows():
    assert len(tiling.tiles(2000, 1500, 640, 0.5)) > len(tiling.tiles(2000, 1500, 640, 0.1))


def test_every_object_under_the_overlap_width_is_whole_in_some_window():
    """The property the overlap buys: a 15px arrowhead is never only seen in halves."""
    width, height, tile, overlap = 2000, 1400, 640, 0.25
    windows = tiling.tiles(width, height, tile, overlap)
    rng = np.random.default_rng(0)
    for _ in range(200):
        size = 40
        x = int(rng.integers(0, width - size))
        y = int(rng.integers(0, height - size))
        assert any(
            wx <= x and wy <= y and x + size <= wx + tile and y + size <= wy + tile
            for wx, wy in windows
        )


# -- the merge ------------------------------------------------------------------------------


def test_nms_keeps_the_higher_scoring_of_two_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=float)
    scores = np.array([0.6, 0.9])
    assert tiling.nms(boxes, scores, 0.5) == [1]


def test_nms_keeps_both_when_they_barely_touch():
    boxes = np.array([[0, 0, 10, 10], [9, 9, 19, 19]], dtype=float)
    assert sorted(tiling.nms(boxes, np.array([0.9, 0.8]), 0.5)) == [0, 1]


def test_nms_on_a_single_box_returns_it():
    assert tiling.nms(np.array([[0, 0, 5, 5]], dtype=float), np.array([0.5]), 0.5) == [0]


def test_the_merge_is_per_class_so_a_nested_pair_survives():
    """A BPMN pool contains its tasks; merging across classes would delete one of them."""
    raw = [
        ([0.0, 0.0, 100.0, 100.0], 0.9, 0),  # rectangle, the pool
        ([10.0, 10.0, 40.0, 40.0], 0.8, 1),  # rounded-rect, a task inside it
    ]
    merged = tiling._merge("page", raw)
    assert {row["cls"] for row in merged} == {"rectangle", "rounded-rect"}


def test_the_merge_collapses_the_seam_duplicate():
    raw = [
        ([0.0, 0.0, 20.0, 20.0], 0.9, 7),
        ([1.0, 1.0, 21.0, 21.0], 0.7, 7),
    ]
    merged = tiling._merge("page", raw)
    assert len(merged) == 1
    assert merged[0]["score"] == pytest.approx(0.9)


def test_the_merge_keeps_the_image_name_on_every_box():
    raw = [([0.0, 0.0, 10.0, 10.0], 0.5, 0)]
    assert tiling._merge("hdbpmn__x", raw)[0]["image"] == "hdbpmn__x"
