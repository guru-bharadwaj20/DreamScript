"""Phase 4.1.4 - layout geometry."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.features import context as ctx
from src.features import layout as ly


def grid(rows: int = 3, columns: int = 3, scale: int = 1) -> np.ndarray:
    """A regular grid of boxes, drawable at any size for the scale-invariance tests."""
    image = np.zeros((240 * rows * scale, 240 * columns * scale), np.uint8)
    for r in range(rows):
        for c in range(columns):
            x, y = 40 * scale + 240 * c * scale, 40 * scale + 240 * r * scale
            cv2.rectangle(image, (x, y), (x + 160 * scale, y + 160 * scale), 255, 3 * scale)
    return image > 0


def scattered() -> np.ndarray:
    image = np.zeros((720, 720), np.uint8)
    for x, y in [(40, 40), (330, 180), (110, 420), (500, 560), (560, 90)]:
        cv2.rectangle(image, (x, y), (x + 120, y + 100), 255, 3)
    return image > 0


def test_all_five_names_are_produced():
    assert set(ly.extract(ctx.from_masks(grid(), page_id="grid"))) == set(ly.NAMES)


def test_nothing_changes_when_the_same_drawing_is_photographed_larger():
    small = ly.extract(ctx.from_masks(grid(scale=1), page_id="small"))
    large = ly.extract(ctx.from_masks(grid(scale=2), page_id="large"))
    for name in ly.NAMES:
        assert small[name] == pytest.approx(large[name], abs=0.05), name


def test_a_grid_scores_higher_than_a_scatter():
    aligned = ly.extract(ctx.from_masks(grid(), page_id="grid"))["layout_grid_score"]
    loose = ly.extract(ctx.from_masks(scattered(), page_id="scatter"))["layout_grid_score"]
    assert aligned == 1.0
    assert loose < aligned


def test_a_vertical_chain_is_a_column_and_scores_as_regular():
    """The finding in the docstring: this measures arrangement, not grid-ness."""
    image = np.zeros((900, 400), np.uint8)
    for i in range(4):
        cv2.rectangle(image, (120, 40 + 220 * i), (280, 180 + 220 * i), 255, 3)
    features = ly.extract(ctx.from_masks(image > 0, page_id="chain"))
    assert features["layout_grid_score"] == 1.0
    assert features["layout_row_regularity"] == pytest.approx(1.0, abs=0.05)


def test_regularity_is_undefined_below_three_lines():
    """Two rows have one gap, and one gap has no variation to measure."""
    assert np.isnan(ly._regularity([0.1, 0.5]))
    assert ly._regularity([0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_irregular_spacing_scores_below_regular_spacing():
    assert ly._regularity([0.0, 0.1, 0.9]) < ly._regularity([0.0, 0.4, 0.8])


def test_an_empty_page_is_nan_everywhere():
    features = ly.extract(ctx.from_masks(np.zeros((300, 300), bool), page_id="blank"))
    assert all(np.isnan(v) for v in features.values())


def test_nearest_neighbour_distance_ignores_the_shape_itself():
    points = np.array([[0.0, 0.0], [0.3, 0.0], [0.9, 0.0]])
    assert ly.nearest_neighbour_distances(points).tolist() == pytest.approx([0.3, 0.3, 0.6])


def test_grouping_is_single_linkage_within_the_tolerance():
    values = np.array([0.10, 0.11, 0.50])
    assert [len(g) for g in ly._groups(values, 0.02)] == [2, 1]
    assert [len(g) for g in ly._groups(values, 0.001)] == [1, 1, 1]
