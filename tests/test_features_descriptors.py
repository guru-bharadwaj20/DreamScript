"""Phase 7.4.1 - the shape descriptor, and the invariances it claims."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.descriptors import NAMES, describe, fourier

INDEX = {name: i for i, name in enumerate(NAMES)}

SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
DIAMOND = np.array([[50, 0], [100, 50], [50, 100], [0, 50]])
WIDE = np.array([[0, 0], [200, 0], [200, 50], [0, 50]])
CIRCLE = np.array(
    [
        [int(500 + 400 * np.cos(t)), int(500 + 400 * np.sin(t))]
        for t in np.linspace(0, 2 * np.pi, 96, endpoint=False)
    ]
)


def column(contour, name):
    return describe(contour)[INDEX[name]]


# -- the vector --------------------------------------------------------------------------------------


def test_the_descriptor_has_one_value_per_declared_column():
    assert describe(SQUARE).shape == (len(NAMES),)


def test_the_column_names_are_unique():
    assert len(set(NAMES)) == len(NAMES)


def test_every_column_is_finite_for_every_shape():
    for contour in (SQUARE, DIAMOND, WIDE, CIRCLE):
        assert np.isfinite(describe(contour)).all()


@pytest.mark.parametrize("contour", [np.array([[0, 0], [1, 1], [2, 2]]), np.array([[0, 0]])])
def test_a_contour_too_small_to_describe_returns_none(contour):
    assert describe(contour) is None


def test_a_zero_area_contour_returns_none_rather_than_dividing_by_it():
    assert describe(np.array([[0, 0], [10, 0], [20, 0], [30, 0]])) is None


# -- what the columns mean ---------------------------------------------------------------------------


def test_a_circle_is_rounder_than_a_square():
    assert column(CIRCLE, "circularity") > column(SQUARE, "circularity")


def test_a_circle_is_nearly_perfectly_circular():
    assert column(CIRCLE, "circularity") > 0.95


def test_a_quadrilateral_has_four_vertices():
    assert column(SQUARE, "vertices") == 4
    assert column(DIAMOND, "vertices") == 4


def test_a_circle_has_more_vertices_than_a_quadrilateral():
    assert column(CIRCLE, "vertices") > 4


def test_the_aspect_is_of_the_rotated_rectangle():
    assert column(WIDE, "rect_aspect") == pytest.approx(4.0, abs=0.1)
    assert column(SQUARE, "rect_aspect") == pytest.approx(1.0, abs=0.05)


def test_a_convex_shape_is_fully_solid():
    assert column(SQUARE, "solidity") == pytest.approx(1.0, abs=0.01)


def test_a_dented_shape_is_less_solid_and_has_a_defect():
    arrow = np.array([[0, 0], [100, 50], [0, 100], [30, 50]])
    assert column(arrow, "solidity") < 0.8
    assert column(arrow, "defect_count") >= 1
    assert column(arrow, "defect_max") > 0


def test_a_convex_shape_has_no_defects():
    assert column(SQUARE, "defect_count") == 0
    assert column(SQUARE, "defect_max") == 0.0


# -- the invariances that are taken -------------------------------------------------------------------


def test_the_descriptor_is_scale_invariant():
    """Nothing in the vector may carry the size of the drawing."""
    assert np.allclose(describe(SQUARE), describe(SQUARE * 7), atol=0.05)


def test_the_fourier_magnitudes_do_not_depend_on_where_the_contour_starts():
    """`findContours` starts wherever it starts; keeping phase would encode that accident."""
    rolled = np.roll(SQUARE, 2, axis=0)
    assert np.allclose(fourier(SQUARE), fourier(rolled), atol=1e-9)


def test_the_fourier_spectrum_does_not_depend_on_how_densely_the_contour_is_stored():
    """Arc-length resampling: the same square as 4 points and as 400 must agree."""
    dense = np.array(
        [
            SQUARE[i] + (SQUARE[(i + 1) % 4] - SQUARE[i]) * f
            for i in range(4)
            for f in np.linspace(0, 1, 100, endpoint=False)
        ]
    )
    assert np.allclose(fourier(SQUARE), fourier(dense), atol=0.02)


def test_the_fourier_block_is_the_declared_width():
    assert len(fourier(SQUARE)) == len([n for n in NAMES if n.startswith("fourier_")])


def test_a_degenerate_contour_gives_zeros_rather_than_nan():
    assert np.isfinite(fourier(np.array([[5, 5], [5, 5], [5, 5], [5, 5]]))).all()


# -- the invariance that is deliberately not taken ----------------------------------------------------


def test_the_rotation_invariant_columns_cannot_tell_a_diamond_from_a_square():
    """The documented blind spot, asserted so it stays documented."""
    for name in ("rect_fill", "solidity", "circularity", "rect_aspect", "vertices"):
        assert column(SQUARE, name) == pytest.approx(column(DIAMOND, name), abs=1e-3)


def test_the_hu_moments_cannot_tell_a_diamond_from_a_square_either():
    square, diamond = describe(SQUARE), describe(DIAMOND)
    hu = [INDEX[f"hu_{i}"] for i in range(7)]
    assert np.allclose(square[hu], diamond[hu], atol=1e-3)


def test_extent_is_the_column_that_separates_them():
    assert column(SQUARE, "extent") == pytest.approx(1.0, abs=0.03)
    assert column(DIAMOND, "extent") == pytest.approx(0.5, abs=0.03)


def test_the_hu_moments_still_separate_shapes_that_genuinely_differ():
    """The symmetric log must not have flattened the signal along with the noise."""
    hu = [INDEX[f"hu_{i}"] for i in range(7)]
    assert np.abs(describe(SQUARE)[hu] - describe(CIRCLE)[hu]).max() > 1.0


def test_a_symmetric_shapes_near_zero_moments_do_not_explode():
    """A signed log of a numerically-zero moment turns sign noise into a swing of tens; that
    was measured at 60.0 between a square and the same square rotated, and is what the
    symmetric log exists to prevent."""
    hu = [INDEX[f"hu_{i}"] for i in range(7)]
    assert np.abs(describe(SQUARE)[hu]).max() < 30
