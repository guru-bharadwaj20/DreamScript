"""Phase 7.2.1 - the text-only feature table, and the shape features it must not touch."""

from __future__ import annotations

import numpy as np
import pytest

from src.features import textregions


class FakeContext:
    """A PageContext with only the fields the text table is allowed to read.

    Deliberately missing `regions`, `segments`, `arrowheads` and `skeleton`: if `extract` ever
    reaches for one of them, these tests raise `AttributeError` rather than quietly succeeding
    on a table that is no longer the cheap one.
    """

    def __init__(self, boxes, height=400, width=600, text_px=1000, ink_px=5000):
        self.id = "fake"
        self.height = height
        self.width = width
        self.text_boxes = list(boxes)
        self.mask = np.zeros((height, width), dtype=bool)
        self.mask.flat[:ink_px] = True
        self.text_mask = np.zeros((height, width), dtype=bool)
        self.text_mask.flat[:text_px] = True

    @property
    def page_area(self):
        return self.height * self.width

    @property
    def long_side(self):
        return max(self.height, self.width)


@pytest.fixture(autouse=True)
def blocks_are_the_boxes(monkeypatch):
    """4.1.6's grouping needs the structural module; the table's own logic is what is tested."""
    from src.features import textstats

    monkeypatch.setattr(textstats, "blocks", lambda context: list(context.text_boxes))


# -- the table is cheap by construction -----------------------------------------------------------


def test_it_declares_fourteen_columns():
    assert len(textregions.NAMES) == 14
    assert len(set(textregions.NAMES)) == 14


def test_every_column_name_is_namespaced():
    assert all(name.startswith("text_") for name in textregions.NAMES)


def test_extraction_never_touches_a_shape_region():
    """The whole point of the table: no region detection, no segments, no skeleton."""
    context = FakeContext([(10, 10, 50, 20), (100, 12, 60, 22)])
    result = textregions.extract(context)
    assert set(result) == set(textregions.NAMES)
    assert not hasattr(context, "regions")


def test_the_strongest_phase_4_text_feature_is_deliberately_absent():
    """`text_inside_share` needs `context.regions`, which is the cost being avoided."""
    assert "text_inside_share" not in textregions.NAMES


# -- the empty page --------------------------------------------------------------------------------


def test_a_page_with_no_text_reports_zero_counts_not_nan():
    """ "No writing" is a fact about the page."""
    result = textregions.extract(FakeContext([]))
    assert result["text_n_blocks"] == 0.0
    assert result["text_blocks_per_area"] == 0.0
    assert result["text_coverage"] == 0.0


def test_a_page_with_no_text_reports_nan_for_statistics_of_the_blocks():
    """The mean width of no labels is not zero, and 4.2.3's indicators record which pages."""
    result = textregions.extract(FakeContext([]))
    for name in ("text_width_mean", "text_aspect_mean", "text_largest_share"):
        assert np.isnan(result[name])


def test_a_page_with_one_block_has_no_neighbour_distance():
    """One block cannot be near anything or aligned with anything - nan, not zero."""
    result = textregions.extract(FakeContext([(10, 10, 40, 20)]))
    assert np.isnan(result["text_nn_distance"])
    assert np.isnan(result["text_row_alignment"])
    assert result["text_n_blocks"] == 1.0


# -- the columns measure what they claim -------------------------------------------------------------


def test_the_block_count_is_the_number_of_labels():
    boxes = [(10 * i, 10, 20, 10) for i in range(7)]
    assert textregions.extract(FakeContext(boxes))["text_n_blocks"] == 7.0


def test_widths_are_normalised_by_the_long_side():
    """A 4000px photograph and a 1000px render of the same page must agree."""
    small = textregions.extract(
        FakeContext([(0, 0, 60, 20), (100, 0, 60, 20)], height=400, width=600)
    )
    big = textregions.extract(
        FakeContext([(0, 0, 120, 40), (200, 0, 120, 40)], height=800, width=1200)
    )
    assert small["text_width_mean"] == pytest.approx(big["text_width_mean"], abs=1e-6)


def test_uniform_labels_have_a_smaller_width_spread_than_varied_ones():
    uniform = textregions.extract(FakeContext([(0, 0, 50, 20), (100, 0, 50, 20)]))
    varied = textregions.extract(FakeContext([(0, 0, 10, 20), (100, 0, 200, 20)]))
    assert uniform["text_width_std"] < varied["text_width_std"]


def test_a_long_thin_block_has_a_larger_aspect_than_a_square_one():
    wide = textregions.extract(FakeContext([(0, 0, 200, 20)]))
    square = textregions.extract(FakeContext([(0, 0, 20, 20)]))
    assert wide["text_aspect_mean"] > square["text_aspect_mean"]


def test_row_alignment_is_one_when_every_block_shares_a_band():
    boxes = [(0, 100, 40, 10), (100, 101, 40, 10), (200, 100, 40, 10)]
    assert textregions.extract(FakeContext(boxes))["text_row_alignment"] == pytest.approx(1.0)


def test_row_alignment_is_zero_when_no_two_blocks_share_a_band():
    boxes = [(0, 0, 40, 10), (0, 150, 40, 10), (0, 300, 40, 10)]
    assert textregions.extract(FakeContext(boxes))["text_row_alignment"] == pytest.approx(0.0)


def test_clustered_labels_have_a_smaller_nearest_neighbour_distance():
    close = textregions.extract(FakeContext([(0, 0, 10, 10), (12, 0, 10, 10)]))
    far = textregions.extract(FakeContext([(0, 0, 10, 10), (500, 380, 10, 10)]))
    assert close["text_nn_distance"] < far["text_nn_distance"]


def test_the_edge_share_counts_blocks_in_the_margin():
    middle = textregions.extract(FakeContext([(290, 190, 20, 20)]))
    corner = textregions.extract(FakeContext([(2, 2, 10, 10)]))
    assert middle["text_edge_share"] == pytest.approx(0.0)
    assert corner["text_edge_share"] == pytest.approx(1.0)


def test_the_largest_share_is_one_for_a_single_block():
    assert textregions.extract(FakeContext([(0, 0, 30, 30)]))["text_largest_share"] == 1.0


def test_the_largest_share_falls_as_blocks_even_out():
    dominated = textregions.extract(FakeContext([(0, 0, 200, 100), (300, 0, 10, 10)]))
    even = textregions.extract(FakeContext([(0, 0, 50, 50), (300, 0, 50, 50)]))
    assert dominated["text_largest_share"] > even["text_largest_share"]


def test_the_ink_share_is_text_pixels_over_all_ink():
    context = FakeContext([(0, 0, 30, 30)], text_px=1000, ink_px=4000)
    assert textregions.extract(context)["text_ink_share"] == pytest.approx(0.25)


def test_every_value_is_finite_or_nan_never_inf():
    context = FakeContext([(0, 0, 30, 30), (100, 0, 30, 30)])
    for value in textregions.extract(context).values():
        assert not np.isinf(value)


# -- the loader ----------------------------------------------------------------------------------------


def test_the_loader_refuses_a_missing_table(monkeypatch, tmp_path):
    monkeypatch.setattr(textregions, "TABLE", tmp_path / "absent.parquet")
    with pytest.raises(FileNotFoundError, match="no text-region table"):
        textregions.load()
