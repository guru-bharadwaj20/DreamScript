"""Phase 7.3.12 - nine raw measurements per node, against the 53 symbols they were quantized to."""

from __future__ import annotations

import numpy as np
import pytest

from src.parse import continuous

FEATURES = (
    "width",
    "height",
    "log_aspect",
    "area",
    "in_degree",
    "out_degree",
    "text_len",
    "x",
    "y",
)


def page(nodes, edges=()):
    return {
        "id": "d",
        "diagram_type": "flowchart",
        "nodes": nodes,
        "edges": [{"id": f"e{i}", "src": s, "dst": d} for i, (s, d) in enumerate(edges)],
    }


def box(node_id, x, y, w, h, text=""):
    return {"id": node_id, "shape": "rectangle", "bbox": [x, y, w, h], "text": text}


# -- the feature row ---------------------------------------------------------------------------------


def test_there_is_one_row_per_node():
    rows = continuous.features(page([box("a", 0, 0, 10, 10), box("b", 0, 20, 10, 10)]))
    assert set(rows) == {"a", "b"}


def test_the_row_has_the_nine_declared_dimensions():
    rows = continuous.features(page([box("a", 0, 0, 10, 10)]))
    assert rows["a"].shape == (len(FEATURES),)


def test_a_page_with_no_nodes_gives_no_rows():
    assert continuous.features(page([])) == {}


def test_degrees_are_the_counts_themselves_not_a_four_way_class():
    """The whole premise: the discrete model cannot say in-degree 7 beats in-degree 3."""
    nodes = [box(n, 0, i * 10, 10, 10) for i, n in enumerate("abcd")]
    rows = continuous.features(page(nodes, [("a", "d"), ("b", "d"), ("c", "d")]))
    assert rows["d"][FEATURES.index("in_degree")] == 3.0
    assert rows["a"][FEATURES.index("out_degree")] == 1.0


def test_an_edge_to_a_node_that_is_not_on_the_page_is_ignored():
    rows = continuous.features(page([box("a", 0, 0, 10, 10)], [("a", "ghost")]))
    assert rows["a"][FEATURES.index("out_degree")] == 1.0
    assert rows["a"][FEATURES.index("in_degree")] == 0.0


def test_the_aspect_is_log_scaled_and_signed():
    """A wide box and a tall box of the same ratio sit equally far either side of zero."""
    rows = continuous.features(page([box("wide", 0, 0, 20, 10), box("tall", 0, 20, 10, 20)]))
    assert rows["wide"][FEATURES.index("log_aspect")] == pytest.approx(np.log(2))
    assert rows["tall"][FEATURES.index("log_aspect")] == pytest.approx(-np.log(2))


def test_a_square_has_zero_log_aspect():
    rows = continuous.features(page([box("a", 0, 0, 10, 10)]))
    assert rows["a"][FEATURES.index("log_aspect")] == pytest.approx(0.0)


def test_a_degenerate_box_does_not_divide_by_zero():
    rows = continuous.features(page([box("flat", 0, 0, 10, 0)]))
    assert np.isfinite(rows["flat"]).all()


def test_a_node_with_no_bbox_still_produces_a_finite_row():
    rows = continuous.features(page([{"id": "a", "shape": "rectangle", "text": ""}]))
    assert np.isfinite(rows["a"]).all()


def test_text_length_is_characters_rather_than_a_keyword_class():
    rows = continuous.features(page([box("a", 0, 0, 10, 10, text="hello")]))
    assert rows["a"][FEATURES.index("text_len")] == 5.0


def test_a_null_text_field_counts_as_zero():
    node = box("a", 0, 0, 10, 10)
    node["text"] = None
    assert continuous.features(page([node]))["a"][FEATURES.index("text_len")] == 0.0


# -- normalisation -----------------------------------------------------------------------------------


def test_geometry_is_normalised_by_the_pages_own_extent():
    """Two pages drawn at different scales must give the same row."""
    small = continuous.features(page([box("a", 0, 0, 10, 10), box("b", 0, 90, 10, 10)]))
    large = continuous.features(page([box("a", 0, 0, 100, 100), box("b", 0, 900, 100, 100)]))
    assert small["a"] == pytest.approx(large["a"])


def test_the_scale_free_columns_survive_the_normalisation():
    """Degree and text length are counts and must not be divided by the page size."""
    rows = continuous.features(page([box("a", 0, 0, 500, 500, text="abc")]))
    assert rows["a"][FEATURES.index("text_len")] == 3.0


def test_a_single_node_page_does_not_divide_by_a_zero_extent():
    rows = continuous.features(page([box("only", 5, 5, 0, 0)]))
    assert np.isfinite(rows["only"]).all()


# -- the sweep ---------------------------------------------------------------------------------------


def test_the_three_covariance_types_are_the_ones_the_task_compares():
    assert set(continuous.COVARIANCES) == {"full", "diag", "tied"}


def test_the_seed_is_declared():
    assert continuous.SEED == 42
