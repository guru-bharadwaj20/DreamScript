"""Phase 7.4.5 - naming the components, and the two names that are allowed to disagree."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.descriptors import NAMES
from src.parse import shapenames


def mean(**overrides):
    """A component mean with 7.4.1's measured rectangle values as the base."""
    base = dict.fromkeys(NAMES, 0.0)
    base.update(circularity=0.215, extent=0.490, solidity=0.574, rect_aspect=3.26)
    base.update(overrides)
    return base


# -- the vocabulary that can be reached ---------------------------------------------------------------


def test_the_vocabulary_is_the_reachable_subset_of_the_plans_names():
    assert set(shapenames.VOCABULARY) == {"rectangle", "diamond", "round", "blob"}


def test_the_two_unreachable_names_are_not_offered():
    """`oval` has no rows in hdbpmn and `arrow` is an edge, so neither can be earned."""
    assert "oval" not in shapenames.VOCABULARY
    assert "arrow" not in shapenames.VOCABULARY


def test_every_name_the_rules_can_return_is_in_the_vocabulary():
    rng = np.random.default_rng(0)
    for _ in range(200):
        candidate = mean(
            circularity=float(rng.uniform(0, 1)),
            extent=float(rng.uniform(0, 1)),
            solidity=float(rng.uniform(0, 1)),
            rect_aspect=float(rng.uniform(1, 6)),
        )
        assert shapenames.name_from(candidate) in shapenames.VOCABULARY


# -- the naming rules ---------------------------------------------------------------------------------


def test_a_measured_rectangle_is_named_rectangle():
    assert shapenames.name_from(mean()) == "rectangle"


def test_a_measured_circle_is_named_round():
    """7.4.1's circle row, which scores 0.520 - nowhere near a textbook 1.0."""
    assert (
        shapenames.name_from(
            mean(circularity=0.520, extent=0.637, solidity=0.830, rect_aspect=1.16)
        )
        == "round"
    )


def test_a_measured_diamond_is_named_diamond():
    assert (
        shapenames.name_from(
            mean(circularity=0.310, extent=0.414, solidity=0.631, rect_aspect=1.20)
        )
        == "diamond"
    )


def test_a_dented_outline_is_a_blob_whatever_else_it_looks_like():
    assert shapenames.name_from(mean(solidity=0.3, circularity=0.9)) == "blob"


def test_the_rules_are_keyed_on_measured_values_not_textbook_ones():
    """A rule wanting circularity 1.0 for a circle would name nothing on this corpus."""
    assert shapenames.name_from(mean(circularity=0.52, solidity=0.83)) == "round"


def test_a_wide_low_extent_shape_is_a_rectangle_not_a_diamond():
    """A diamond is squarish; a wide box that happens to fill its box loosely is not one."""
    assert shapenames.name_from(mean(extent=0.45, rect_aspect=4.0)) == "rectangle"


# -- the component table -------------------------------------------------------------------------------


@pytest.fixture
def two_components():
    X = np.zeros((10, len(NAMES)))
    circle = NAMES.index("circularity")
    X[:5, circle] = 0.52
    X[5:, circle] = 0.21
    labels = np.array(["circle"] * 5 + ["rectangle"] * 5, dtype=object)
    assignment = np.array([0] * 5 + [1] * 5)
    responsibility = np.tile([0.9, 0.1], (10, 1))
    return X, labels, assignment, responsibility


def test_every_component_gets_a_row(two_components):
    rows = shapenames.describe_components(*two_components, k=2)
    assert [row["component"] for row in rows] == [0, 1]


def test_the_sizes_account_for_every_node(two_components):
    rows = shapenames.describe_components(*two_components, k=2)
    assert sum(row["size"] for row in rows) == 10


def test_a_component_nothing_was_assigned_to_is_reported_not_dropped(two_components):
    """An empty component is a finding about K, so it must survive into the table."""
    X, labels, assignment, responsibility = two_components
    responsibility = np.tile([0.6, 0.3, 0.1], (10, 1))
    rows = shapenames.describe_components(X, labels, assignment, responsibility, k=3)
    assert len(rows) == 3
    assert rows[2]["size"] == 0
    assert rows[2]["label_name"] is None


def test_the_label_name_is_the_commonest_label_in_the_component(two_components):
    rows = shapenames.describe_components(*two_components, k=2)
    assert rows[0]["label_name"] == "circle"
    assert rows[1]["label_name"] == "rectangle"


def test_a_pure_component_reports_purity_one(two_components):
    rows = shapenames.describe_components(*two_components, k=2)
    assert rows[0]["label_purity"] == 1.0


def test_purity_is_of_the_component_not_of_the_corpus():
    """Pooled purity hides the case where some components are clean and others are noise."""
    X = np.zeros((10, len(NAMES)))
    labels = np.array(["circle"] * 2 + ["rectangle"] * 8, dtype=object)
    assignment = np.array([0, 0] + [1] * 8)
    responsibility = np.tile([0.9, 0.1], (10, 1))
    rows = shapenames.describe_components(X, labels, assignment, responsibility, k=2)
    assert rows[0]["label_purity"] == 1.0
    assert rows[0]["size"] == 2


def test_the_label_mix_sums_to_one(two_components):
    X, labels, assignment, responsibility = two_components
    labels = np.array(["circle"] * 3 + ["rectangle"] * 2 + ["rectangle"] * 5, dtype=object)
    rows = shapenames.describe_components(X, labels, assignment, responsibility, k=2)
    assert sum(rows[0]["label_mix"].values()) == pytest.approx(1.0, abs=1e-3)


def test_both_names_are_recorded_so_they_can_be_compared(two_components):
    rows = shapenames.describe_components(*two_components, k=2)
    assert rows[0]["shape_name"] and rows[0]["label_name"]


# -- the report ----------------------------------------------------------------------------------------


def test_the_markdown_has_a_row_per_component(two_components):
    rows = shapenames.describe_components(*two_components, k=2)
    text = shapenames.markdown(
        {
            "k": 2,
            "rows": 10,
            "covariance": "full",
            "components": rows,
            "distinct_label_names": 2,
            "distinct_shape_names": 2,
            "names_agree": 2,
        }
    )
    assert text.count("\n| 0 |") or "| 0 |" in text
    assert "| 1 |" in text
