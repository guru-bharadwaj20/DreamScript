"""Phase 8.3 - the Ward dendrogram and the nesting the plan hoped for."""

from __future__ import annotations

import numpy as np
import pytest

from src.cluster import hierarchy
from src.features.descriptors import NAMES

# -- the declared design ---------------------------------------------------------------------------


def test_the_cuts_bracket_the_plans_middle_tier_and_the_true_class_count():
    assert 3 in hierarchy.CUTS  # the plan's {box-like, circular, pointed}
    assert 4 in hierarchy.CUTS  # 8.2's recommendation, and hdbpmn's class count


def test_the_linkage_is_ward():
    """Ward is 8.1's objective read bottom-up, which is what makes a disagreement informative."""
    assert hierarchy.METHOD == "ward"


# -- the tree --------------------------------------------------------------------------------------


@pytest.fixture
def blobs():
    """Three separated clouds, one of which is itself two sub-clouds."""
    rng = np.random.default_rng(0)
    a = rng.normal(0.0, 0.3, size=(40, len(NAMES)))
    b = rng.normal(20.0, 0.3, size=(40, len(NAMES)))
    c = rng.normal(40.0, 0.3, size=(40, len(NAMES)))
    c[:20] += 3.0
    labels = np.array(["a"] * 40 + ["b"] * 40 + ["c"] * 40, dtype=object)
    return np.vstack([a, b, c]), labels


def test_the_linkage_matrix_has_one_row_per_merge(blobs):
    X, _ = blobs
    assert hierarchy.tree(X).shape == (len(X) - 1, 4)


def test_a_cut_is_numbered_from_zero(blobs):
    """8.1's conventions, so the two partitions can be compared without an off-by-one."""
    X, _ = blobs
    assignment = hierarchy.cut(hierarchy.tree(X), 3)
    assert set(assignment.tolist()) == {0, 1, 2}


def test_three_obvious_clouds_are_recovered_at_a_cut_of_three(blobs):
    """The floor: if the tree cannot separate three separated blobs, no corpus number means anything."""
    from sklearn.metrics import adjusted_rand_score

    X, labels = blobs
    assert adjusted_rand_score(labels, hierarchy.cut(hierarchy.tree(X), 3)) > 0.95


def test_a_deeper_cut_only_ever_refines_a_shallower_one(blobs):
    """A hierarchy that let a cluster's members separate and rejoin would not be a hierarchy."""
    X, _ = blobs
    Z = hierarchy.tree(X)
    shallow, deep = hierarchy.cut(Z, 3), hierarchy.cut(Z, 6)
    for p in set(shallow.tolist()):
        kids = deep[shallow == p]
        # every deep cluster appearing under this parent appears under no other parent
        for kid in set(kids.tolist()):
            assert set(shallow[deep == kid].tolist()) == {p}


def test_cophenetic_correlation_is_high_when_the_tree_fits(blobs):
    """Well-separated blobs are exactly the case a dendrogram describes faithfully."""
    X, _ = blobs
    assert hierarchy.cophenetic(hierarchy.tree(X), X) > 0.8


def test_the_columns_are_standardised_before_the_tree_is_built():
    """Ward minimises squared distance, so one wide column would own every merge."""
    X = np.zeros((20, len(NAMES)))
    X[:, 0] = np.arange(20)
    out = hierarchy.scaled(X)
    assert out[:, 0].std() == pytest.approx(1.0)


# -- the nesting test ------------------------------------------------------------------------------


def test_a_split_that_separates_labels_is_recorded_as_such():
    parent = np.array([0, 0, 0, 0])
    child = np.array([0, 0, 1, 1])
    labels = np.array(["rectangle", "rectangle", "circle", "circle"], dtype=object)
    result = hierarchy.refinement(parent, child, labels)
    assert result[0]["split_by_label"] is True
    assert sorted(result[0]["child_dominants"]) == ["circle", "rectangle"]


def test_a_split_that_only_refines_the_same_axis_is_recorded_as_not_a_taxonomy():
    """Both children keeping the parent's label means the deeper cut found a ranking, not a kind."""
    parent = np.array([0, 0, 0, 0])
    child = np.array([0, 0, 1, 1])
    labels = np.array(["rectangle"] * 4, dtype=object)
    assert hierarchy.refinement(parent, child, labels)[0]["split_by_label"] is False


def test_an_unsplit_parent_reports_one_child():
    parent = np.array([0, 0, 1, 1])
    child = np.array([0, 0, 1, 1])
    labels = np.array(["a", "a", "b", "b"], dtype=object)
    assert [r["children"] for r in hierarchy.refinement(parent, child, labels)] == [1, 1]


def test_every_parent_gets_a_row_including_a_singleton():
    parent = np.array([0, 0, 1])
    child = np.array([0, 1, 2])
    labels = np.array(["a", "b", "c"], dtype=object)
    assert [r["parent"] for r in hierarchy.refinement(parent, child, labels)] == [0, 1]


# -- the reported levels ---------------------------------------------------------------------------


def test_each_level_reports_the_smallest_cluster_beside_its_score(blobs):
    """8.2's guard: a score improving while the partition dissolves is only visible beside sizes."""
    X, labels = blobs
    for row in hierarchy.levels(hierarchy.tree(X), X, labels, cuts=(2, 3)):
        assert "smallest_cluster" in row
        assert "ari" in row


def test_descriptor_means_are_named_columns_a_reader_can_picture(blobs):
    X, labels = blobs
    Z = hierarchy.tree(X)
    means = hierarchy.descriptor_means(X, hierarchy.cut(Z, 3), 3)
    assert "circularity" in means[0]
    assert "solidity" in means[0]
    assert sum(m["size"] for m in means) == len(X)
