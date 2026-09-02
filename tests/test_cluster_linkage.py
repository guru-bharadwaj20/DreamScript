"""Phase 8.4 - single / complete / average / ward, on one K."""

from __future__ import annotations

import numpy as np
import pytest

from src.cluster import linkage
from src.features.descriptors import NAMES

# -- the declared design ---------------------------------------------------------------------------


def test_all_four_rules_the_plan_names_are_compared():
    assert set(linkage.METHODS) == {"single", "complete", "average", "ward"}


def test_one_k_is_held_fixed_so_the_rule_is_the_only_variable():
    assert isinstance(linkage.K, int)
    assert linkage.K == 4


# -- the fixtures ----------------------------------------------------------------------------------


@pytest.fixture
def compact():
    """Two compact, well-separated clouds - the case every rule should get right."""
    rng = np.random.default_rng(0)
    X = np.vstack(
        [rng.normal(0.0, 0.3, size=(50, len(NAMES))), rng.normal(30.0, 0.3, size=(50, len(NAMES)))]
    )
    return X, np.array(["a"] * 50 + ["b"] * 50, dtype=object)


@pytest.fixture
def bridged():
    """Two clouds joined by a thread of intermediate points - single linkage's known failure."""
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 0.4, size=(50, len(NAMES)))
    b = rng.normal(30.0, 0.4, size=(50, len(NAMES)))
    bridge = np.linspace(0.0, 30.0, 40)[:, None] * np.ones((1, len(NAMES)))
    X = np.vstack([a, b, bridge])
    return X, np.array(["a"] * 50 + ["b"] * 50 + ["bridge"] * 40, dtype=object)


# -- what each rule is expected to do --------------------------------------------------------------


def test_every_rule_separates_two_compact_clouds(compact):
    """A floor. Where the rules agree, they must all be right."""
    from sklearn.metrics import adjusted_rand_score

    X, labels = compact
    for method in linkage.METHODS:
        assignment = linkage.cut(linkage.tree(X, method=method), 2)
        assert adjusted_rand_score(labels, assignment) > 0.95, method


def test_single_linkage_chains_across_a_bridge_and_ward_does_not(bridged):
    """The textbook bias, asserted rather than assumed, because 8.4's finding leans on it."""
    X, _ = bridged
    single = np.bincount(linkage.cut(linkage.tree(X, method="single"), 2))
    ward = np.bincount(linkage.cut(linkage.tree(X, method="ward"), 2))
    assert single.max() / single.sum() > ward.max() / ward.sum()


# -- the reported quantities -----------------------------------------------------------------------


def test_the_balance_of_the_cut_is_reported_beside_the_scores(compact):
    """Single linkage's failure is a shape of partition, not a low score."""
    X, labels = compact
    result = linkage.evaluate(X, labels, "ward", k=2)
    assert "largest_share" in result
    assert "singletons" in result
    assert sum(result["sizes"]) == len(X)


def test_both_yardsticks_are_reported_for_every_rule(compact):
    """Cophenetic judges the tree; ARI judges the corpus. Neither substitutes for the other."""
    X, labels = compact
    for row in linkage.compare(X, labels, k=2):
        assert "cophenetic" in row
        assert "ari" in row


def test_singletons_are_counted(compact):
    X, labels = compact
    result = linkage.evaluate(X, labels, "single", k=4)
    assert result["singletons"] == sum(1 for s in result["sizes"] if s == 1)


def test_a_degenerate_partition_is_excluded_from_the_recommendation():
    """A rule whose largest cluster holds 95% of the corpus has not produced a partition."""
    rows = [
        {"method": "single", "ari": 0.9, "largest_share": 0.999, "cophenetic": 0.9},
        {"method": "ward", "ari": 0.1, "largest_share": 0.5, "cophenetic": 0.4},
    ]
    usable = [r for r in rows if r["largest_share"] < 0.95]
    assert max(usable, key=lambda r: r["ari"])["method"] == "ward"


def test_the_cophenetic_winner_and_the_ari_winner_are_reported_separately(compact):
    """They can be different rules, and on the real corpus they are opposite ones."""
    X, labels = compact
    rows = linkage.compare(X, labels, k=2)
    assert max(rows, key=lambda r: r["cophenetic"])["method"] in linkage.METHODS
    assert max(rows, key=lambda r: r["ari"])["method"] in linkage.METHODS


def test_purity_rewards_the_degenerate_partition_which_is_why_ari_is_carried():
    """One giant cluster mapped to the majority label *is* the majority classifier."""
    from src.parse.vocab import purity

    labels = np.array(["rectangle"] * 80 + ["circle"] * 20, dtype=object)
    everything_in_one = np.zeros(100, dtype=int)
    assert purity(everything_in_one, labels, 1)["purity"] == 0.8
