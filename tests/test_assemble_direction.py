"""Phase 10.1.5 - behaviour of arrow direction resolution."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.assemble import direction as D
from src.ir.model import Edge, Node


def _nodes() -> dict[str, Node]:
    return {
        "a": Node(id="a", shape="rectangle", bbox=[0.0, 0.0, 20.0, 20.0], semantic_role="start"),
        "b": Node(id="b", shape="rectangle", bbox=[100.0, 0.0, 20.0, 20.0], semantic_role="end"),
    }


def _edge(polyline=None) -> Edge:
    return Edge(id="e", src="a", dst="b", polyline=polyline or [[20.0, 10.0], [100.0, 10.0]])


def _evidence(**kwargs) -> D.Evidence:
    kwargs.setdefault("diagonal", 200.0)
    kwargs.setdefault("weights", {name: 1.0 for name in D.FEATURES})
    return D.Evidence(**kwargs)


# -- the antisymmetry that pins chance at 0.5 ---------------------------------------------


def test_every_feature_negates_when_the_two_endpoints_swap_places():
    """Antisymmetry is what pins chance at 0.5, so it is checked rather than asserted."""
    evidence = _evidence(det_heads=[(100.0, 10.0, 0.9)], geo_heads=[(100.0, 10.0, 0.0)])
    forward = D.features(_edge(), _nodes(), evidence)

    swapped = _nodes()
    swapped["a"].bbox, swapped["b"].bbox = swapped["b"].bbox, swapped["a"].bbox
    swapped["a"].semantic_role, swapped["b"].semantic_role = "end", "start"
    backward = D.features(_edge(), swapped, evidence)

    for name in D.FEATURES:
        assert backward[name] == pytest.approx(-forward[name])
    assert abs(forward["head_det"]) > 0.8


def test_naming_the_edge_the_other_way_round_changes_nothing():
    """The canonical orientation is `sorted((src, dst))`, so the IR's own order cannot leak in."""
    nodes = _nodes()
    evidence = _evidence(det_heads=[(100.0, 10.0, 0.9)])
    forward = D.features(_edge(), nodes, evidence)
    named = Edge(id="e", src="b", dst="a", polyline=_edge().polyline)
    assert D.features(named, nodes, evidence) == forward


def test_the_features_do_not_depend_on_the_stored_waypoint_order():
    nodes = _nodes()
    evidence = _evidence(det_heads=[(100.0, 10.0, 0.9)])
    straight = D.features(_edge(), nodes, evidence)
    reversed_ = D.features(_edge(polyline=[[100.0, 10.0], [20.0, 10.0]]), nodes, evidence)
    assert straight == reversed_


def test_a_polyline_is_oriented_by_the_node_centres_not_by_its_stored_order():
    nodes = _nodes()
    line = D._oriented(_edge(polyline=[[100.0, 10.0], [20.0, 10.0]]), nodes["a"], nodes["b"])
    assert line[0][0] < line[-1][0]


# -- each source of evidence, alone --------------------------------------------------------


def test_an_arrowhead_at_one_end_makes_that_end_the_target():
    nodes = _nodes()
    evidence = _evidence(det_heads=[(100.0, 10.0, 0.9)], weights={"head_det": 4.0})
    decision = D.resolve(_edge(), nodes, evidence)
    assert (decision.src, decision.dst) == ("a", "b")
    assert decision.features["head_det"] > 0

    flipped = D.resolve(
        _edge(), nodes, _evidence(det_heads=[(20.0, 10.0, 0.9)], weights={"head_det": 4.0})
    )
    assert (flipped.src, flipped.dst) == ("b", "a")
    assert flipped.flipped


def test_a_head_further_away_than_the_radius_barely_votes():
    nodes = _nodes()
    near = D.features(_edge(), nodes, _evidence(det_heads=[(100.0, 10.0, 1.0)]))["head_det"]
    far = D.features(_edge(), nodes, _evidence(det_heads=[(100.0, 60.0, 1.0)]))["head_det"]
    assert near > 0.9
    assert far < 0.05


def test_a_geometric_head_pointing_back_along_the_line_does_not_vote_for_that_end():
    nodes = _nodes()
    forwards = D.features(_edge(), nodes, _evidence(geo_heads=[(100.0, 10.0, 0.0)]))["head_geo"]
    backwards = D.features(_edge(), nodes, _evidence(geo_heads=[(100.0, 10.0, math.pi)]))[
        "head_geo"
    ]
    assert forwards > 0.9
    assert backwards == pytest.approx(0.0)


def test_a_start_node_is_a_source_and_a_terminal_node_is_a_target():
    nodes = _nodes()
    assert D.features(_edge(), nodes, _evidence())["role"] == pytest.approx(1.0)


def test_a_role_the_state_space_does_not_constrain_says_nothing():
    nodes = _nodes()
    nodes["a"].semantic_role = "process"
    nodes["b"].semantic_role = "process"
    assert D.features(_edge(), nodes, _evidence())["role"] == pytest.approx(0.0)


def test_the_flow_prior_is_the_unit_vector_between_the_two_node_centres():
    nodes = _nodes()
    scores = D.features(_edge(), nodes, _evidence())
    assert scores["prior_x"] == pytest.approx(1.0)
    assert scores["prior_y"] == pytest.approx(0.0)


# -- the combination ------------------------------------------------------------------------


def test_a_weight_of_zero_removes_a_source_from_the_decision():
    nodes = _nodes()
    evidence = _evidence(det_heads=[(20.0, 10.0, 1.0)], weights={"head_det": 0.0, "role": 1.0})
    decision = D.resolve(_edge(), nodes, evidence)
    assert (decision.src, decision.dst) == ("a", "b")
    assert decision.contributions["head_det"] == 0.0


def test_a_confident_arrowhead_outvotes_a_contrary_prior():
    nodes = _nodes()
    nodes["a"].semantic_role = "process"
    nodes["b"].semantic_role = "process"
    evidence = _evidence(det_heads=[(20.0, 10.0, 1.0)], weights={"head_det": 5.0, "prior_x": 1.0})
    assert D.resolve(_edge(), nodes, evidence).src == "b"


def test_resolve_returns_none_rather_than_guessing_when_there_is_no_geometry():
    nodes = _nodes()
    assert D.resolve(Edge(id="e", src="a", dst="b"), nodes, _evidence()) is None
    assert (
        D.resolve(Edge(id="e", src="a", dst="a", polyline=[[0, 0], [1, 1]]), nodes, _evidence())
        is None
    )
    assert D.resolve(_edge(), {"a": nodes["a"]}, _evidence()) is None


def test_the_decision_reports_why_it_decided():
    nodes = _nodes()
    decision = D.resolve(_edge(), nodes, _evidence(det_heads=[(100.0, 10.0, 1.0)]))
    assert set(decision.features) == set(D.FEATURES)
    assert set(decision.contributions) == set(D.FEATURES)
    assert decision.logodds == pytest.approx(sum(decision.contributions.values()))
    assert decision.pair == ("a", "b")


# -- the fit --------------------------------------------------------------------------------


def test_the_logistic_fit_has_no_intercept_so_a_dead_feature_leaves_a_coin():
    rows = [{"y": 1.0, "f": 0.0} for _ in range(80)] + [{"y": 0.0, "f": 0.0} for _ in range(20)]
    weights = D.fit(D._matrix(rows, ("f",)), np.array([r["y"] for r in rows]))
    assert weights[0] == pytest.approx(0.0, abs=1e-6)
    assert D._accuracy(rows, ("f",), weights) == pytest.approx(0.8)


def test_the_logistic_fit_recovers_a_separating_direction():
    rows = [{"y": 1.0, "f": 1.0}] * 50 + [{"y": 0.0, "f": -1.0}] * 50
    weights = D.fit(D._matrix(rows, ("f",)), np.array([r["y"] for r in rows]))
    assert weights[0] > 0
    assert D._accuracy(rows, ("f",), weights) == pytest.approx(1.0)


def test_the_reported_ablations_cover_each_source_alone_each_pair_and_all_three():
    labels = {label for label, _ in D.ABLATIONS}
    assert {"head", "prior", "role", "head+prior", "head+role", "prior+role", "all"} <= labels
    for _, names in D.ABLATIONS:
        assert set(names) <= set(D.FEATURES)


# -- the measured result --------------------------------------------------------------------


def test_the_written_experiment_reports_the_target_per_source_and_meets_it_on_hdbpmn():
    import json

    from src.assemble.corpus import RUNS

    path = RUNS / "direction.json"
    if not path.is_file():
        pytest.skip("run `python -m src.assemble.direction` first")
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["target"] == 0.90
    assert result["ablation"]["all"]["hdbpmn"] >= 0.90
    assert result["meets_target"]["hdbpmn"] is True
    # The control that matters: the combination must beat the free flow prior.
    assert result["beats_prior_by"]["hdbpmn"] > 0
    # flowchartseg carries no edges at all, so it cannot appear as a scored corpus.
    assert "flowchartseg" not in result["ablation"]["all"]
