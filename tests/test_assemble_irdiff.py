"""Phase 10.2.5 - the IR diff metric: node correspondence, edge cascade, and the GED bound."""

from __future__ import annotations

import pytest

from src.assemble import irdiff as D
from src.ir.model import Diagram, Edge, Node


def graph(
    nodes: list[tuple[str, str, list[float] | None, str]],
    edges: list[tuple[str, str]],
) -> Diagram:
    """A Diagram from `(id, shape, bbox, text)` nodes and `(src, dst)` edges."""
    return Diagram(
        id="t",
        diagram_type="flowchart",
        nodes=[Node(id=i, shape=s, bbox=b, text=x) for i, s, b, x in nodes],
        edges=[Edge(id=f"e{k}", src=a, dst=b) for k, (a, b) in enumerate(edges)],
    )


def box_chain(n: int, texted: bool = True) -> Diagram:
    """A straight-line chain of `n` boxes, non-overlapping, each with distinct text."""
    nodes = [
        (f"n{i}", "rectangle", [float(i * 10), 0.0, 8.0, 8.0], f"step{i}" if texted else "")
        for i in range(n)
    ]
    edges = [(f"n{i}", f"n{i + 1}") for i in range(n - 1)]
    return graph(nodes, edges)


# -- identity ---------------------------------------------------------------------------


def test_identity_is_exact_under_geometric_and_gated_matching_on_real_corpus_files():
    from src.assemble.corpus import pages, truth

    checked = 0
    for page in list(pages())[:30]:
        diagram = truth(page)
        if not diagram.nodes:
            continue
        for matcher in ("geometric", "gated"):
            d = D.diff(diagram, diagram, match=matcher)
            assert d.node_f1 == 1.0
            assert d.ged == 0.0
            assert d.edits["node_insert"] == []
            assert d.edits["node_delete"] == []
        checked += 1
    assert checked > 0


def test_identity_can_fail_for_text_matching_when_a_node_carries_no_text():
    """Empty-vs-empty text is scored 0 by design, so a textless node cannot match itself
    under the text rule. This is the metric's documented behaviour, not a bug to paper over."""
    blank = graph([("a", "rectangle", [0.0, 0.0, 8.0, 8.0], "")], [])
    d = D.diff(blank, blank, match="text")
    assert d.node_f1 == 0.0


def test_identity_holds_for_text_matching_when_every_node_has_text():
    diagram = box_chain(4, texted=True)
    for matcher in D.MATCHERS:
        d = D.diff(diagram, diagram, match=matcher)
        assert d.node_f1 == 1.0
        assert d.ged == 0.0


# -- matching rule comparison -------------------------------------------------------------


def test_a_relabelled_node_still_matches_geometrically_but_not_by_text():
    truth = box_chain(3, texted=True)
    predicted = D.perturb(truth, "rename_labels", k=1)
    geometric = D.diff(predicted, truth, match="geometric")
    text = D.diff(predicted, truth, match="text")
    assert geometric.node_f1 == 1.0
    assert text.node_f1 < 1.0


def test_a_jittered_node_still_matches_by_text_but_not_geometrically():
    truth = box_chain(3, texted=True)
    predicted = D.perturb(truth, "jitter_boxes", k=3)
    geometric = D.diff(predicted, truth, match="geometric")
    text = D.diff(predicted, truth, match="text")
    assert text.node_f1 == 1.0
    assert geometric.node_f1 < 1.0


def test_gated_never_promotes_a_pair_geometry_rejected():
    """`gated` mixes in text only as a tie-break among boxes that already pass the IoU gate,
    so it can never do better than geometric at finding correspondences geometry refused."""
    truth = box_chain(5, texted=True)
    for kind in D.PERTURBATIONS:
        for k in (1, 2, 3):
            predicted = D.perturb(truth, kind, k)
            geometric = D.diff(predicted, truth, match="geometric")
            gated = D.diff(predicted, truth, match="gated")
            assert gated.node_f1 <= geometric.node_f1 + 1e-9 or gated.node_f1 == geometric.node_f1


# -- the edge cascade ---------------------------------------------------------------------


def test_jittering_a_node_out_of_the_iou_gate_leaves_its_edges_in_the_predicted_graph():
    """The perturbation the cascade study needs: an unmatched node whose edges are still
    present in the predicted graph, so they become unprojectable rather than simply absent."""
    truth = box_chain(4, texted=True)
    predicted = D.perturb(truth, "jitter_boxes", k=4)
    assert len(predicted.edges) == len(truth.edges)
    d = D.diff(predicted, truth, match="geometric")
    assert d.counts["matched_nodes"] < d.counts["true_nodes"]
    assert d.counts["edges_lost_to_unmatched_nodes"] > 0


def test_deleting_a_node_deletes_its_edges_too_so_no_cascade_is_visible():
    """The bug the previous agent was mid-fix on: `delete_nodes` removes the edges along with
    the node, so a missed node never leaves a dangling predicted edge behind to measure."""
    truth = box_chain(4, texted=True)
    predicted = D.perturb(truth, "delete_nodes", k=1)
    d = D.diff(predicted, truth, match="geometric")
    assert d.counts["edges_lost_to_unmatched_nodes"] == 0


def test_edge_f1_degrades_faster_than_node_f1_under_the_jitter_cascade():
    truth = box_chain(6, texted=True)
    predicted = D.perturb(truth, "jitter_boxes", k=3)
    d = D.diff(predicted, truth, match="geometric")
    assert d.edge_f1 <= d.node_f1


# -- graph edit distance: bound vs. brute-force exact --------------------------------------


def test_the_assignment_ged_never_falls_below_the_brute_force_exact_ged():
    truth = box_chain(5, texted=True)
    for kind in D.PERTURBATIONS:
        for k in (1, 2):
            predicted = D.perturb(truth, kind, k)
            approximate = D.ged_bound(predicted, truth, match="combined")
            exact = D.exact_ged(predicted, truth)
            assert approximate >= exact - 1e-9


def test_the_assignment_ged_matches_exact_ged_on_a_single_clean_deletion():
    """One deleted node with no ambiguity in the rest of the graph: the similarity-optimal
    assignment and the cost-optimal one coincide, so the bound is tight."""
    truth = box_chain(4, texted=True)
    predicted = D.perturb(truth, "delete_nodes", k=1)
    approximate = D.ged_bound(predicted, truth, match="combined")
    exact = D.exact_ged(predicted, truth)
    assert approximate == pytest.approx(exact)


def test_exact_ged_refuses_graphs_above_its_node_limit():
    big = box_chain(D.EXACT_MAX_NODES + 1, texted=True)
    with pytest.raises(ValueError):
        D.exact_ged(big, big)


def test_ged_is_zero_only_for_identical_graphs():
    truth = box_chain(4, texted=True)
    predicted = D.perturb(truth, "delete_edges", k=1)
    assert D.ged_bound(predicted, truth, match="geometric") > 0.0
    assert D.ged_bound(truth, truth, match="geometric") == 0.0


# -- monotonicity -----------------------------------------------------------------------


@pytest.mark.parametrize("kind", D.PERTURBATIONS)
def test_ged_normalised_is_non_decreasing_as_damage_increases(kind):
    truth = box_chain(6, texted=True)
    values = [
        D.diff(D.perturb(truth, kind, k), truth, match="combined").ged_normalised for k in range(5)
    ]
    assert D._non_decreasing(values)


@pytest.mark.parametrize("kind", D.PERTURBATIONS)
def test_node_f1_is_non_increasing_as_damage_increases_under_combined_matching(kind):
    truth = box_chain(6, texted=True)
    values = [D.diff(D.perturb(truth, kind, k), truth, match="combined").node_f1 for k in range(5)]
    assert D._non_increasing(values)


def test_more_deleted_nodes_never_raises_node_f1():
    truth = box_chain(8, texted=True)
    f1s = [
        D.diff(D.perturb(truth, "delete_nodes", k), truth, match="geometric").node_f1
        for k in range(6)
    ]
    assert D._non_increasing(f1s)
    assert f1s[0] == 1.0


# -- similarity primitives ------------------------------------------------------------------


def test_box_similarity_is_iou_and_zero_for_a_missing_box():
    assert D.box_similarity(None, [0.0, 0.0, 1.0, 1.0]) == 0.0
    assert D.box_similarity([0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 2.0, 2.0]) == pytest.approx(1.0)


def test_text_similarity_of_two_blanks_is_the_empty_text_constant_not_one():
    assert D.text_similarity("", "") == D.EMPTY_TEXT_SIMILARITY


def test_text_similarity_of_identical_nonempty_text_is_one():
    assert D.text_similarity("hello", "hello") == pytest.approx(1.0)


def test_similarity_matrix_rejects_an_unknown_matcher():
    with pytest.raises(ValueError):
        D.similarity_matrix([], [], "nonsense")


# -- the trivial predictor path ------------------------------------------------------------


def test_detections_as_diagram_produces_nodes_with_no_edges():
    class Page:
        name = "p"

    boxes = [{"cls": "rectangle", "bbox": [0.0, 0.0, 1.0, 1.0], "score": 0.9}]
    diagram = D.detections_as_diagram(Page(), boxes)
    assert len(diagram.nodes) == 1
    assert diagram.edges == []
