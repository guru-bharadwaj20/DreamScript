"""Phase 10.1.6 - binding a floating text region to its edge, and typing the bound label."""

from __future__ import annotations

from src.assemble import edgelabels as E

# -- classify() -----------------------------------------------------------------------------


def test_yes_and_its_synonyms_classify_as_yes():
    assert E.classify("Yes") == "yes"
    assert E.classify("true") == "yes"
    assert E.classify("ja") == "yes"


def test_no_and_its_synonyms_classify_as_no():
    assert E.classify("No") == "no"
    assert E.classify("false") == "no"
    assert E.classify("non") == "no"


def test_bare_y_and_n_are_not_read_as_yes_no():
    """fa_bresler's alphabet is single letters; a rule that reads `n` as no would mistype it."""
    assert E.classify("y") != "yes"
    assert E.classify("n") != "no"


def test_er_cardinality_notation_classifies_as_cardinality():
    assert E.classify("1..*") == "cardinality"
    assert E.classify("0..n") == "cardinality"
    assert E.classify("(0,n)") == "cardinality"
    assert E.classify("*") == "cardinality"


def test_single_letter_automaton_symbols_classify_as_symbol():
    assert E.classify("a") == "symbol"
    assert E.classify("0") == "symbol"
    assert E.classify("a,b") == "symbol"
    assert E.classify("0,1") == "symbol"


def test_a_phrase_with_letters_and_no_other_shape_classifies_as_condition():
    assert E.classify("risk above threshold") == "condition"
    assert E.classify("application rejected") == "condition"


def test_an_empty_label_classifies_as_other():
    assert E.classify("") == "other"
    assert E.classify("   ") == "other"


def test_has_condition_marker_distinguishes_a_boolean_phrase_from_a_bare_noun():
    assert E.has_condition_marker("risk above threshold") is True
    assert E.has_condition_marker("insurance claim") is False


# -- geometry: distance and arc position -----------------------------------------------------


def test_a_point_on_the_polyline_has_zero_distance():
    polyline = [(0, 0), (10, 0), (10, 10)]
    d, arc = E.polyline_distance((10, 0), polyline)
    assert d == 0.0
    assert 0.0 < arc < 1.0


def test_arc_position_is_zero_at_the_start_and_one_at_the_end():
    polyline = [(0, 0), (10, 0)]
    _, arc_start = E.polyline_distance((0, 0), polyline)
    _, arc_end = E.polyline_distance((10, 0), polyline)
    assert arc_start == 0.0
    assert arc_end == 1.0


def test_arc_position_of_the_true_midpoint_of_a_straight_edge_is_one_half():
    polyline = [(0, 0), (10, 0)]
    _, arc = E.polyline_distance((5, 0), polyline)
    assert arc == 0.5


def test_midpoint_of_an_l_shaped_polyline_is_not_at_the_corner():
    """An L with a long first leg and a short second leg puts the arc midpoint on the first leg,
    which is the case the plan's "near edge midpoint" phrasing has to survive: a label sitting at
    the corner is not at the midpoint unless the two legs happen to be equal length."""
    polyline = [(0, 0), (90, 0), (90, 10)]
    mx, my = E.midpoint(polyline)
    assert (mx, my) == (50.0, 0.0)


def test_a_single_point_polyline_does_not_crash_distance_or_midpoint():
    d, arc = E.polyline_distance((3, 4), [(0, 0)])
    assert d == 5.0
    assert arc == 0.5
    assert E.midpoint([(0, 0)]) == (0.0, 0.0)


# -- bind(): the binding rule, node refusal, and the competition case ------------------------


def _edge(edge_id: str, polyline: list[tuple[float, float]]) -> dict:
    return {"id": edge_id, "label": edge_id, "polyline": polyline}


def test_a_label_near_one_edge_and_far_from_another_binds_to_the_near_one():
    edges = [_edge("e0", [(0, 0), (100, 0)]), _edge("e1", [(0, 500), (100, 500)])]
    bindings = E.bind(edges, [[45, -2, 10, 4]], diagonal=1000.0)
    assert len(bindings) == 1
    assert bindings[0].edge == "e0"
    assert bindings[0].rejected is None


def test_a_region_mostly_inside_a_node_box_is_refused_even_though_an_edge_is_closer():
    """The interesting failure mode this module must not have: stealing a node's own text."""
    edges = [_edge("e0", [(0, 0), (100, 0)])]
    node_boxes = [[0, -5, 20, 10]]
    bindings = E.bind(edges, [[2, -3, 10, 6]], node_boxes=node_boxes, diagonal=1000.0)
    assert bindings[0].edge is None
    assert bindings[0].rejected == "node"


def test_a_region_far_from_every_edge_is_left_unbound_rather_than_forced_to_the_nearest():
    edges = [_edge("e0", [(0, 0), (100, 0)])]
    bindings = E.bind(edges, [[5000, 5000, 10, 10]], diagonal=1000.0)
    assert bindings[0].edge is None
    assert bindings[0].rejected == "far"


def test_a_label_between_two_parallel_flows_out_of_one_gateway_is_marked_contested():
    """The competition case: two edges leaving a gateway close enough together that a label
    sitting between them could plausibly belong to either."""
    edges = [
        _edge("e0", [(0, 0), (100, 20)]),
        _edge("e1", [(0, 0), (100, -20)]),
    ]
    bindings = E.bind(edges, [[48, -2, 4, 4]], diagonal=1000.0)
    assert bindings[0].edge is not None
    assert bindings[0].contested is True


def test_a_label_clearly_closer_to_one_of_two_edges_is_not_contested():
    edges = [
        _edge("e0", [(0, 0), (100, 0)]),
        _edge("e1", [(0, 500), (100, 500)]),
    ]
    bindings = E.bind(edges, [[50, 1, 4, 4]], diagonal=1000.0)
    assert bindings[0].contested is False


def test_the_midpoint_rule_binds_a_region_near_the_corner_of_an_l_to_the_wrong_edge_end():
    """`rule="midpoint"` scores the plan's literal wording. A region sitting right at the corner
    of a long L-shaped edge is far from that edge's midpoint (which sits on the long leg) and can
    lose to a short straight edge whose midpoint happens to be nearer - the scenario the write-up's
    arc-position measurement says is common on this corpus."""
    edges = [
        _edge("e0", [(0, 0), (100, 0), (100, 100)]),  # midpoint at (50, 0)
        _edge("e1", [(96, 96), (104, 96)]),  # short edge whose midpoint is near the L's corner
    ]
    region = [96, 92, 4, 4]  # near the L's corner (100, 100), far from its own midpoint
    polyline_binding = E.bind(edges, [region], diagonal=1000.0, rule="polyline")[0]
    midpoint_binding = E.bind(edges, [region], diagonal=1000.0, rule="midpoint")[0]
    assert polyline_binding.edge == "e0"
    assert midpoint_binding.edge == "e1"


# -- match_read(): the ground-truth construction ---------------------------------------------


def test_an_exact_read_after_normalisation_matches_its_label():
    label, how = E.match_read("Yes", ["yes", "no"])
    assert label == "yes"
    assert how == "exact"


def test_a_near_read_within_the_cer_gate_matches_a_multi_character_label():
    label, how = E.match_read("compiete", ["complete"])
    assert label == "complete"
    assert how == "near"


def test_a_single_character_read_never_matches_by_near_agreement():
    """At length 1 every string is within one edit of every other; only an exact read counts."""
    label, how = E.match_read("b", ["a"])
    assert label is None
    assert how == "none"


def test_a_read_that_matches_nothing_returns_none():
    label, how = E.match_read("xyz123", ["yes", "no", "complete"])
    assert label is None
    assert how == "none"
