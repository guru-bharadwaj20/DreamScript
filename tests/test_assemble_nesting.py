"""Phase 10.1.8 - the tree builder's invariants, and the repair rules the corpus never exercised.

sketch2code's 484 pages fire none of `tree`'s repairs: no cycles, no multi-parent nodes, no
dangling parents. A rule whose only evidence is that it never ran has not been checked, so every
one of them is constructed here by hand.
"""

from __future__ import annotations

import json

from src.assemble.nesting import (
    COVER_SWEEP,
    PAGE_ROOT,
    Nesting,
    NestNode,
    from_boxes,
    from_contains,
    parents_from_boxes,
    reference_parents,
    tree,
)
from src.ir.model import Diagram, Edge, Node


def box(x, y, w, h):
    return [float(x), float(y), float(w), float(h)]


def contains_diagram(pairs, ids):
    return Diagram(
        id="t",
        diagram_type="wireframe",
        nodes=[Node(id=i, shape="rectangle", bbox=None) for i in ids],
        edges=[
            Edge(id=f"e{k}", src=s, dst=d, attrs={"kind": "contains"})
            for k, (s, d) in enumerate(pairs)
        ],
    )


# -- the tree builder ----------------------------------------------------------------------


def test_a_chain_of_parents_becomes_a_tree_with_increasing_depth():
    built = tree([NestNode("a"), NestNode("b", parent="a"), NestNode("c", parent="b")])
    assert built.roots == ["a"]
    assert built.depth == {"a": 0, "b": 1, "c": 2}
    assert built.max_depth == 2
    assert built.well_formed


def test_several_roots_are_joined_under_one_synthetic_page_root():
    built = tree([NestNode("a"), NestNode("b")])
    assert built.roots == [PAGE_ROOT]
    assert built.children[PAGE_ROOT] == ["a", "b"]
    assert built.depth["a"] == 1


def test_a_single_root_page_is_left_alone_rather_than_given_a_synthetic_parent():
    built = tree([NestNode("a"), NestNode("b", parent="a")])
    assert built.roots == ["a"]
    assert PAGE_ROOT not in built.parent


def test_a_cycle_is_broken_at_the_link_that_closes_it_and_the_break_is_recorded():
    built = tree([NestNode("a", parent="c"), NestNode("b", parent="a"), NestNode("c", parent="b")])
    assert built.well_formed
    # Breaking the cycle leaves exactly one root, so no synthetic root is needed.
    assert built.roots == ["b"]
    assert built.depth == {"b": 0, "c": 1, "a": 2}
    assert [d["reason"] for d in built.dropped] == ["cycle"]


def test_a_node_that_is_its_own_parent_is_dropped_rather_than_looping_forever():
    built = tree([NestNode("a", parent="a")])
    assert built.parent["a"] is None
    assert built.dropped[0]["reason"] == "self-parent"


def test_a_parent_that_is_not_on_the_page_is_dropped_and_named():
    built = tree([NestNode("a", parent="ghost")])
    assert built.dropped == [{"node": "a", "parent": "ghost", "reason": "unknown-parent"}]


def test_every_node_gets_a_depth_so_well_formed_means_nothing_was_stranded():
    built = tree([NestNode(str(i), parent=str(i - 1) if i else None) for i in range(50)])
    assert built.well_formed
    assert built.max_depth == 49


def test_children_come_back_in_reading_order_top_to_bottom_then_left_to_right():
    kids = [
        NestNode("bottom", box(0, 100, 10, 10), "root"),
        NestNode("top_right", box(100, 0, 10, 10), "root"),
        NestNode("top_left", box(0, 2, 10, 10), "root"),
    ]
    built = tree([NestNode("root", box(0, 0, 200, 200)), *kids])
    assert built.children["root"] == ["top_left", "top_right", "bottom"]


def test_nodes_without_geometry_keep_their_input_order_instead_of_being_sorted():
    built = tree([NestNode("r"), NestNode("z", parent="r"), NestNode("a", parent="r")])
    assert built.children["r"] == ["z", "a"]


def test_the_nesting_serialises_to_json_able_primitives():
    built = tree([NestNode("a"), NestNode("b", parent="a")])
    assert json.loads(json.dumps(built.to_dict()))["roots"] == ["a"]
    assert isinstance(built, Nesting)


# -- from contains edges -------------------------------------------------------------------


def test_contains_edges_become_a_parent_map_with_the_source_as_the_parent():
    built = from_contains(contains_diagram([("a", "b"), ("b", "c")], ["a", "b", "c"]))
    assert built.parent["b"] == "a"
    assert built.depth["c"] == 2


def test_a_node_claimed_by_two_parents_keeps_the_first_relation_in_file_order():
    built = from_contains(contains_diagram([("a", "c"), ("b", "c")], ["a", "b", "c"]))
    assert built.parent["c"] == "a"
    assert [d["reason"] for d in built.dropped] == ["multi-parent"]


def test_edges_that_are_not_containment_are_ignored():
    diagram = contains_diagram([("a", "b")], ["a", "b"])
    diagram.edges.append(Edge(id="flow", src="b", dst="a", attrs={"kind": "sequence"}))
    assert from_contains(diagram).parent["a"] is None


# -- the geometric finder ------------------------------------------------------------------


def test_the_innermost_enclosing_box_wins_not_the_outermost():
    boxes = [
        ("pool", box(0, 0, 100, 100)),
        ("lane", box(0, 0, 100, 40)),
        ("task", box(10, 10, 10, 10)),
    ]
    assert parents_from_boxes(boxes)["task"] == "lane"
    assert parents_from_boxes(boxes, innermost=False)["task"] == "pool"


def test_a_box_that_overshoots_its_container_is_still_contained_within_tolerance():
    boxes = [("outer", box(0, 0, 100, 100)), ("inner", box(-5, 10, 100, 10))]
    assert parents_from_boxes(boxes, cover=0.9)["inner"] == "outer"
    assert parents_from_boxes(boxes, cover=0.99)["inner"] is None


def test_two_nearly_identical_boxes_do_not_become_each_others_parent():
    boxes = [("a", box(0, 0, 100, 100)), ("b", box(1, 1, 99, 99))]
    assert parents_from_boxes(boxes) == {"a": None, "b": None}


def test_a_box_that_merely_overlaps_is_not_a_parent():
    boxes = [("a", box(0, 0, 100, 100)), ("b", box(80, 80, 100, 100))]
    assert parents_from_boxes(boxes)["b"] is None


def test_from_boxes_produces_a_single_rooted_tree_over_a_pool_lane_task_page():
    built = from_boxes(
        [
            ("pool", box(0, 0, 200, 100)),
            ("lane", box(0, 0, 200, 50)),
            ("task", box(10, 10, 20, 20)),
            ("stray", box(500, 500, 10, 10)),
        ]
    )
    assert built.roots == [PAGE_ROOT]
    assert built.depth["task"] == 3
    assert built.well_formed


def test_the_finder_is_deterministic_when_two_candidate_parents_have_equal_area():
    boxes = [("z", box(0, 0, 50, 50)), ("a", box(0, 0, 50, 50)), ("k", box(1, 1, 10, 10))]
    assert parents_from_boxes(boxes)["k"] == "a"


# -- the reference ------------------------------------------------------------------------


def test_only_bpmn_container_tags_may_be_a_reference_parent():
    diagram = Diagram(
        id="t",
        diagram_type="bpmn",
        nodes=[
            Node(
                id="lane", shape="rectangle", bbox=box(0, 0, 100, 100), attrs={"bpmn_tag": "lane"}
            ),
            Node(id="big", shape="rectangle", bbox=box(0, 0, 90, 90), attrs={"bpmn_tag": "task"}),
            Node(id="t1", shape="rectangle", bbox=box(5, 5, 10, 10), attrs={"bpmn_tag": "task"}),
        ],
    )
    # `big` encloses `t1` and is the innermost box that does, but a task is not a container.
    assert reference_parents(diagram)["t1"] == "lane"
    assert reference_parents(diagram)["lane"] is None


def test_the_tolerance_sweep_brackets_the_constant_that_was_chosen():
    from src.assemble.nesting import COVER

    assert min(COVER_SWEEP) < COVER < max(COVER_SWEEP)
