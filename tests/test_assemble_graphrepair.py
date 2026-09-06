"""Phase 10.2.2 - the three graph repairs.

The theme of every test here is 10.2.1's lesson: a repair must not fire on a correct graph. Most
tests build one legal diagram and check the repair leaves it untouched, then break it in exactly
the one way the repair claims to fix and check the log entry names the right nodes.
"""

from __future__ import annotations

from src.assemble import graphrepair as G
from src.ir.model import Diagram, Edge, Node


def flowchart() -> Diagram:
    """start -> process -> end. Legal; every repair must leave it alone."""
    return Diagram(
        id="fc",
        diagram_type="flowchart",
        nodes=[
            Node("n1", "circle", [0, 0, 10, 10], "go", "start"),
            Node("n2", "rectangle", [0, 20, 10, 10], "do", "process"),
            Node("n3", "circle", [0, 40, 10, 10], "", "end"),
        ],
        edges=[Edge("e1", "n1", "n2"), Edge("e2", "n2", "n3")],
    )


def machine() -> Diagram:
    return Diagram(
        id="sm",
        diagram_type="state_machine",
        nodes=[
            Node("q0", "circle", [0, 0, 10, 10], "q0", "initial-state"),
            Node("q1", "circle", [20, 0, 10, 10], "q1", "state"),
            Node("q2", "double-circle", [40, 0, 10, 10], "q2", "final-state"),
        ],
        edges=[Edge("t1", "q0", "q1"), Edge("t2", "q1", "q2")],
    )


# -- insert_implicit_end --------------------------------------------------------------------


def test_a_well_formed_flowchart_gets_no_implicit_end():
    new, log = G.insert_implicit_end(flowchart())
    assert log == []
    assert {n.id for n in new.nodes} == {"n1", "n2", "n3"}


def test_a_flowchart_with_no_end_gets_one_wired_from_every_sink():
    diagram = flowchart()
    diagram.nodes[2].semantic_role = "process"  # remove the only end
    new, log = G.insert_implicit_end(diagram)
    assert len(log) == 1
    entry = log[0]
    assert entry.repair == "insert_implicit_end"
    assert entry.before == 0 and entry.after == 1
    assert "n3" in entry.refs  # n3 is the sink once it stops being the end
    end_id = entry.refs[-1]
    assert any(n.id == end_id and n.semantic_role == "end" for n in new.nodes)
    assert any(e.src == "n3" and e.dst == end_id for e in new.edges)


def test_a_flowchart_with_several_ends_is_left_alone():
    """FC.END_ONE's `!= 1` firing includes 'too many'; this repair only acts on zero."""
    diagram = flowchart()
    diagram.nodes.append(Node("n4", "circle", [20, 40, 10, 10], "", "end"))
    _, log = G.insert_implicit_end(diagram)
    assert log == []


def test_start_never_gets_wired_as_a_sink():
    diagram = Diagram(
        id="isolated_start",
        diagram_type="flowchart",
        nodes=[Node("n1", "circle", [0, 0, 10, 10], "go", "start")],
        edges=[],
    )
    _, log = G.insert_implicit_end(diagram)
    assert log == []  # no edge evidence at all - the gate must skip it, not invent a graph


def test_insert_implicit_end_does_not_touch_state_machines():
    diagram = machine()
    diagram.nodes[2].semantic_role = "state"  # no final-state now
    _, log = G.insert_implicit_end(diagram)
    assert log == []


# -- merge_duplicate_nodes ------------------------------------------------------------------


def test_a_well_formed_flowchart_has_no_duplicates_to_merge():
    _, log = G.merge_duplicate_nodes(flowchart())
    assert log == []


def test_two_boxes_with_the_same_text_and_heavy_overlap_are_merged():
    diagram = flowchart()
    diagram.nodes.append(Node("n2b", "rectangle", [0.5, 20.5, 10, 10], "do", "process"))
    new, log = G.merge_duplicate_nodes(diagram)
    assert len(log) == 1
    entry = log[0]
    assert set(entry.refs) == {"n2", "n2b"}
    assert entry.before == 2 and entry.after == 1
    assert {n.id for n in new.nodes} == {"n1", "n2", "n3"}


def test_overlap_without_matching_text_is_not_a_duplicate():
    diagram = flowchart()
    diagram.nodes.append(Node("n2b", "rectangle", [0.5, 20.5, 10, 10], "something else", "process"))
    _, log = G.merge_duplicate_nodes(diagram)
    assert log == []


def test_matching_text_without_overlap_is_not_a_duplicate():
    diagram = flowchart()
    diagram.nodes.append(Node("n2b", "rectangle", [500, 500, 10, 10], "do", "process"))
    _, log = G.merge_duplicate_nodes(diagram)
    assert log == []


def test_merging_redirects_edges_and_drops_the_created_self_loop():
    diagram = flowchart()
    diagram.nodes.append(Node("n2b", "rectangle", [0.5, 20.5, 10, 10], "do", "process"))
    diagram.edges.append(Edge("e3", "n1", "n2b"))  # duplicate of e1 once n2b folds into n2
    new, _ = G.merge_duplicate_nodes(diagram)
    pairs = [(e.src, e.dst) for e in new.edges]
    assert pairs.count(("n1", "n2")) == 1  # not doubled by the redirect
    assert all(e.src != e.dst for e in new.edges)


# -- drop_unreachable_noise ------------------------------------------------------------------


def test_a_well_formed_flowchart_has_no_noise_to_drop():
    _, log = G.drop_unreachable_noise(flowchart())
    assert log == []


def test_an_isolated_unlabelled_node_is_dropped():
    diagram = flowchart()
    diagram.nodes.append(Node("x1", "rectangle", [500, 500, 5, 5], "", "unknown"))
    new, log = G.drop_unreachable_noise(diagram)
    assert len(log) == 1
    assert log[0].refs == ("x1",)
    assert log[0].before == 1 and log[0].after == 0
    assert "x1" not in {n.id for n in new.nodes}


def test_a_component_carrying_an_anchor_role_is_never_dropped_however_small():
    """158 hdbpmn pages have several starts; a second start-only component must survive."""
    diagram = flowchart()
    diagram.nodes.append(Node("n5", "circle", [500, 500, 10, 10], "also", "start"))
    _, log = G.drop_unreachable_noise(diagram)
    assert log == []


def test_a_component_larger_than_the_noise_cap_is_left_for_a_human():
    diagram = flowchart()
    diagram.nodes += [
        Node("x1", "rectangle", [500, 500, 5, 5], "", "process"),
        Node("x2", "rectangle", [510, 500, 5, 5], "", "process"),
        Node("x3", "rectangle", [520, 500, 5, 5], "", "process"),
        Node("x4", "rectangle", [530, 500, 5, 5], "", "process"),
    ]
    diagram.edges += [
        Edge("f1", "x1", "x2"),
        Edge("f2", "x2", "x3"),
        Edge("f3", "x3", "x4"),
    ]
    _, log = G.drop_unreachable_noise(diagram)
    assert log == []


def test_drop_unreachable_noise_does_not_touch_er_diagrams():
    diagram = Diagram(
        id="er",
        diagram_type="er_diagram",
        nodes=[
            Node("e1", "rectangle", [0, 0, 10, 10], "Book", "entity"),
            Node("x1", "rectangle", [500, 500, 5, 5], "", "unknown"),
        ],
        edges=[],
    )
    _, log = G.drop_unreachable_noise(diagram)
    assert log == []


# -- the pipeline and the log shape -----------------------------------------------------------


def test_repair_runs_all_three_in_order_and_logs_each():
    diagram = flowchart()
    diagram.nodes[2].semantic_role = "process"  # drop the end
    diagram.nodes.append(Node("n2b", "rectangle", [0.5, 20.5, 10, 10], "do", "process"))
    diagram.nodes.append(Node("x1", "rectangle", [500, 500, 5, 5], "", "unknown"))
    new, log = G.repair(diagram)
    fired = {entry.repair for entry in log}
    assert fired == {"merge_duplicate_nodes", "drop_unreachable_noise", "insert_implicit_end"}
    assert "x1" not in {n.id for n in new.nodes}
    assert not any(n.id == "n2b" for n in new.nodes)


def test_a_repair_entry_serialises_to_the_documented_shape():
    entry = G.RepairEntry("insert_implicit_end", ("n3", "repair_end_0"), "detail", 0, 1)
    assert entry.to_dict() == {
        "repair": "insert_implicit_end",
        "refs": ["n3", "repair_end_0"],
        "detail": "detail",
        "before": 0,
        "after": 1,
    }


def test_iou_is_symmetric_and_zero_for_disjoint_boxes():
    assert G._iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert G._iou([0, 0, 10, 10], [100, 100, 10, 10]) == 0.0
    assert G._iou([0, 0, 10, 10], [5, 5, 10, 10]) == G._iou([5, 5, 10, 10], [0, 0, 10, 10])
