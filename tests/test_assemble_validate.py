"""Phase 10.2.1 - the type-specific semantic validators.

A validator is worth having only if it fires on a broken graph and stays silent on a good one, so
most of these build one diagram of each kind and break it in one place. The other half of the
file is about the thing the corpus sweep turns on: a rule that cannot be asked must be *skipped*
and not scored as a pass.
"""

from __future__ import annotations

from src.assemble import validate as V
from src.ir.model import Diagram, Edge, Node


def flowchart() -> Diagram:
    """start -> decision -> two ends. Legal by every rule in the registry."""
    return Diagram(
        id="fc",
        diagram_type="flowchart",
        nodes=[
            Node("n1", "circle", [0, 0, 10, 10], "go", "start"),
            Node("n2", "diamond", [0, 20, 10, 10], "ok?", "decision"),
            Node("n3", "circle", [0, 40, 10, 10], "", "end"),
            Node("n4", "circle", [20, 40, 10, 10], "", "end"),
        ],
        edges=[
            Edge("e1", "n1", "n2"),
            Edge("e2", "n2", "n3"),
            Edge("e3", "n2", "n4"),
        ],
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


def er() -> Diagram:
    return Diagram(
        id="er",
        diagram_type="er_diagram",
        nodes=[
            Node("e1", "rectangle", [0, 0, 10, 10], "Book", "entity"),
            Node("a1", "ellipse", [20, 0, 10, 10], "isbn", "attribute"),
            Node("e2", "rectangle", [0, 30, 10, 10], "Author", "entity"),
            Node("a2", "ellipse", [20, 30, 10, 10], "name", "attribute"),
        ],
        edges=[Edge("l1", "e1", "a1"), Edge("l2", "a2", "e2")],
    )


def wireframe() -> Diagram:
    return Diagram(
        id="wf",
        diagram_type="wireframe",
        nodes=[
            Node("c0", "rectangle", [0, 0, 100, 100], "", "container"),
            Node("b1", "rectangle", [10, 10, 20, 10], "Send", "ui-button"),
        ],
        edges=[Edge("k1", "c0", "b1", attrs={"kind": "contains"})],
    )


def rules(diagram: Diagram) -> dict[str, V.Violation]:
    return {v.rule: v for v in V.validate(diagram)}


# -- the good cases ------------------------------------------------------------------------


def test_a_well_formed_diagram_of_each_type_has_no_violations():
    for diagram in (flowchart(), machine(), er(), wireframe()):
        assert V.validate(diagram) == [], diagram.diagram_type


def test_an_unrecognised_diagram_type_is_not_an_error_it_is_no_opinion():
    """`unknown` is a legal `diagram_type`; refusing to validate is the right answer for it."""
    diagram = flowchart()
    diagram.diagram_type = "unknown"
    assert V.validate(diagram) == []


# -- flowchart -----------------------------------------------------------------------------


def test_a_flowchart_with_two_starts_names_both_of_them_in_refs():
    """The 158 hdbpmn pages with several triggers: a repair must see all of them, not a count."""
    diagram = flowchart()
    diagram.nodes.append(Node("n5", "circle", [30, 0, 10, 10], "also", "start"))
    violation = rules(diagram)["FC.START_ONE"]
    assert set(violation.refs) == {"n1", "n5"}
    assert violation.severity == V.WARNING


def test_a_flowchart_with_no_start_also_fires_start_one_with_empty_refs():
    diagram = flowchart()
    diagram.nodes[0].semantic_role = "process"
    assert rules(diagram)["FC.START_ONE"].refs == ()


def test_a_flowchart_with_no_end_fires_end_one():
    diagram = flowchart()
    for node in diagram.nodes:
        if node.semantic_role == "end":
            node.semantic_role = "process"
    assert "FC.END_ONE" in rules(diagram)


def test_a_decision_with_one_way_out_and_one_way_in_is_a_split_violation():
    diagram = flowchart()
    diagram.edges = [e for e in diagram.edges if e.id != "e3"]
    found = rules(diagram)
    assert found["FC.DECISION_BRANCH"].refs == ("n2",)
    assert found["FC.DECISION_SPLIT"].refs == ("n2",)
    assert found["FC.DECISION_SPLIT"].severity == V.ERROR


def test_a_merging_gateway_trips_the_plans_rule_but_not_the_qualified_one():
    """The finding: 516 of 541 thin hdbpmn decisions are merges, and merges are not defects."""
    diagram = flowchart()
    diagram.edges = [e for e in diagram.edges if e.id != "e3"]
    diagram.nodes.append(Node("n5", "circle", [30, 0, 10, 10], "", "process"))
    diagram.edges.append(Edge("e4", "n5", "n2"))
    found = rules(diagram)
    assert "FC.DECISION_BRANCH" in found
    assert "FC.DECISION_SPLIT" not in found


# -- state machine -------------------------------------------------------------------------


def test_an_automaton_with_no_initial_state_fires_initial_one():
    diagram = machine()
    diagram.nodes[0].semantic_role = "state"
    assert rules(diagram)["SM.INITIAL_ONE"].severity == V.ERROR


def test_an_unreachable_state_is_named():
    diagram = machine()
    diagram.edges = [e for e in diagram.edges if e.id != "t2"]
    assert rules(diagram)["SM.REACHABLE"].refs == ("q2",)


def test_reachability_follows_edge_direction_and_not_mere_adjacency():
    """q2 -> q1 does not make q2 reachable; a state machine is a directed graph."""
    diagram = machine()
    diagram.edges = [Edge("t1", "q0", "q1"), Edge("t2", "q2", "q1")]
    assert rules(diagram)["SM.REACHABLE"].refs == ("q2",)


# -- er and wireframe ----------------------------------------------------------------------


def test_an_entity_with_no_attribute_is_named():
    diagram = er()
    diagram.edges = [e for e in diagram.edges if e.id != "l2"]
    assert rules(diagram)["ER.ENTITY_ATTRS"].refs == ("e2",)


def test_an_attribute_counts_in_either_edge_direction():
    """`l2` points attribute -> entity, and that entity is still attributed."""
    assert "ER.ENTITY_ATTRS" not in rules(er())


def test_a_widget_with_no_containment_edge_is_an_orphan():
    diagram = wireframe()
    diagram.nodes.append(Node("b2", "rectangle", [40, 10, 20, 10], "Cancel", "ui-button"))
    assert rules(diagram)["WF.ORPHAN_WIDGET"].refs == ("b2",)


def test_a_container_with_no_parent_is_the_root_and_not_an_orphan():
    assert "WF.ORPHAN_WIDGET" not in rules(wireframe())


# -- evidence gating, which is what the corpus sweep turns on -------------------------------


def test_a_diagram_whose_roles_are_all_unknown_offers_no_role_evidence():
    """didi and flowchartseg, 4,319 files: no roles were ever assigned, so nothing is asked."""
    diagram = flowchart()
    for node in diagram.nodes:
        node.semantic_role = "unknown"
    assert V.ROLES not in V.evidence(diagram)
    assert V.validate(diagram) == []


def test_a_diagram_with_no_edges_offers_no_edge_evidence():
    """flowchartseg carries nodes and zero edges by construction."""
    diagram = flowchart()
    diagram.edges = []
    assert V.EDGES not in V.evidence(diagram)
    assert "FC.DECISION_BRANCH" not in rules(diagram)


def test_a_rule_without_its_evidence_is_skipped_rather_than_passed():
    diagram = flowchart()
    diagram.edges = []
    row = V.score(diagram)
    assert "FC.DECISION_BRANCH" in row["skipped"]
    assert "FC.DECISION_BRANCH" not in row["evaluable"]
    assert "FC.START_ONE" in row["evaluable"]


def test_reachability_is_guarded_on_there_being_somewhere_to_start_from():
    """The 53 fa_bresler automata with no initial state cannot pass a reachability check."""
    diagram = machine()
    diagram.nodes[0].semantic_role = "state"
    row = V.score(diagram)
    assert "SM.REACHABLE" in row["skipped"]
    assert "SM.INITIAL_ONE" in row["evaluable"]


def test_a_rule_belonging_to_another_type_is_neither_evaluable_nor_skipped():
    """A wireframe rule is not 'skipped' on a flowchart; it was never in scope."""
    row = V.score(flowchart())
    assert "WF.ORPHAN_WIDGET" not in row["skipped"] + row["evaluable"]


def test_a_rule_that_was_never_asked_reports_a_null_rate_and_not_zero():
    """ER has a denominator of zero on this corpus and must not render as a clean pass."""
    tally = V.Tally()
    tally.add(V.score(flowchart()))
    rows = {r["rule"]: r for r in tally.rows()}
    assert rows["ER.ENTITY_ATTRS"]["rate"] is None
    assert rows["FC.START_ONE"]["rate"] == 0.0


def test_every_registered_rule_has_a_validator_that_can_emit_it():
    """The registry and the validators must not drift; a rule with no code is a lie."""
    emitted = set()
    for broken in _broken_examples():
        emitted |= {v.rule for v in V.validate(broken)}
    assert emitted == set(V.RULES)


def _broken_examples() -> list[Diagram]:
    a = flowchart()
    a.nodes[0].semantic_role = "process"  # no start, no end untouched
    b = flowchart()
    b.edges = [e for e in b.edges if e.id != "e3"]  # thin split decision
    c = flowchart()
    for node in c.nodes:
        if node.semantic_role == "end":
            node.semantic_role = "process"
    d = machine()
    d.nodes[0].semantic_role = "state"
    e = machine()
    e.edges = [x for x in e.edges if x.id != "t2"]
    f = er()
    f.edges = [x for x in f.edges if x.id != "l2"]
    g = wireframe()
    g.nodes.append(Node("b2", "rectangle", [40, 10, 20, 10], "Cancel", "ui-button"))
    return [a, b, c, d, e, f, g]


def test_a_violation_serialises_to_a_json_able_dict():
    violation = V.Violation("FC.START_ONE", V.WARNING, ("n1", "n5"), "2 start nodes, expected 1")
    assert violation.to_dict() == {
        "rule": "FC.START_ONE",
        "severity": V.WARNING,
        "refs": ["n1", "n5"],
        "detail": "2 start nodes, expected 1",
    }


def test_assembled_skips_cleanly_when_the_sibling_module_is_absent():
    """10.2.1 must not depend on 10.1 having landed."""
    result = V.assembled()
    assert "skipped" in result or "rules" in result
