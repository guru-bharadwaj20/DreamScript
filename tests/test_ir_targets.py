"""Phase 2.2.6 - target code pairs.

The emitters are checked by *running* what they produce. A generated program that only parses
proves the emitter produced Python; one that runs, and accepts the right strings, proves it
produced the diagram.
"""

from __future__ import annotations

import pytest

from src.ir import targets
from src.ir.model import Diagram, Edge, Node


def flowchart() -> Diagram:
    return Diagram(
        id="fc",
        diagram_type="flowchart",
        nodes=[
            Node("s", "circle", [0, 0, 10, 10], "order received", "start"),
            Node("d", "diamond", [0, 20, 10, 10], 'is it "urgent"?', "decision"),
            Node("p", "rectangle", [0, 40, 10, 10], "ship today", "process"),
            Node("e", "circle", [0, 60, 10, 10], "done", "end"),
        ],
        edges=[
            Edge("e1", "s", "d", True, ""),
            Edge("e2", "d", "p", True, "yes"),
            Edge("e3", "d", "e", True, "no"),
            Edge("e4", "p", "e", True, ""),
        ],
        meta={"source": "hdbpmn", "geometry": "annotated", "image_size": [100, 100]},
    )


def automaton() -> Diagram:
    return Diagram(
        id="fa",
        diagram_type="state_machine",
        nodes=[
            Node("q0", "circle", [0, 0, 10, 10], "q0", "initial-state"),
            Node("q1", "double-circle", [0, 20, 10, 10], "q1", "final-state"),
        ],
        edges=[
            Edge("t1", "q0", "q1", True, "a"),
            Edge("t2", "q1", "q1", True, "a,b"),
        ],
        meta={"source": "fa_bresler", "geometry": "derived", "image_size": [100, 100]},
    )


def run_generated(code: str) -> dict:
    namespace: dict = {"__name__": "generated"}
    exec(compile(code, "<generated>", "exec"), namespace)  # the check itself
    return namespace


# -- slugging -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("Ship today", "ship_today"),
        ("  Check   counter ", "check_counter"),
        ("2nd attempt", "step_2nd_attempt"),
        ("class", "class_"),
        ("!!!", "step"),
        ("", "step"),
    ],
)
def test_slug(text, want):
    assert targets.slug(text) == want


def test_duplicate_step_names_are_made_unique():
    d = flowchart()
    d.nodes[2].text = "order received"  # same handwriting as the start node
    module = run_generated(targets.emit_flowchart_python(d))
    assert len({f.__name__ for f in module["STEPS"].values()}) == len(module["STEPS"])


# -- flowchart ------------------------------------------------------------------------


def test_flowchart_target_runs_and_follows_the_arrows():
    module = run_generated(targets.emit_flowchart_python(flowchart()))
    state = module["run"]()
    assert state["trace"][0] == "order_received"
    assert "is_it_urgent" in state["decisions"]


def test_flowchart_target_honours_a_branch_choice():
    module = run_generated(targets.emit_flowchart_python(flowchart()))
    taken = module["run"](choose=lambda node, options: dict(options).get("no", options[0][1]))
    assert "ship_today" not in taken["trace"]
    both = module["run"](choose=lambda node, options: dict(options).get("yes", options[0][1]))
    assert "ship_today" in both["trace"]


def test_quotes_in_handwriting_do_not_break_the_generated_file():
    """14 of 693 real files failed to compile before the docstring used repr."""
    d = flowchart()
    d.nodes[1].text = 'he said "no" \\ then left'
    run_generated(targets.emit_flowchart_python(d))


def test_flowchart_target_terminates_on_a_cycle():
    d = flowchart()
    d.edges.append(Edge("e5", "e", "s", True, ""))  # a loop back to the start
    module = run_generated(targets.emit_flowchart_python(d))
    assert module["run"](max_steps=50) is not None


# -- state machine --------------------------------------------------------------------


def test_automaton_target_accepts_and_rejects_correctly():
    module = run_generated(targets.emit_state_machine_python(automaton()))
    assert module["accepts"]("a")
    assert module["accepts"]("aab")
    assert not module["accepts"]("")
    assert not module["accepts"]("b")


def test_a_comma_separated_edge_label_becomes_two_transitions():
    """People write `a,b` on one arrow and mean two symbols."""
    module = run_generated(targets.emit_state_machine_python(automaton()))
    assert set(module["TRANSITIONS"]["q1"]) == {"a", "b"}
    assert module["ALPHABET"] == ["a", "b"]


def test_automaton_without_a_start_state_rejects_everything():
    d = automaton()
    d.nodes[0].semantic_role = "state"
    module = run_generated(targets.emit_state_machine_python(d))
    assert module["START"] is None
    assert not module["accepts"]("a")


# -- selection ------------------------------------------------------------------------


def test_sources_without_structure_get_no_target():
    d = flowchart()
    d.meta["source"] = "didi"
    assert targets.emit(d) is None
    assert "didi" in targets.EXCLUDED


def test_every_excluded_source_states_a_reason():
    for source, reason in targets.EXCLUDED.items():
        assert source not in targets.EMITTERS
        assert len(reason) > 40


def test_every_emitter_language_has_an_extension():
    for language, _ in targets.EMITTERS.values():
        assert language in targets.EXTENSION
