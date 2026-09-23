"""Phase 14.3 - the ladder is only attributable if the rungs differ in exactly one thing.

So the tests check the two mechanics that could quietly break that: the text transplant (does
`gold_structure` really carry the predicted words and nothing else of the prediction?) and the
arithmetic of the drops. The rest is the report's shape - a rung that cannot be emitted has to
count as a failure rather than vanish from the denominator.
"""

from __future__ import annotations

from src.eval import propagate
from src.ir.model import Diagram, Edge, Node


def diagram(texts, *, offset: float = 0.0, kind: str = "flowchart") -> Diagram:
    nodes = [
        Node(id=f"n{i}", shape="rectangle", bbox=[i * 100.0 + offset, 0.0, 80.0, 40.0], text=text)
        for i, text in enumerate(texts)
    ]
    edges = [
        Edge(id=f"e{i}", src=f"n{i}", dst=f"n{i + 1}", directed=True) for i in range(len(nodes) - 1)
    ]
    return Diagram(id="d", diagram_type=kind, nodes=nodes, edges=edges)


def test_transplant_copies_the_partners_text_and_keeps_the_structure():
    gold = diagram(["read order", "check stock", "ship"])
    predicted = diagram(["read ordev", "check stack", "ship"], offset=2.0)

    merged = propagate.transplant_text(gold, predicted)

    assert [node.text for node in merged.nodes] == ["read ordev", "check stack", "ship"]
    assert [node.id for node in merged.nodes] == [node.id for node in gold.nodes]
    assert len(merged.edges) == len(gold.edges)
    assert [node.text for node in gold.nodes][0] == "read order"  # the input is untouched


def test_an_unmatched_node_gets_no_text_rather_than_an_invented_one():
    gold = diagram(["read order", "check stock"])
    predicted = Diagram(id="p", diagram_type="flowchart", nodes=[gold.nodes[0]], edges=[])

    merged = propagate.transplant_text(gold, predicted)

    assert merged.nodes[0].text == "read order"
    assert merged.nodes[1].text == ""


def test_transplant_is_symmetric_in_direction_but_not_in_meaning():
    """`gold_text` keeps the predicted graph; only the words come from the truth."""
    gold = diagram(["a", "b", "c"])
    predicted = diagram(["x", "y"], offset=1.0)

    merged = propagate.transplant_text(predicted, gold)

    assert len(merged.nodes) == 2
    assert [node.text for node in merged.nodes] == ["a", "b"]


def _row(page, **rungs):
    return {
        "page": page,
        "source": "hdbpmn",
        "rungs": {
            name: {"emitted": True, "functional": value, "reason": "equal" if value else "differ"}
            for name, value in rungs.items()
        },
    }


def test_drops_are_measured_against_the_gold_rung():
    rows = [
        _row("a", gold=True, gold_structure=True, gold_text=False, predicted=False),
        _row("b", gold=True, gold_structure=False, gold_text=True, predicted=False),
        _row("c", gold=False, gold_structure=False, gold_text=False, predicted=False),
        _row("d", gold=True, gold_structure=True, gold_text=True, predicted=True),
    ]
    summary = propagate.summarise(rows)

    assert summary["pass_rate"] == {
        "gold": 0.75,
        "gold_structure": 0.5,
        "gold_text": 0.5,
        "predicted": 0.25,
    }
    assert summary["attributable_drop"] == {
        "text_only": 0.25,
        "structure_only": 0.25,
        "both": 0.5,
    }
    assert summary["pages"] == 4


def test_a_rung_that_could_not_be_emitted_counts_as_a_failure():
    rows = [_row("a", gold=True, gold_structure=True, gold_text=True, predicted=True)]
    rows[0]["rungs"]["predicted"] = {"emitted": False, "functional": False, "reason": "no_emission"}
    summary = propagate.summarise(rows)
    assert summary["pass_rate"]["predicted"] == 0.0
    assert summary["failure_reasons"]["predicted"] == {"no_emission": 1}


def test_a_skipped_page_leaves_the_denominator_rather_than_scoring_zero():
    rows = [
        _row("a", gold=True, gold_structure=True, gold_text=True, predicted=True),
        {"page": "b", "source": "hdbpmn", "skipped": "emitter refused", "rungs": {}},
    ]
    summary = propagate.summarise(rows)
    assert summary["pages"] == 1
    assert summary["pass_rate"]["gold"] == 1.0


def test_per_source_rates_are_reported_separately():
    rows = [
        _row("a", gold=True, predicted=True),
        {**_row("b", gold=True, predicted=False), "source": "fa_bresler"},
    ]
    summary = propagate.summarise(rows)
    assert summary["by_source"]["hdbpmn"]["predicted"] == 1.0
    assert summary["by_source"]["fa_bresler"]["predicted"] == 0.0


def test_render_names_every_rung_and_its_inputs():
    rows = [_row("a", gold=True, gold_structure=False, gold_text=False, predicted=False)]
    text = propagate.render(
        {"split": "test", "seconds": 1.0, **propagate.summarise(rows), "rows": rows}
    )
    for rung in propagate.RUNGS:
        assert f"`{rung}`" in text
    assert "attributable drop" in text


def test_emit_returns_none_rather_than_raising_on_a_graph_it_cannot_answer():
    broken = Diagram(id="d", diagram_type="not_a_type", nodes=[], edges=[])
    assert propagate.emit(broken, "hdbpmn") is None
