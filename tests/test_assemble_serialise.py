"""Phase 10.2.4 - the round trip against the loader, and the escaping that actually breaks."""

from __future__ import annotations

import json

import pytest

from src.assemble import serialise as S
from src.assemble.serialise import IR_DIR
from src.ir.model import Diagram, Edge, Node
from src.ir.vocab import SHAPES

CORPUS_FILES = S.corpus_files(IR_DIR)


def diagram(shape: str = "rectangle", text: str = "hi") -> Diagram:
    nodes = [
        Node(id="a", shape=shape, bbox=None, text=text),
        Node(id="b", shape=shape, bbox=None, text=text),
    ]
    edges = [Edge(id="e0", src="a", dst="b", label=text)]
    return Diagram(id="t", diagram_type="flowchart", nodes=nodes, edges=edges)


# -- the round trip, against the loader, not just against json ----------------------------


@pytest.mark.parametrize("path", CORPUS_FILES, ids=lambda p: p.name)
def test_every_real_corpus_file_round_trips_through_the_loader(path):
    """`json.loads(json.dumps(x)) == x` is not the bar - 9.3.7 passed that and still broke
    `Diagram.from_dict`. This goes all the way back to a `Diagram` and re-validates."""
    raw = path.read_bytes()
    loaded = Diagram.from_dict(json.loads(raw.decode("utf-8")))
    result = S.round_trip(loaded, original=raw)
    assert result.ok, f"{path.name}: {result.note or result.problems}"


def test_corpus_has_the_expected_five_sources_and_file_count():
    sources = {p.parent.name for p in CORPUS_FILES}
    assert sources == {"didi", "fa_bresler", "flowchartseg", "hdbpmn", "sketch2code"}
    assert len(CORPUS_FILES) == 5796


def test_byte_instability_in_the_corpus_is_only_ever_crlf_line_endings():
    """The measured 125 unstable files (all hdbpmn) are crlf-on-disk, not a content change -
    confirmed here rather than only asserted in the docstring."""
    unstable_reasons = set()
    for path in CORPUS_FILES:
        raw = path.read_bytes()
        loaded = Diagram.from_dict(json.loads(raw.decode("utf-8")))
        first = loaded.to_dict()
        text = json.dumps(first, indent=2, ensure_ascii=False)
        expected = (text + "\n").encode("utf-8")
        if raw != expected:
            unstable_reasons.add(S.byte_reason(raw, expected))
    assert unstable_reasons == {"crlf-line-endings"}


# -- the 9.3.7-shaped regression: json-identical, loader-rejected --------------------------


def test_a_dict_that_round_trips_through_json_can_still_break_the_loader():
    """The exact 9.3.7 shape: `low_conf_text` written as bare id strings instead of the schema
    objects the loader expects. `json.loads(json.dumps(x)) == x` holds; `Diagram.from_dict`
    must not silently accept it."""
    bad = {
        "ir_version": "1.0",
        "id": "regress",
        "diagram_type": "flowchart",
        "nodes": [],
        "edges": [],
        "unresolved_edges": [],
        "crossed_out": [],
        "low_conf_text": ["node-1", "node-2"],  # should be [{"id": ..., ...}, ...]
        "meta": {},
    }
    text = json.dumps(bad)
    assert json.loads(text) == bad  # the check that used to be the whole test
    with pytest.raises((TypeError, ValueError)):
        Diagram.from_dict(json.loads(text))


def test_a_schema_invalid_diagram_is_caught_by_round_trip_even_when_the_loader_accepts_it():
    """A `Diagram` the loader builds without error but the schema rejects - the second half of
    the bug class, where `from_dict` succeeds and `problems()` is the only thing that notices."""
    bad = diagram()
    bad.diagram_type = "not-a-real-diagram-type"
    result = S.round_trip(bad)
    assert result.loads
    assert not result.schema_valid
    assert not result.ok


# -- escaping: the adversarial label set, verified as output -------------------------------


@pytest.mark.parametrize("label", S.ADVERSARIAL, ids=lambda s: repr(s)[:40])
def test_every_adversarial_label_produces_structurally_valid_dot(label):
    d = diagram(text=label)
    problems = S.dot_problems_of(S.to_dot(d))
    assert problems == [], problems


@pytest.mark.parametrize("label", S.ADVERSARIAL, ids=lambda s: repr(s)[:40])
def test_every_adversarial_label_produces_structurally_valid_mermaid(label):
    d = diagram(text=label)
    problems = S.mermaid_problems_of(S.to_mermaid(d))
    assert problems == [], problems


@pytest.mark.parametrize("shape", SHAPES)
def test_every_shape_with_every_adversarial_label_is_covered_by_the_corpus_sweep(shape):
    """`adversarial_check()` crosses all 24 labels with all 12 shapes; this pins that the sweep
    that ran during `run()` actually found zero failures, not merely that it ran."""
    report = S.adversarial_check()
    assert report["shapes_per_label"] == len(SHAPES)
    assert report["dot_failures"] == []
    assert report["mermaid_failures"] == []


def test_dot_binary_check_reports_unavailable_rather_than_faking_a_pass():
    """If `dot` is not on PATH, the check must say so - never claim a verification that did not
    run."""
    import shutil

    ok, note = S.dot_binary_check("digraph g { a -> b; }\n")
    if shutil.which("dot") is None:
        assert ok is None
        assert "not" in note.lower() or "PATH" in note
    else:
        assert ok is True


def test_quotes_already_escaped_in_a_label_do_not_close_the_dot_string_early():
    """Backslash must be escaped before the quote, or `\\"` in a hand-drawn label becomes `\\\\"`
    and closes the string early."""
    escaped = S.dot_label('a \\" already escaped')
    problems = S.dot_problems_of(f'digraph g {{ a [label="{escaped}"]; }}\n')
    assert problems == []


def test_an_empty_label_is_never_emitted_as_a_bare_empty_mermaid_bracket():
    """`A[""]` is a parse error in several Mermaid versions; an empty label must become
    something non-empty."""
    assert S.mermaid_label("") != ""
    d = diagram(text="")
    assert '""' not in S.to_mermaid(d) or S.mermaid_label("") not in ('""', "")


def test_a_mermaid_link_token_inside_a_label_is_defused():
    text = S.to_mermaid(diagram(text="a --> b"))
    assert S.mermaid_problems_of(text) == []


def test_a_dot_keyword_as_a_label_still_produces_valid_dot():
    d = diagram(text="digraph")
    assert S.dot_problems_of(S.to_dot(d)) == []


# -- previews: shape coverage and the size cap ---------------------------------------------


def test_every_ir_shape_has_a_dot_mapping():
    assert set(SHAPES) <= set(S.DOT_SHAPES)


def test_every_ir_shape_has_a_mermaid_bracket_mapping():
    assert set(SHAPES) <= set(S.MERMAID_BRACKETS)


def test_capping_a_preview_reports_how_many_nodes_were_cut():
    nodes = [Node(id=f"n{i}", shape="rectangle", bbox=None, text=f"n{i}") for i in range(10)]
    d = Diagram(id="big", diagram_type="flowchart", nodes=nodes, edges=[])
    dot = S.to_dot(d, cap=3)
    assert "+7 more" in dot
    mermaid = S.to_mermaid(d, cap=3)
    assert "+7 more" in mermaid


def test_an_edge_touching_a_cut_node_is_dropped_rather_than_dangling():
    nodes = [Node(id=f"n{i}", shape="rectangle", bbox=None, text="x") for i in range(5)]
    edges = [Edge(id="e", src="n0", dst="n4")]
    d = Diagram(id="cut", diagram_type="flowchart", nodes=nodes, edges=edges)
    dot = S.to_dot(d, cap=2)
    assert "->" not in dot.split("\n", 4)[-1] or "n4" not in dot


def test_ids_are_made_dot_and_mermaid_safe_and_stay_stable_within_one_emit():
    seen: dict[str, str] = {}
    first = S.safe_id("1-bad id!", seen)
    second = S.safe_id("1-bad id!", seen)
    assert first == second
    assert first[0].isalpha() or first[0] == "_"
    assert S._ID_SAFE.search(first) is None


def test_two_different_raw_ids_never_collide_after_sanitising():
    seen: dict[str, str] = {}
    a = S.safe_id("a!", seen)
    b = S.safe_id("a?", seen)
    assert a != b


# -- the size distribution and whether a preview needs a node cap --------------------------


def test_the_measured_node_cap_is_below_the_largest_diagrams_in_the_corpus():
    """The corpus contains 200-node sketch2code pages; MERMAID_NODE_CAP (60) must actually cap
    something, not sit above every real diagram."""
    sizes = [len(Diagram.load(p).nodes) for p in CORPUS_FILES if p.parent.name == "sketch2code"]
    if not sizes:
        pytest.skip("needs the sketch2code IR from the DVC payload; run `dvc pull`")
    assert max(sizes) > S.MERMAID_NODE_CAP


def test_percentiles_of_an_empty_list_do_not_raise():
    assert S._percentiles([]) == {}
