"""Phase 12.1.6 - the five target emitters, and the diagram shapes that used to break them.

Three of these are regression tests for bugs the module shipped with, each found by running the
emitter over the whole corpus rather than over a fixture:

    a suite of comments        a BPMN branch containing only an intermediate event emitted
                               `if cond(ctx):` followed by a comment line, which is a non-empty
                               list of lines and an empty Python suite. 9 of 693 hdbpmn pages.
    a SQL reserved word        `_ident` guarded Python's keywords, not SQL's, so an entity called
                               "order" emitted `CREATE TABLE order (...)`. 77 of 200 synthetic
                               ER diagrams.
    a net soldered at one end  a disconnected circuit put two parts on nets nothing else touched.
                               40 of 200 synthetic circuits.

Everything is checked with the checkers the repo already has - `src.codegen.quality` for
Python/React/SPICE and `src.eval.sql.check` for DDL - never with a private check written here,
so a test passing means the pair would survive 12.1.7's filter.

The corpus-wide tests skip rather than fail when `data/processed/ir` is absent; the synthetic and
hand-written ones need no dataset at all.
"""

from __future__ import annotations

import ast

import pytest

from src.codegen import quality, targets
from src.eval import sql
from src.parse.sequences import IR_DIR, load_ir, traversal
from src.synth import graphs

needs_corpus = pytest.mark.skipif(not IR_DIR.is_dir(), reason="no IR corpus in this checkout")

DIAGRAM_TYPES = ("flowchart", "state_machine", "er", "wireframe", "circuit")


def emit(diagram: dict, diagram_type: str) -> str:
    order, _ = traversal(diagram)
    return targets.for_type(diagram_type)(diagram, order)


def synthetic(diagram_type: str, count: int = 40) -> list[dict]:
    return list(graphs.generate(count, diagram_types=(diagram_type,)))


# ------------------------------------------------------------------------------------------
# dispatch
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("diagram_type", DIAGRAM_TYPES)
def test_for_type_returns_a_two_argument_emitter_for_each_of_the_five(diagram_type):
    generator = targets.for_type(diagram_type)
    assert callable(generator)
    assert isinstance(generator(synthetic(diagram_type, 1)[0], []), str)


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [("bpmn", "flowchart"), ("state-machine", "state_machine"), ("erd", "er"), ("ui", "wireframe")],
)
def test_aliases_dispatch_to_the_same_generator(alias, canonical):
    assert targets.for_type(alias) is targets.for_type(canonical)


def test_an_unknown_diagram_type_raises_rather_than_defaulting_to_python():
    with pytest.raises(KeyError):
        targets.for_type("sankey")


def test_every_language_name_is_one_the_quality_filter_knows():
    """A pair labelled with a language `quality.check` does not know is rejected by the gate."""
    assert set(targets.LANGUAGES.values()) <= set(quality.CHECKS)
    assert set(targets.LANGUAGES) == set(DIAGRAM_TYPES)


# ------------------------------------------------------------------------------------------
# identifiers out of OCR text
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "   ", None, "3 items", "class", "match", "!!!", "日本語", "a" * 400, "Do the thing?"],
)
def test_ident_is_always_a_usable_python_identifier(text):
    name = targets._ident(text)
    assert name.isidentifier()
    ast.parse(f"{name} = 1")  # an identifier that is also a keyword would fail here


@pytest.mark.parametrize("text", ["order", "group", "table", "select", "default", "index"])
def test_sql_ident_escapes_sql_reserved_words_that_are_ordinary_english(text):
    """Regression: `CREATE TABLE order (...)` is a syntax error, and `order` is a real entity."""
    name = targets._sql_ident(text)
    assert name.lower() not in targets._SQL_KEYWORDS
    assert sql.check(f"CREATE TABLE {name} (\n    {name}_id INTEGER PRIMARY KEY\n);")["ok"]


# ------------------------------------------------------------------------------------------
# the hostile diagrams: what a real page actually contains
# ------------------------------------------------------------------------------------------


def node(node_id, role="process", text="", **attrs):
    return {
        "id": node_id,
        "shape": "rectangle",
        "bbox": [0.0, 0.0, 10.0, 10.0],
        "text": text,
        "semantic_role": role,
        "confidence": 1.0,
        "source_id": node_id,
        "attrs": attrs,
    }


def edge(edge_id, src, dst, label="", **attrs):
    return {
        "id": edge_id,
        "src": src,
        "dst": dst,
        "directed": True,
        "label": label,
        "polyline": None,
        "confidence": 1.0,
        "source_id": edge_id,
        "attrs": attrs,
    }


HOSTILE: dict[str, dict] = {
    "empty": {"id": "empty", "nodes": [], "edges": []},
    "no_edges": {"id": "no_edges", "nodes": [node("a"), node("b")], "edges": []},
    "self_loop": {"id": "self_loop", "nodes": [node("a")], "edges": [edge("e0", "a", "a")]},
    "cycle": {
        "id": "cycle",
        "nodes": [node("a", "start"), node("b"), node("c", "decision")],
        "edges": [edge("e0", "a", "b"), edge("e1", "b", "c"), edge("e2", "c", "a")],
    },
    "disconnected": {
        "id": "disconnected",
        "nodes": [node("a"), node("b"), node("c"), node("d")],
        "edges": [edge("e0", "a", "b"), edge("e1", "c", "d")],
    },
    "unresolved_edges": {
        "id": "unresolved_edges",
        "nodes": [node("a"), node("b")],
        "edges": [
            edge("e0", None, "b"),
            edge("e1", "a", None),
            edge("e2", None, None),
            edge("e3", "a", "ghost"),
        ],
    },
    "garbled_ocr": {
        "id": "garbled_ocr",
        "nodes": [
            node("a", "process", ""),
            node("b", "process", "|||~~"),
            node("c", "process", "class"),
            node("d", "process", "3 items"),
            node("e", "process", "日本語"),
        ],
        "edges": [edge("e0", "a", "b"), edge("e1", "b", "c"), edge("e2", "c", "d")],
    },
    "duplicate_labels": {
        "id": "duplicate_labels",
        "nodes": [node("a", "state", "q"), node("b", "state", "q"), node("c", "state", "q")],
        "edges": [edge("e0", "a", "b", "x"), edge("e1", "b", "c", "y")],
    },
    "out_degree_five": {
        "id": "out_degree_five",
        "nodes": [node("a", "fork")] + [node(f"n{i}") for i in range(5)],
        "edges": [edge(f"e{i}", "a", f"n{i}") for i in range(5)],
    },
    "out_degree_five_decision": {
        "id": "out_degree_five_decision",
        "nodes": [node("a", "decision")] + [node(f"n{i}") for i in range(5)],
        "edges": [edge(f"e{i}", "a", f"n{i}") for i in range(5)],
    },
    "no_id": {"nodes": [node("a")], "edges": []},
    "deep_containment": {
        "id": "deep_containment",
        "nodes": [node(f"w{i}", "container") for i in range(60)],
        "edges": [edge(f"c{i}", f"w{i}", f"w{i + 1}", kind="contains") for i in range(59)],
    },
    "containment_cycle": {
        "id": "containment_cycle",
        "nodes": [node("a", "container"), node("b", "container")],
        "edges": [
            edge("c0", "a", "b", kind="contains"),
            edge("c1", "b", "a", kind="contains"),
        ],
    },
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
@pytest.mark.parametrize("diagram_type", DIAGRAM_TYPES)
def test_no_emitter_crashes_on_a_hostile_diagram(diagram_type, name):
    assert isinstance(emit(HOSTILE[name], diagram_type), str)


@pytest.mark.parametrize("name", sorted(HOSTILE))
@pytest.mark.parametrize("diagram_type", DIAGRAM_TYPES)
def test_every_hostile_diagram_still_produces_output_that_passes_its_checker(diagram_type, name):
    ok, detail = targets.verify(emit(HOSTILE[name], diagram_type), diagram_type)
    assert ok, f"{diagram_type}/{name}: {detail}"


# ------------------------------------------------------------------------------------------
# 1-2. Python: flowchart and state machine
# ------------------------------------------------------------------------------------------


def test_a_branch_whose_only_content_is_a_comment_gets_a_pass():  # regression, 9/693 hdbpmn
    """`if c:` followed by a bare comment is a non-empty line list and an empty Python suite."""
    diagram = {
        "id": "comment_branch",
        "nodes": [
            node("d", "decision", "ok?"),
            node("left", "event", "customer notified"),
            node("right", "process", "organize disbursement"),
            node("join", "join", ""),
        ],
        "edges": [
            edge("e0", "d", "left"),
            edge("e1", "d", "right"),
            edge("e2", "left", "join"),
            edge("e3", "right", "join"),
        ],
    }
    code, mode = targets.flowchart_mode(diagram, traversal(diagram)[0])
    assert mode == "structured"
    assert "pass" in code
    ast.parse(code)


def test_flowchart_mode_reports_the_dispatch_fallback_rather_than_hiding_it():
    """An *exclusive* choice of five ways has no if/else form and still falls back."""
    _, mode = targets.flowchart_mode(
        HOSTILE["out_degree_five_decision"], traversal(HOSTILE["out_degree_five_decision"])[0]
    )
    assert mode == "dispatch"


def test_a_five_way_fork_keeps_its_structure_instead_of_falling_back():
    """A parallel split of any width is emitted as its branches in sequence.

    It used to raise `_Irreducible` on out-degree 3+, which sent every BPMN pool to the dispatch
    fallback - and that fallback could only route to two of the five, leaving the operations on
    the other three on no path through the program at all.
    """
    diagram = HOSTILE["out_degree_five"]
    code, mode = targets.flowchart_mode(diagram, traversal(diagram)[0])
    assert mode == "structured"
    # The fixture's five branch nodes carry no text, so each emits the same `step(ctx)` call -
    # what matters is that there are five of them and not two.
    assert code.count("ctx = step(ctx)") == 5


def test_a_reducible_flowchart_keeps_its_loop_and_does_not_fall_back():
    diagram = {
        "id": "loop",
        "nodes": [node("h", "decision", "more?"), node("b", "process", "consume")],
        "edges": [edge("e0", "h", "b"), edge("e1", "b", "h")],
    }
    code, mode = targets.flowchart_mode(diagram, traversal(diagram)[0])
    assert mode == "structured"
    assert "while " in code


@pytest.mark.parametrize("diagram_type", ["flowchart", "state_machine"])
def test_synthetic_python_targets_compile_not_merely_parse(diagram_type):
    for diagram in synthetic(diagram_type):
        code = emit(diagram, diagram_type)
        ok, kind, detail = quality.check_python(code)
        assert ok, f"{kind}: {detail}"


def test_the_emitted_state_machine_class_actually_runs():
    diagram = {
        "id": "dfa",
        "nodes": [
            node("s0", "initial-state", "q0", initial=True),
            node("s1", "final-state", "q1", accepting=True),
        ],
        "edges": [edge("e0", "s0", "s1", "a"), edge("e1", "s1", "s0", "b")],
    }
    namespace: dict = {}
    exec(compile(emit(diagram, "state_machine"), "<machine>", "exec"), namespace)  # noqa: S102
    machine = namespace["DfaMachine"]()
    assert machine.INITIAL == "q0"
    assert machine.accepts(["a"]) is True
    assert machine.accepts(["a", "b"]) is False
    with pytest.raises(ValueError):
        machine.step("zzz")


def test_a_machine_with_no_accepting_state_gets_an_empty_set_not_an_empty_dict():
    """Regression: `ACCEPTING = {}` is a dict, and a dict of accepting states is meaningless."""
    namespace: dict = {}
    code = emit(HOSTILE["duplicate_labels"], "state_machine")
    exec(compile(code, "<machine>", "exec"), namespace)  # noqa: S102
    machine = next(v for k, v in namespace.items() if k.endswith("Machine"))
    assert isinstance(machine.ACCEPTING, set)  # `{}` would be a dict
    assert not machine.ACCEPTING


def test_states_whose_ocr_text_collided_do_not_merge_into_one_row():
    """Three states all read as "q"; the transition table must still have three rows."""
    code = emit(HOSTILE["duplicate_labels"], "state_machine")
    namespace: dict = {}
    exec(compile(code, "<machine>", "exec"), namespace)  # noqa: S102
    machine = next(v for k, v in namespace.items() if k.endswith("Machine"))
    assert len(machine.STATES) == len(set(machine.STATES)) == 3
    assert len(machine.TABLE) == 3


def test_the_transitions_library_is_not_imported_by_the_emitted_code():
    """`transitions` is NOT installed here; an emitted import of it would never run."""
    code = emit(synthetic("state_machine", 1)[0], "state_machine")
    assert "import transitions" not in code
    assert "TRANSITIONS" in code and "STATES" in code


# ------------------------------------------------------------------------------------------
# 3. SQL DDL
# ------------------------------------------------------------------------------------------


def test_synthetic_er_targets_satisfy_the_committed_sql_checker():
    for diagram in synthetic("er", 60):
        result = sql.check(emit(diagram, "er"))
        assert result["ok"], f"{result['kind']}: {result['detail']}"


def test_er_output_is_a_real_schema_and_not_an_empty_one_that_scores_100():
    """`sql.check` rejects an empty schema; this asserts the emitter is above that floor too."""
    tables = relationships = junctions = 0
    for diagram in synthetic("er", 60):
        result = sql.check(emit(diagram, "er"))
        tables += len(result["tables"])
        relationships += len(result["relationships"])
        junctions += len(result["junction_tables"])
    assert tables >= 120 and relationships >= 60 and junctions >= 1


def test_an_n_m_relationship_becomes_a_junction_table_the_checker_recognises():
    diagram = {
        "id": "nm",
        "nodes": [
            node("E0", "entity", "student"),
            node("E1", "entity", "course"),
            node("R0", "relationship", "enrolment", cardinality="n-m"),
        ],
        "edges": [edge("a", "E0", "R0", "1"), edge("b", "R0", "E1", "n")],
    }
    result = sql.check(emit(diagram, "er"))
    assert result["ok"], result["detail"]
    assert [j["cardinality"] for j in result["junction_tables"]] == ["many-to-many"]


def test_a_hallucinated_column_type_is_narrowed_rather_than_emitted():
    diagram = {
        "id": "types",
        "nodes": [node("E0", "entity", "thing", columns=[["a", "BANANA"], ["b", "INTEGER"]])],
        "edges": [],
    }
    ddl = emit(diagram, "er")
    assert "BANANA" not in ddl
    assert sql.check(ddl)["ok"]


# ------------------------------------------------------------------------------------------
# 4. React + Tailwind
# ------------------------------------------------------------------------------------------


def test_synthetic_wireframe_targets_pass_the_structural_jsx_check():
    for diagram in synthetic("wireframe", 60):
        ok, kind, detail = quality.check_react(emit(diagram, "wireframe"))
        assert ok, f"{kind}: {detail}"


def test_the_component_is_a_single_default_export_with_tailwind_classes():
    code = emit(synthetic("wireframe", 1)[0], "wireframe")
    assert code.count("export default function") == 1
    assert "className=" in code and "min-h-screen" in code


def test_an_unknown_html_tag_is_not_passed_through_into_the_jsx():
    diagram = {
        "id": "tags",
        "nodes": [node("a", "container", "", html_tag="<script>alert(1)</script>")],
        "edges": [],
    }
    code = emit(diagram, "wireframe")
    assert "script" not in code
    assert quality.check_react(code)[0]


def test_a_containment_cycle_terminates_instead_of_recursing_forever():
    code = emit(HOSTILE["containment_cycle"], "wireframe")
    assert quality.check_react(code)[0]


# ------------------------------------------------------------------------------------------
# 5. SPICE
# ------------------------------------------------------------------------------------------


def test_synthetic_circuit_targets_pass_the_documented_structural_check():
    for diagram in synthetic("circuit", 60):
        ok, kind, detail = quality.check_spice(emit(diagram, "circuit"))
        assert ok, f"{kind}: {detail}"


def test_a_disconnected_circuit_has_its_orphan_nets_terminated():  # regression, 40/200
    """A net on exactly one card is a wire soldered at one end; the checker rejects the deck."""
    disconnected = graphs.random_diagram("circuit", "disconnected", 0)
    netlist = emit(disconnected, "circuit")
    ok, kind, detail = quality.check_spice(netlist)
    assert ok, f"{kind}: {detail}"
    assert "R_TERM" in netlist


def test_every_netlist_has_a_ground_a_source_and_an_end_card():
    for diagram in synthetic("circuit", 20) + [HOSTILE["empty"]]:
        netlist = emit(diagram, "circuit")
        lines = netlist.strip().splitlines()
        assert lines[-1] == ".end"
        assert any(line[:1].upper() == "V" for line in lines)
        assert any(" 0 " in line or line.endswith(" 0") for line in lines)


# ------------------------------------------------------------------------------------------
# the corpus
# ------------------------------------------------------------------------------------------


@needs_corpus
@pytest.mark.parametrize(
    ("source", "diagram_type"),
    [
        ("hdbpmn", "flowchart"),
        ("fa_bresler", "state_machine"),
        ("sketch2code", "wireframe"),
    ],
)
def test_every_real_diagram_of_a_backed_type_emits_code_that_checks(source, diagram_type):
    """The three types with real diagrams behind them. ER and circuit have none in this repo."""
    diagrams = load_ir([source])
    assert diagrams, f"no {source} IR"
    for diagram in diagrams:
        ok, detail = targets.verify(emit(diagram, diagram_type), diagram_type)
        assert ok, f"{source}/{diagram['id']}: {detail}"


@needs_corpus
def test_the_hdbpmn_fallback_rate_is_the_one_the_docstring_records():
    """63.6% dispatch on 693 pages. A drift either way means the emitter changed silently.

    Was 71.4% until a fork of out-degree 3+ stopped being refused; the 54 pages that moved are
    parallel gateways whose branches the fallback could not all reach.
    """
    modes = {"structured": 0, "dispatch": 0}
    for diagram in load_ir(["hdbpmn"]):
        modes[targets.flowchart_mode(diagram, traversal(diagram)[0])[1]] += 1
    total = sum(modes.values())
    assert total == 693
    assert 0.61 <= modes["dispatch"] / total <= 0.66
